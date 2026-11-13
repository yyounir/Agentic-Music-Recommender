"""Generation back-ends for the agent.

Two engines implement the same two-method interface:

``GeminiEngine``
    Calls the Google Gemini API through ``google-genai`` with Pydantic response
    schemas. Used whenever ``GEMINI_API_KEY`` is set.

``OfflineHeuristicEngine``
    A dependency-free, fully deterministic stand-in that performs the same two
    jobs (intent parsing and playlist curation) with rule-based logic. It exists
    so that unit tests, the evaluation harness and the ``--demo`` run are
    reproducible and runnable without a key or network access. It is a real part
    of the system, not a mock: the agent, the retriever and every guardrail run
    unchanged on top of it.

``FaultInjectingEngine``
    Wraps either engine and deliberately corrupts the first *n* curation
    responses. It is used by the guardrail demonstration and the retry tests to
    exercise the self-correction loop on demand.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from src import prompts
from src.retrieval import PROJECT_ROOT, EditorialKnowledgeBase
from src.schemas import CuratedPlaylist, GuardrailReport, Plan, RetrievalResult


def load_dotenv(path: Optional[Path] = None) -> List[str]:
    """Load ``KEY=VALUE`` pairs from the project's ``.env`` into ``os.environ``.

    Variables already present in the real environment always win, so an explicit
    shell export or a CI secret is never shadowed by a stale file. Uses
    ``python-dotenv`` when installed and falls back to a minimal parser so the
    project still works without it. ``.env`` is gitignored.
    """
    env_path = path or (PROJECT_ROOT / ".env")
    if not env_path.exists():
        return []

    try:  # preferred: handles quoting and `export` prefixes
        from dotenv import dotenv_values

        pairs = dotenv_values(env_path)
    except ImportError:  # minimal fallback
        pairs = {}
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            pairs[key.strip().removeprefix("export ").strip()] = value.strip().strip(
                "\"'"
            )

    loaded: List[str] = []
    for key, value in pairs.items():
        if value is not None and key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


load_dotenv()

# Override with GEMINI_MODEL in .env or the shell. Model ids come and go: list the
# ones your key can serve with `client.models.list()`. gemini-2.5-flash is no
# longer served to new keys, and the newest flash ids are frequently rate-limited,
# so MODEL_FALLBACKS gives the engine somewhere to go rather than failing the run.
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
MODEL_FALLBACKS = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"]

# Transport-level retry. These are *not* self-correction retries: a 503 is the
# absence of an answer, not a bad answer, so it must never consume the guardrail
# correction budget.
TRANSIENT_STATUS = (429, 500, 502, 503, 504)
TRANSPORT_RETRIES = 3
TRANSPORT_BACKOFF_SECONDS = 2.0


# --------------------------------------------------------------------------- #
# Engine interface
# --------------------------------------------------------------------------- #
@dataclass
class CurationRequest:
    """Everything an engine needs to produce (or repair) a playlist."""

    query: str
    plan: Plan
    retrieval: RetrievalResult
    candidate_block: str
    feedback: Optional[GuardrailReport] = None
    previous: Optional[CuratedPlaylist] = None
    attempt: int = 1


class Engine(Protocol):
    provider: str
    model: str

    def plan(self, query: str, vocabulary: Dict[str, Any]) -> str: ...

    def curate(self, request: CurationRequest) -> str: ...


def extract_json(text: str) -> Dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Tolerates ```json fences and leading prose, both of which models emit even
    when told not to.
    """
    if text is None:
        raise ValueError("Model returned no text")
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
        depth, end = 0, None
        for index in range(start, len(cleaned)):
            if cleaned[index] == "{":
                depth += 1
            elif cleaned[index] == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            raise ValueError(f"Unbalanced JSON in model output: {text[:200]!r}")
        parsed = json.loads(cleaned[start:end])
    if not isinstance(parsed, dict):
        raise ValueError("Model output was not a JSON object")
    return parsed


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
class GeminiEngine:
    """Structured-output wrapper around google-genai."""

    provider = "gemini"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        try:
            from google import genai  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "google-genai is not installed. Run `pip install -r requirements.txt`."
            ) from exc
        from google import genai

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.model = model or DEFAULT_GEMINI_MODEL
        self._client = genai.Client(api_key=key)
        # Models to try, in order, starting with the configured one.
        self._candidates: List[str] = [self.model] + [
            name for name in MODEL_FALLBACKS if name != self.model
        ]
        # `self.model` is the *configured* model and is never mutated: every call
        # restarts the chain at the preferred model so a temporary outage does not
        # permanently downgrade the engine. What actually answered is recorded
        # per call instead.
        self.models_used: List[str] = []
        self.transport_notices: List[str] = []

    def drain_notices(self) -> List[str]:
        """Return and clear notices accumulated since the last drain.

        Notices are per-request, not per-engine: without draining, a fallback
        that happened on an earlier query would be reported again on every
        later one.
        """
        notices, self.transport_notices = self.transport_notices, []
        return notices

    @staticmethod
    def _status_code(exc: BaseException) -> Optional[int]:
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if isinstance(code, int):
            return code
        match = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
        return int(match.group(1)) if match else None

    def _generate(
        self,
        system_instruction: str,
        contents: List[Any],
        response_schema: Any,
        temperature: float,
    ) -> str:
        """Call Gemini, retrying transient failures and falling back on model ids.

        Raises the last error only when every model in the chain is exhausted;
        the agent then treats that as a failed attempt and degrades as usual.
        """
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=temperature,
        )

        last_error: Optional[BaseException] = None
        for model_name in self._candidates:
            for attempt in range(TRANSPORT_RETRIES):
                try:
                    response = self._client.models.generate_content(
                        model=model_name, contents=contents, config=config
                    )
                    self.models_used.append(model_name)
                    if model_name != self.model:
                        self.transport_notices.append(
                            f"The configured model '{self.model}' was unavailable "
                            f"for this request; it was answered by '{model_name}'."
                        )
                    return response.text or ""
                except Exception as exc:  # noqa: BLE001 - classified below
                    last_error = exc
                    status = self._status_code(exc)
                    if status in TRANSIENT_STATUS and attempt < TRANSPORT_RETRIES - 1:
                        time.sleep(TRANSPORT_BACKOFF_SECONDS * (2**attempt))
                        continue
                    break  # permanent error, or retries exhausted: next model
        raise RuntimeError(
            f"Gemini call failed on every candidate model "
            f"({', '.join(self._candidates)}): {last_error}"
        ) from last_error

    def plan(self, query: str, vocabulary: Dict[str, Any]) -> str:
        return self._generate(
            prompts.build_planner_prompt(query, vocabulary),
            [prompts.build_planner_user_message(query)],
            Plan,
            temperature=0.0,
        )

    def curate(self, request: CurationRequest) -> str:
        from google.genai import types

        contents: List[Any] = []
        # Few-shot turns: the persona is taught by example, not description alone.
        for example in prompts.CURATOR_FEW_SHOT:
            contents.append(
                types.Content(
                    role="user", parts=[types.Part(text=example["input"])]
                )
            )
            contents.append(
                types.Content(
                    role="model", parts=[types.Part(text=example["output"])]
                )
            )

        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=prompts.build_curator_user_message(
                            request.query,
                            request.plan,
                            request.retrieval,
                            request.candidate_block,
                        )
                    )
                ],
            )
        )
        if request.feedback is not None:
            contents.append(
                types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=(
                                request.previous.model_dump_json()
                                if request.previous is not None
                                else "{}"
                            )
                        )
                    ],
                )
            )
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=prompts.build_correction_message(
                                request.feedback,
                                request.previous,
                                request.candidate_block,
                            )
                        )
                    ],
                )
            )

        return self._generate(
            prompts.CURATOR_SYSTEM_PROMPT,
            contents,
            CuratedPlaylist,
            temperature=0.3 if request.feedback is None else 0.0,
        )


# --------------------------------------------------------------------------- #
# Offline deterministic engine
# --------------------------------------------------------------------------- #
# Everyday words mapped onto the catalog's closed mood vocabulary.
_MOOD_SYNONYMS: Dict[str, str] = {
    "happy": "happy", "joyful": "happy", "cheerful": "happy", "upbeat": "happy",
    "sunny": "happy", "feel-good": "happy", "chill": "chill", "chilled": "chill",
    "laid-back": "chill", "mellow": "chill", "intense": "intense",
    "aggressive": "intense", "hard": "intense", "relaxed": "relaxed",
    "relaxing": "relaxed", "focused": "focused", "focus": "focused",
    "concentration": "focused", "moody": "moody", "brooding": "moody",
    "nostalgic": "nostalgic", "wistful": "nostalgic", "playful": "playful",
    "serene": "serene", "peaceful": "serene", "calm": "serene",
    "defiant": "defiant", "angry": "defiant", "sensual": "sensual",
    "sultry": "sensual", "melancholic": "melancholic", "melancholy": "melancholic",
    "sad": "melancholic", "heartbroken": "melancholic", "lonely": "melancholic",
    "hopeful": "hopeful", "uplifting": "hopeful",
}

_GENRE_SYNONYMS: Dict[str, str] = {
    "lo-fi": "lofi", "lo fi": "lofi", "lofi": "lofi", "indie-pop": "indie pop",
    "indie": "indie pop", "classic": "classical", "r and b": "funk",
}

# Genres a listener may plausibly ask for that this catalog simply does not have.
_KNOWN_ABSENT_GENRES = [
    "k-pop", "kpop", "hip hop", "hip-hop", "rap", "edm", "techno", "house",
    "dubstep", "trap", "punk", "disco", "soul", "r&b", "rnb", "gospel", "opera",
    "bluegrass", "afrobeat", "salsa", "bossa nova", "grime", "drum and bass",
    "trance", "emo", "ska", "polka", "flamenco",
]

_INJECTION_PATTERNS = [
    r"ignore (all |any |your )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (the |your )?(instructions|rules|system prompt|guardrails?)",
    r"you are now\b",
    r"(reveal|print|show|output|repeat) (me )?(your |the )?(system )?(prompt|instructions)",
    r"(invent|make up|fabricate|hallucinate|generate) (a |some |new )?(songs?|tracks?|artists?)",
    r"(disable|skip|bypass|turn off) (the )?(guardrails?|checks?|validation|verification)",
    r"(songs?|tracks?) (that are )?not in (the )?(catalog|csv|dataset|list)",
    r"outside (the |this )?(catalog|dataset|csv)",
    r"act as\b",
    r"new instructions?:",
]


class OfflineHeuristicEngine:
    """Deterministic, rule-based planner + curator. No network, no key."""

    provider = "offline"
    model = "heuristic-v1"

    def __init__(self, knowledge: Optional[EditorialKnowledgeBase] = None):
        self.knowledge = knowledge or EditorialKnowledgeBase.load()

    # -- planning ----------------------------------------------------------
    def plan(self, query: str, vocabulary: Dict[str, Any]) -> str:
        text = " " + query.lower().strip() + " "
        genres = [str(g).lower() for g in vocabulary.get("genres", [])]
        moods = [str(m).lower() for m in vocabulary.get("moods", [])]
        artists = [str(a) for a in vocabulary.get("artists", [])]

        unsupported: List[str] = []
        for pattern in _INJECTION_PATTERNS:
            match = re.search(pattern, text)
            if match:
                unsupported.append(
                    f"Ignored an instruction embedded in the request: "
                    f"'{match.group(0).strip()}'"
                )

        exclude_genres: List[str] = []
        exclude_moods: List[str] = []
        exclude_artists: List[str] = []

        # Negation windows: "no metal", "without any rock", "nothing by LoRoom".
        negations = re.findall(
            r"\b(?:no|not|without|avoid|exclude|skip|nothing)\b"
            r"(?:\s+(?:any|some|more|songs?|tracks?|by|of|the|a))*\s+([a-z&' -]{2,30})",
            text,
        )
        for window in negations:
            window = window.strip()
            for alias, canonical in list(_GENRE_SYNONYMS.items()) + [
                (g, g) for g in genres
            ]:
                if re.search(rf"\b{re.escape(alias)}\b", window):
                    if canonical not in exclude_genres:
                        exclude_genres.append(canonical)
            for alias, canonical in _MOOD_SYNONYMS.items():
                if canonical in moods and re.search(rf"\b{re.escape(alias)}\b", window):
                    if canonical not in exclude_moods:
                        exclude_moods.append(canonical)
            for artist in artists:
                if artist.lower() in window and artist.lower() not in exclude_artists:
                    exclude_artists.append(artist.lower())

        # Positive genre / mood mentions (excluding negated ones).
        chosen_genres: List[str] = []
        for alias, canonical in list(_GENRE_SYNONYMS.items()) + [(g, g) for g in genres]:
            if canonical not in genres or canonical in exclude_genres:
                continue
            if re.search(rf"\b{re.escape(alias)}\b", text) and canonical not in chosen_genres:
                negated = re.search(
                    rf"\b(?:no|not|without|avoid|exclude|skip)\b[^.;,]{{0,20}}?\b{re.escape(alias)}\b",
                    text,
                )
                if not negated:
                    chosen_genres.append(canonical)

        chosen_moods: List[str] = []
        for alias, canonical in _MOOD_SYNONYMS.items():
            if canonical not in moods or canonical in exclude_moods:
                continue
            if re.search(rf"\b{re.escape(alias)}\b", text) and canonical not in chosen_moods:
                chosen_moods.append(canonical)

        # Listening contexts come straight from the editorial context notes.
        contexts = [
            note["context"] for note in self.knowledge.match_contexts([query])
        ]

        # Genres the listener asked for that this catalog does not carry.
        for absent in _KNOWN_ABSENT_GENRES:
            if re.search(rf"\b{re.escape(absent)}\b", text):
                unsupported.append(
                    f"The catalog has no {absent} tracks; the request was answered "
                    "from the genres that are available."
                )

        min_bpm, max_bpm = self._parse_bpm(text)
        min_energy, max_energy = self._parse_energy(text)
        acoustic = self._parse_acoustic(text)
        track_count = self._parse_count(text)

        rationale_bits: List[str] = []
        if contexts:
            rationale_bits.append(f"context: {', '.join(contexts)}")
        if chosen_genres:
            rationale_bits.append(f"genres: {', '.join(chosen_genres)}")
        if chosen_moods:
            rationale_bits.append(f"moods: {', '.join(chosen_moods)}")
        if min_bpm is not None or max_bpm is not None:
            rationale_bits.append(f"tempo: {min_bpm or '-'}..{max_bpm or '-'} BPM")
        if exclude_genres or exclude_artists or exclude_moods:
            rationale_bits.append(
                "exclusions: "
                + ", ".join(exclude_genres + exclude_moods + exclude_artists)
            )

        payload = {
            "moods": chosen_moods,
            "genres": chosen_genres,
            "listening_contexts": contexts,
            "exclude_genres": exclude_genres,
            "exclude_moods": exclude_moods,
            "exclude_artists": exclude_artists,
            "min_bpm": min_bpm,
            "max_bpm": max_bpm,
            "min_energy": min_energy,
            "max_energy": max_energy,
            "acoustic_preference": acoustic,
            "track_count": track_count,
            "rationale": "Parsed request as " + ("; ".join(rationale_bits) or "an open request with no hard constraints"),
            "unsupported_requests": unsupported,
        }
        return json.dumps(payload)

    @staticmethod
    def _parse_bpm(text: str) -> tuple[Optional[int], Optional[int]]:
        between = re.search(
            r"\b(?:between|from)\s+(\d{2,3})\s*(?:bpm)?\s*(?:and|to|-|–)\s*(\d{2,3})\s*bpm",
            text,
        )
        if between:
            return int(between.group(1)), int(between.group(2))
        dashed = re.search(r"\b(\d{2,3})\s*(?:-|–|to)\s*(\d{2,3})\s*bpm", text)
        if dashed:
            return int(dashed.group(1)), int(dashed.group(2))

        min_bpm: Optional[int] = None
        max_bpm: Optional[int] = None
        upper = re.search(
            r"\b(?:under|below|less than|slower than|at most|max(?:imum)?|up to|no (?:more|faster) than)\s+(\d{2,3})\s*bpm",
            text,
        )
        if upper:
            max_bpm = int(upper.group(1))
        lower = re.search(
            r"\b(?:over|above|at least|more than|faster than|min(?:imum)?|starting at)\s+(\d{2,3})\s*bpm",
            text,
        )
        if lower:
            min_bpm = int(lower.group(1))
        if min_bpm is None and max_bpm is None:
            around = re.search(r"\b(?:around|about|near|roughly|~)\s*(\d{2,3})\s*bpm", text)
            if around:
                centre = int(around.group(1))
                return max(centre - 8, 0), centre + 8
            bare = re.search(r"\b(\d{2,3})\s*bpm", text)
            if bare:
                centre = int(bare.group(1))
                return max(centre - 8, 0), centre + 8
        return min_bpm, max_bpm

    @staticmethod
    def _parse_energy(text: str) -> tuple[Optional[float], Optional[float]]:
        if re.search(r"\b(high[- ]energy|energetic|high energy)\b", text):
            return 0.7, None
        if re.search(r"\b(low[- ]energy|very quiet|barely there|sleepy)\b", text):
            return None, 0.45
        return None, None

    @staticmethod
    def _parse_acoustic(text: str) -> str:
        if re.search(r"\b(no|not|without|avoid)\s+(\w+\s+){0,2}acoustic", text):
            return "avoid"
        if re.search(r"\b(electronic only|synth only|nothing acoustic)\b", text):
            return "avoid"
        if re.search(r"\b(acoustic|unplugged|organic|instrumental and acoustic)\b", text):
            return "prefer"
        return "any"

    @staticmethod
    def _parse_count(text: str) -> int:
        words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a dozen": 12,
        }
        # The count and the noun may be separated by adjectives ("3 K-pop and trap
        # songs") or a hyphen ("4-song playlist"). The gap deliberately excludes
        # digits and "bpm" so that a tempo bound is never read as a track count.
        item = r"(?:songs?|tracks?|picks?|cuts?)"
        gap = r"(?:(?!\bbpm\b)[^.,;:!?\d]){0,30}?"
        numeric = re.search(rf"\b(\d{{1,2}}){gap}{item}\b", text)
        if numeric:
            return max(1, min(int(numeric.group(1)), 20))
        topn = re.search(r"\btop\s+(\d{1,2})\b", text)
        if topn:
            return max(1, min(int(topn.group(1)), 20))
        for word, value in words.items():
            if re.search(rf"\b{word}\b{gap}{item}\b", text):
                return value
        return 5

    # -- curation ----------------------------------------------------------
    def curate(self, request: CurationRequest) -> str:
        plan = request.plan
        candidates = list(request.retrieval.candidates)
        chosen = candidates[: plan.track_count]

        tracks = []
        for candidate in chosen:
            tracks.append(
                {
                    "song_id": candidate.song_id,
                    "title": candidate.title,
                    "artist": candidate.artist,
                    "reason": self._reason_for(candidate, plan),
                }
            )

        name = self._playlist_name(plan)
        summary = self._summary(plan, request.retrieval, chosen)
        return json.dumps(
            {"playlist_name": name, "summary": summary, "tracks": tracks}
        )

    def _reason_for(self, candidate: Any, plan: Plan) -> str:
        bits: List[str] = [
            f"{candidate.genre} at {candidate.tempo_bpm} BPM with "
            f"{candidate.energy:.2f} energy and {candidate.valence:.2f} valence"
        ]
        if plan.listening_contexts:
            contexts = self.knowledge.track_contexts(candidate.song_id)
            overlap = [
                context
                for context in contexts
                if any(
                    word in context
                    for want in plan.listening_contexts
                    for word in want.split()
                )
            ]
            if overlap:
                bits.append(f"editorially filed under {overlap[0]}")
        descriptors = self.knowledge.track_descriptors(candidate.song_id)
        if descriptors:
            bits.append(f"reads as {descriptors[0]}")
        if plan.acoustic_preference == "prefer" and candidate.acousticness >= 0.6:
            bits.append(f"acousticness {candidate.acousticness:.2f}")
        if plan.acoustic_preference == "avoid" and candidate.acousticness <= 0.3:
            bits.append(f"acousticness only {candidate.acousticness:.2f}")
        note = (candidate.editorial_note or "").split(".")[0]
        sentence = "; ".join(bits).capitalize() + "."
        if note:
            sentence += f" Editorial note: {note}."
        return sentence

    @staticmethod
    def _playlist_name(plan: Plan) -> str:
        if plan.listening_contexts:
            return plan.listening_contexts[0].title() + " Set"
        if plan.moods and plan.genres:
            return f"{plan.moods[0].title()} {plan.genres[0].title()}"
        if plan.genres:
            return f"{plan.genres[0].title()} Selection"
        if plan.moods:
            return f"{plan.moods[0].title()} Mix"
        return "Custom Mix"

    @staticmethod
    def _summary(plan: Plan, retrieval: RetrievalResult, chosen: List[Any]) -> str:
        if not chosen:
            return "No catalog track satisfied the stated constraints."
        low = min(c.tempo_bpm for c in chosen)
        high = max(c.tempo_bpm for c in chosen)
        genres = sorted({c.genre for c in chosen})
        summary = (
            f"{len(chosen)} track(s) spanning {low}-{high} BPM across "
            f"{', '.join(genres)}, ranked by catalog features blended with "
            "retrieved editorial notes."
        )
        if retrieval.relaxations:
            summary += " " + retrieval.relaxations[0]
        return summary


# --------------------------------------------------------------------------- #
# Fault injection (guardrail demos + retry tests)
# --------------------------------------------------------------------------- #
@dataclass
class FaultInjectingEngine:
    """Wraps an engine and corrupts its first ``faults`` curation responses.

    ``mode`` selects what kind of failure to simulate:

    * ``"hallucination"`` - inserts a track that is not in the catalog.
    * ``"constraint"``    - inserts a candidate that violates a stated exclusion
      (falling back to over-returning tracks if no violating candidate exists).
    """

    inner: Any
    faults: int = 1
    mode: str = "hallucination"
    calls: int = 0
    provider: str = field(init=False, default="")
    model: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.provider = f"{self.inner.provider}+fault-injection"
        self.model = self.inner.model

    def reset(self) -> None:
        """Restore the fault budget, so each evaluation case starts corrupted."""
        self.calls = 0

    def plan(self, query: str, vocabulary: Dict[str, Any]) -> str:
        return self.inner.plan(query, vocabulary)

    def curate(self, request: CurationRequest) -> str:
        raw = self.inner.curate(request)
        self.calls += 1
        if self.calls > self.faults:
            return raw
        payload = extract_json(raw)
        tracks = payload.get("tracks", [])
        if self.mode == "hallucination":
            tracks.insert(
                0,
                {
                    "song_id": 999,
                    "title": "Neon Cathedral",
                    "artist": "The Phantom Orchestra",
                    "reason": "A perfect fit for this mood.",
                },
            )
            payload["tracks"] = tracks[: max(len(tracks) - 1, 1)]
        else:
            offender = self._find_constraint_offender(request)
            if offender is None:
                # Nothing in the catalog violates the plan, so simulate the other
                # common constraint failure instead: returning too many tracks.
                payload["tracks"] = tracks + tracks[:1]
            else:
                tracks.insert(0, offender)
                payload["tracks"] = tracks[: max(len(tracks) - 1, 1)]
        return json.dumps(payload)

    @staticmethod
    def _find_constraint_offender(
        request: CurationRequest,
    ) -> Optional[Dict[str, Any]]:
        """Pick a real catalog row that breaks one of the plan's hard constraints.

        The offender is deliberately drawn from the *full* catalog rather than
        the candidate pool, because deterministic filtering has already removed
        violating tracks from that pool - which is exactly the layering this
        demo is meant to show.
        """
        from src.retrieval import get_catalog

        plan = request.plan
        catalog = get_catalog()
        excluded_genres = {g.lower() for g in plan.exclude_genres}
        excluded_artists = {a.lower() for a in plan.exclude_artists}

        for _, row in catalog.iterrows():
            genre = str(row["genre"]).lower()
            artist = str(row["artist"]).lower()
            tempo = int(row["tempo_bpm"])
            breaks = (
                genre in excluded_genres
                or artist in excluded_artists
                or (plan.min_bpm is not None and tempo < plan.min_bpm)
                or (plan.max_bpm is not None and tempo > plan.max_bpm)
            )
            if breaks:
                return {
                    "song_id": int(row["id"]),
                    "title": str(row["title"]),
                    "artist": str(row["artist"]),
                    "reason": (
                        f"Ignoring the listener's constraint because "
                        f"{row['title']} has {float(row['energy']):.2f} energy."
                    ),
                }
        return None


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def build_engine(
    prefer: str = "auto",
    knowledge: Optional[EditorialKnowledgeBase] = None,
) -> tuple[Any, List[str]]:
    """Pick an engine. ``prefer`` is ``"auto"``, ``"gemini"`` or ``"offline"``.

    Returns ``(engine, notices)``; notices explain any downgrade so the UI and
    the trace log can show which back-end actually answered.
    """
    notices: List[str] = []
    if prefer == "offline":
        return OfflineHeuristicEngine(knowledge), notices

    has_key = bool(
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )
    if prefer == "gemini" or (prefer == "auto" and has_key):
        try:
            return GeminiEngine(), notices
        except Exception as exc:  # noqa: BLE001 - any failure must degrade, not crash
            notices.append(
                f"Gemini engine unavailable ({exc}); using the deterministic "
                "offline engine instead."
            )
            return OfflineHeuristicEngine(knowledge), notices

    notices.append(
        "GEMINI_API_KEY is not set; using the deterministic offline engine. "
        "Set the key to run the same pipeline against Gemini."
    )
    return OfflineHeuristicEngine(knowledge), notices
