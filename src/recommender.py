"""Music recommender - baseline scoring engine + agentic curation workflow.

This module has two layers:

1. **Baseline (Module 3 starter).** ``Song``, ``UserProfile``, ``Recommender``,
   ``load_songs``, ``score_song`` and ``recommend_songs`` are preserved exactly
   as the original project defined them. They are a deterministic, weighted
   feature-similarity recommender over ``data/songs.csv`` and are still used as
   a reference implementation and by the original unit tests.

2. **Agentic workflow (this upgrade).** :class:`MusicCuratorAgent` runs a
   four-step loop over the same catalog:

   ``Plan`` -> ``Retrieve`` -> ``Act`` -> ``Verify & Refine``

   Step 1 parses a natural-language request into a Pydantic ``Plan``.
   Step 2 combines deterministic Pandas filtering over ``data/songs.csv`` with
   lexical retrieval over ``data/editorial_notes.json``, blending both into one
   candidate ranking. Step 3 asks a constrained "Music Curator" persona for a
   structured playlist. Step 4 verifies that playlist against the catalog and
   the plan, feeds any violations back for up to three self-corrections, and
   falls back to deterministic top matches if the model never converges.

See ``architecture.mmd`` for the full data flow.
"""

import csv
import math
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

DEFAULT_USER_PROFILE: Dict[str, object] = {
    "favorite_genre": "pop",
    "favorite_mood": "happy",
    "target_energy": 0.8,
    "likes_acoustic": False,
    "target_tempo_bpm": 120,
    "target_valence": 0.8,
    "target_danceability": 0.75,
}


def _finite_number(value: object, label: str) -> float:
    """Validate and convert a feature value to a finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top k songs ranked by preference score."""
        ranked_songs = sorted(
            self.songs,
            key=lambda song: score_song(vars(user), vars(song))[0],
            reverse=True,
        )
        return ranked_songs[:max(k, 0)]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explain how a song matches the user's preferences."""
        _, reasons = score_song(vars(user), vars(song))
        return "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    csv_file_path = Path(csv_path)
    if not csv_file_path.is_absolute() and not csv_file_path.exists():
        project_root_path = Path(__file__).resolve().parent.parent / csv_file_path
        if project_root_path.exists():
            csv_file_path = project_root_path

    with csv_file_path.open(newline="", encoding="utf-8") as csv_file:
        songs = []
        for row in csv.DictReader(csv_file):
            row["id"] = int(row["id"])
            row["tempo_bpm"] = int(row["tempo_bpm"])
            for field in ("energy", "valence", "danceability", "acousticness"):
                row[field] = float(row[field])
            songs.append(row)
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Return a song's preference score and human-readable reasons."""
    score = 0.0
    reasons: List[str] = []

    for preference_keys, song_key, label, points in (
        (("favorite_genre", "genre"), "genre", "Genre", 1.5),
        (("favorite_mood", "mood"), "mood", "Mood", 2.0),
    ):
        preference = next(
            (user_prefs[key] for key in preference_keys
             if user_prefs.get(key) is not None),
            None,
        )
        song_value = song.get(song_key)
        if preference is None or song_value is None:
            continue
        if not isinstance(preference, str) or not isinstance(song_value, str):
            raise TypeError(f"{label} preferences and song values must be strings")
        if preference.strip().casefold() == song_value.strip().casefold():
            score += points
            reasons.append(f"{label} match (+{points:g} points)")
        else:
            reasons.append(f"{label} mismatch (+0 points)")

    numeric_features: Tuple[
        Tuple[Tuple[str, str], str, str, float, float], ...
    ] = (
        (("target_energy", "energy"), "energy", "Energy", 1.0, 4.0),
        (("target_valence", "valence"), "valence", "Valence", 1.0, 2.0),
        (("target_danceability", "danceability"), "danceability", "Danceability", 1.0, 2.0),
        (("target_tempo_bpm", "tempo_bpm"), "tempo_bpm", "Tempo", 120.0, 2.0),
    )
    for preference_keys, song_key, label, distance_scale, max_points in numeric_features:
        target = next(
            (user_prefs[key] for key in preference_keys
             if user_prefs.get(key) is not None),
            None,
        )
        value = song.get(song_key)
        if target is None or value is None:
            continue

        target_value = _finite_number(target, label)
        song_value = _finite_number(value, label)
        if distance_scale == 1.0 and not (
            0.0 <= target_value <= 1.0 and 0.0 <= song_value <= 1.0
        ):
            raise ValueError(f"{label} target and song value must be between 0 and 1")
        if distance_scale != 1.0 and (target_value < 0 or song_value < 0):
            raise ValueError(f"{label} target and song value must be non-negative")

        feature_points = max_points * max(
            0.0, 1.0 - abs(target_value - song_value) / distance_scale
        )
        score += feature_points
        reasons.append(
            f"{label} similarity: {feature_points:.2f}/{max_points:g} points "
            f"(target {target_value:g}, song {song_value:g})"
        )

    likes_acoustic = user_prefs.get("likes_acoustic")
    acousticness = song.get("acousticness")
    if likes_acoustic is not None and acousticness is not None:
        if not isinstance(likes_acoustic, bool):
            raise TypeError("likes_acoustic preference must be a boolean")
        acoustic_value = _finite_number(acousticness, "Acousticness")
        if not 0.0 <= acoustic_value <= 1.0:
            raise ValueError("Song acousticness must be between 0 and 1")
        acoustic_points = float(
            acoustic_value if likes_acoustic else 1.0 - acoustic_value
        )
        score += acoustic_points
        reasons.append(
            f"Acoustic preference similarity: {acoustic_points:.2f}/1 point"
        )

    if not reasons:
        reasons.append("No comparable preferences were provided.")
    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Return the top k songs with scores and explanations."""
    ranked_songs = sorted(
        ((song, *score_song(user_prefs, song)) for song in songs),
        key=lambda recommendation: recommendation[1],
        reverse=True,
    )
    return [
        (song, score, "; ".join(reasons))
        for song, score, reasons in ranked_songs[:max(k, 0)]
    ]


# =========================================================================== #
#                                                                             #
#  AGENTIC LAYER - Plan -> Retrieve -> Act -> Verify & Refine                 #
#                                                                             #
# =========================================================================== #

import time as _time
from typing import Optional

from pydantic import ValidationError as _ValidationError

from src.guardrails import (
    deterministic_fallback,
    to_recommendations,
    verify_playlist,
)
from src.llm_client import CurationRequest, build_engine, extract_json
from src.retrieval import (
    EditorialKnowledgeBase,
    catalog_vocabulary,
    format_candidates_for_prompt,
    get_catalog,
    retrieve_candidates,
)
from src.schemas import (
    MAX_TRACK_COUNT,
    AgentResult,
    AgentTrace,
    AttemptTrace,
    CuratedPlaylist,
    GuardrailReport,
    GuardrailViolation,
    Plan,
)
from src.tracing import append_trace, new_trace_id

MAX_QUERY_LENGTH = 2000
DEFAULT_MAX_RETRIES = 3


class QueryValidationError(ValueError):
    """Raised when a request cannot be processed at all."""


def validate_query(query: object) -> str:
    """Normalise and bounds-check a natural-language request."""
    if query is None or not isinstance(query, str):
        raise QueryValidationError("The listening request must be a string.")
    cleaned = query.strip()
    if not cleaned:
        raise QueryValidationError("The listening request cannot be empty.")
    if len(cleaned) > MAX_QUERY_LENGTH:
        raise QueryValidationError(
            f"The listening request is too long "
            f"({len(cleaned)} characters, limit {MAX_QUERY_LENGTH})."
        )
    return cleaned


class MusicCuratorAgent:
    """Four-step agentic recommender with guardrails and self-correction."""

    def __init__(
        self,
        engine=None,
        catalog=None,
        knowledge=None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        trace_path=None,
        prefer: str = "auto",
    ):
        self.knowledge = knowledge or EditorialKnowledgeBase.load()
        self.catalog = get_catalog() if catalog is None else catalog
        self.vocabulary = catalog_vocabulary(self.catalog, self.knowledge)
        self.max_retries = max(0, int(max_retries))
        self.trace_path = trace_path
        self.startup_notices: List[str] = []
        if engine is None:
            engine, notices = build_engine(prefer=prefer, knowledge=self.knowledge)
            self.startup_notices = notices
        self.engine = engine

    # ------------------------------------------------------------------ #
    # Step 1 - Plan
    # ------------------------------------------------------------------ #
    def plan_request(self, query: str) -> Tuple[Plan, Optional[str]]:
        """Parse a request into a validated ``Plan``.

        A model that returns unusable JSON - or an API that is down, rate-limited
        or serving a retired model id - must not take the whole run down.

        Degrading to an *empty* plan is the dangerous answer: a plan with no
        constraints means the constraint guardrail has nothing to enforce, so the
        agent would confidently return tracks that violate what the listener
        explicitly asked for. (This is exactly what happened when a free-tier
        daily quota killed the planner mid-benchmark: "5 songs above 200 BPM"
        came back with five tracks between 66 and 124 BPM.)

        So planner failure falls back to the deterministic heuristic planner,
        which needs no network and does extract hard constraints. Only if that
        also fails do we accept an unconstrained plan.
        """
        try:
            raw = self.engine.plan(query, self.vocabulary)
            payload = extract_json(raw)
            return Plan.model_validate(payload), None
        except Exception as exc:  # noqa: BLE001 - planner failure must degrade, not crash
            primary_error = f"{type(exc).__name__}: {exc}"

        # Secondary planner: rule-based, always available.
        try:
            from src.llm_client import OfflineHeuristicEngine

            fallback_planner = OfflineHeuristicEngine(self.knowledge)
            payload = extract_json(
                fallback_planner.plan(query, self.vocabulary)
            )
            plan = Plan.model_validate(payload)
            return (
                plan.model_copy(
                    update={
                        "rationale": (
                            "Parsed by the deterministic fallback planner because "
                            f"the primary planner failed. {plan.rationale}"
                        )
                    }
                ),
                f"{primary_error} (recovered with the heuristic planner)",
            )
        except Exception as exc:  # noqa: BLE001 - last resort
            return (
                Plan(
                    rationale="Neither planner could parse this request; answering "
                    "without hard constraints.",
                ),
                f"{primary_error}; fallback planner also failed: "
                f"{type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #
    def recommend(
        self,
        query: str,
        track_count: Optional[int] = None,
        log: bool = True,
    ) -> Tuple[AgentResult, AgentTrace]:
        """Run Plan -> Retrieve -> Act -> Verify and return the result + trace."""
        started = _time.perf_counter()
        cleaned_query = validate_query(query)
        trace_id = new_trace_id()
        models_before = len(getattr(self.engine, "models_used", []))
        notices: List[str] = list(self.startup_notices)

        # ---- Step 1: Plan -------------------------------------------------
        plan, plan_error = self.plan_request(cleaned_query)
        if plan_error:
            notices.append(
                "The primary planner failed; this request was parsed by the "
                "deterministic fallback planner."
                if "recovered with the heuristic planner" in plan_error
                else "Neither planner could parse this request, so it was answered "
                "without hard constraints. Treat these recommendations with caution."
            )
        if track_count is not None:
            plan = plan.model_copy(
                update={"track_count": min(max(int(track_count), 1), MAX_TRACK_COUNT)}
            )
        notices.extend(plan.unsupported_requests)

        # ---- Step 2: Retrieve ---------------------------------------------
        retrieval = retrieve_candidates(
            plan=plan,
            query=cleaned_query,
            catalog=self.catalog,
            knowledge=self.knowledge,
        )
        if retrieval.strict_match_count == 0:
            unmet = "; ".join(retrieval.filters_applied)
            notices.append(
                f"No track in the catalog satisfies these constraints ({unmet}), so "
                "nothing was recommended rather than returning tracks that break the "
                "request."
            )
            if retrieval.near_misses:
                closest = ", ".join(
                    f"{c.title} ({c.genre}, {c.tempo_bpm} BPM, energy {c.energy:.2f})"
                    for c in retrieval.near_misses
                )
                notices.append(f"Closest tracks in the catalog, for reference: {closest}.")
        elif retrieval.strict_match_count < plan.track_count:
            notices.append(
                f"Only {retrieval.strict_match_count} catalog track(s) satisfy every "
                f"constraint, fewer than the {plan.track_count} requested."
            )
        candidate_block = format_candidates_for_prompt(retrieval.candidates)

        # ---- Steps 3 & 4: Act, then Verify & Refine -----------------------
        attempts: List[AttemptTrace] = []
        playlist: Optional[CuratedPlaylist] = None
        report: Optional[GuardrailReport] = None
        feedback: Optional[GuardrailReport] = None
        previous: Optional[CuratedPlaylist] = None
        fallback_used = False

        # If deterministic filtering left nothing, no generated answer could pass
        # verification. Short-circuit instead of burning retries on an
        # unsatisfiable request: the correct answer is an explicit empty result.
        attempt_budget = 0 if not retrieval.candidates else self.max_retries + 1

        for attempt_number in range(1, attempt_budget + 1):
            request = CurationRequest(
                query=cleaned_query,
                plan=plan,
                retrieval=retrieval,
                candidate_block=candidate_block,
                feedback=feedback,
                previous=previous,
                attempt=attempt_number,
            )
            parse_error: Optional[str] = None
            parsed: Optional[CuratedPlaylist] = None
            raw_output = ""
            engine_failed = False
            try:
                raw_output = self.engine.curate(request)
            except Exception as exc:  # noqa: BLE001 - engine failure must not crash the run
                parse_error = f"{type(exc).__name__}: {exc}"
                engine_failed = True
            if parse_error is None:
                try:
                    parsed = CuratedPlaylist.model_validate(extract_json(raw_output))
                except (_ValidationError, ValueError) as exc:
                    parse_error = f"{type(exc).__name__}: {exc}"

            if parsed is None:
                current_report = GuardrailReport(
                    passed=False,
                    checks_run=["format"],
                    violations=[
                        GuardrailViolation(
                            check="format",
                            message=(
                                "Your response was not valid JSON matching the "
                                f"required schema ({parse_error}). Return a single "
                                "JSON object with playlist_name, summary and tracks."
                            ),
                        )
                    ],
                )
            else:
                current_report = verify_playlist(
                    playlist=parsed,
                    plan=plan,
                    retrieval=retrieval,
                    catalog=self.catalog,
                    knowledge=self.knowledge,
                )

            attempts.append(
                AttemptTrace(
                    attempt=attempt_number,
                    prompt_kind="initial" if attempt_number == 1 else "self_correction",
                    raw_model_output=(raw_output or "")[:4000],
                    parsed_playlist=parsed,
                    guardrail=current_report,
                    parse_error=parse_error,
                )
            )

            playlist, report = parsed, current_report
            if current_report.passed and parsed is not None:
                break

            # A transport failure (API down, quota exhausted, every fallback model
            # exhausted) is the absence of an answer, not a bad answer. Re-prompting
            # cannot fix it and each attempt costs the full retry/backoff budget, so
            # go straight to the deterministic fallback instead of making the user
            # wait minutes for three more identical failures.
            if engine_failed:
                notices.append(
                    "The generation API could not be reached, so the playlist was "
                    "built by deterministic ranking instead."
                )
                break

            feedback, previous = current_report, parsed

        retries = max(0, len(attempts) - 1)

        # ---- Explicit empty result for unsatisfiable requests --------------
        if attempt_budget == 0:
            playlist = CuratedPlaylist(
                playlist_name="No Matching Tracks",
                summary=(
                    "This catalog contains no track that satisfies the request. "
                    "Nothing is recommended rather than offering a track that "
                    "breaks a stated constraint."
                ),
                tracks=[],
            )
            report = verify_playlist(
                playlist=playlist,
                plan=plan,
                retrieval=retrieval,
                catalog=self.catalog,
                knowledge=self.knowledge,
            )

        # ---- Graceful fallback --------------------------------------------
        elif report is None or not report.passed or playlist is None:
            fallback_used = True
            playlist = deterministic_fallback(plan, retrieval, self.knowledge)
            report = verify_playlist(
                playlist=playlist,
                plan=plan,
                retrieval=retrieval,
                catalog=self.catalog,
                knowledge=self.knowledge,
            )
            notices.append(
                f"The curator model failed verification {len(attempts)} time(s); "
                "returned deterministic top matches instead."
            )

        recommendations = to_recommendations(playlist, self.catalog, self.knowledge)

        # Transport-level events (model fallbacks, retried 503s) are collected by
        # the engine during the run and surfaced alongside the agent's own notices.
        # They are drained, not read, so one request's fallback is not reported
        # again on every later request.
        drain = getattr(self.engine, "drain_notices", None)
        if callable(drain):
            notices.extend(drain())

        # De-duplicate notices while preserving order.
        seen_notices: List[str] = []
        for notice in notices:
            if notice not in seen_notices:
                seen_notices.append(notice)

        result = AgentResult(
            query=cleaned_query,
            playlist_name=playlist.playlist_name or "Custom Mix",
            summary=playlist.summary,
            recommendations=recommendations,
            plan=plan,
            guardrail=report,
            retries=retries,
            fallback_used=fallback_used,
            provider=getattr(self.engine, "provider", "unknown"),
            notices=seen_notices,
            trace_id=trace_id,
        )

        trace = AgentTrace(
            trace_id=trace_id,
            query=cleaned_query,
            provider=getattr(self.engine, "provider", "unknown"),
            # The model(s) that actually answered this request, which may differ
            # from the configured one if the fallback chain was used.
            model=", ".join(
                dict.fromkeys(getattr(self.engine, "models_used", [])[models_before:])
            )
            or getattr(self.engine, "model", ""),
            plan=plan,
            plan_error=plan_error,
            retrieval=retrieval,
            attempts=attempts,
            retries=retries,
            fallback_used=fallback_used,
            final_output=result,
            duration_ms=int((_time.perf_counter() - started) * 1000),
            notices=seen_notices,
        )

        if log:
            try:
                append_trace(trace, self.trace_path)
            except OSError:  # pragma: no cover - logging must never break a run
                pass

        return result, trace


def recommend_from_query(
    query: str,
    track_count: Optional[int] = None,
    prefer: str = "auto",
) -> Tuple[AgentResult, AgentTrace]:
    """Convenience one-shot entry point used by the CLI and the Streamlit UI."""
    agent = MusicCuratorAgent(prefer=prefer)
    return agent.recommend(query, track_count=track_count)
