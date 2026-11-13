"""Step 2 of the agent: multi-source retrieval.

Two sources are combined:

1. ``data/songs.csv``            - deterministic Pandas filtering on hard constraints
   (genre/mood, BPM bounds, energy bounds, exclusions).
2. ``data/editorial_notes.json`` - a lexical retrieval pass over curation notes.

The editorial pass is not decoration. Its score is folded into ``total_score``
*before* candidates are truncated, and context notes contribute genre
preferences and tempo priors, so the notes change *which* tracks reach the
curator and in what order - not just what the curator says about them.
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.schemas import Candidate, EditorialHit, Plan, RetrievalResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SONGS_PATH = PROJECT_ROOT / "data" / "songs.csv"
DEFAULT_NOTES_PATH = PROJECT_ROOT / "data" / "editorial_notes.json"

NUMERIC_COLUMNS = ("energy", "valence", "danceability", "acousticness")
REQUIRED_COLUMNS = (
    "id",
    "title",
    "artist",
    "genre",
    "mood",
    "energy",
    "tempo_bpm",
    "valence",
    "danceability",
    "acousticness",
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "for",
    "from", "give", "i", "if", "in", "is", "it", "its", "me", "my", "need",
    "of", "on", "or", "please", "song", "songs", "that", "the", "them", "then",
    "this", "to", "track", "tracks", "want", "was", "with", "would", "you",
    "playlist", "some", "something", "make", "build", "put", "together",
}

# Editorial weight relative to feature score. Large enough that a track the
# editorial layer explicitly recommends for a context can overtake a track with
# marginally better raw numbers; small enough that it cannot rescue a track that
# violates the listener's stated preferences.
EDITORIAL_WEIGHT = 2.5


def _resolve(path: Optional[str | Path], default: Path) -> Path:
    if path is None:
        return default
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    rooted = PROJECT_ROOT / candidate
    return rooted if rooted.exists() else candidate


def _tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9']+", str(text).lower())
        if token not in _STOPWORDS and len(token) > 1
    ]


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #
def load_catalog(csv_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load data/songs.csv into a validated, typed DataFrame."""
    resolved = _resolve(csv_path, DEFAULT_SONGS_PATH)
    if not resolved.exists():
        raise FileNotFoundError(f"Song catalog not found: {resolved}")

    frame = pd.read_csv(resolved)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Catalog is missing required columns: {', '.join(missing)}")

    frame = frame.copy()
    frame["id"] = frame["id"].astype(int)
    frame["tempo_bpm"] = frame["tempo_bpm"].astype(int)
    for column in NUMERIC_COLUMNS:
        frame[column] = frame[column].astype(float)
    for column in ("title", "artist", "genre", "mood"):
        frame[column] = frame[column].astype(str).str.strip()
    frame["genre_key"] = frame["genre"].str.lower()
    frame["mood_key"] = frame["mood"].str.lower()
    frame["artist_key"] = frame["artist"].str.lower()
    if frame["id"].duplicated().any():
        raise ValueError("Catalog contains duplicate song ids")
    return frame


@lru_cache(maxsize=4)
def _cached_catalog(resolved: str) -> pd.DataFrame:
    return load_catalog(resolved)


def get_catalog(csv_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Cached catalog accessor; returns a defensive copy."""
    resolved = str(_resolve(csv_path, DEFAULT_SONGS_PATH))
    return _cached_catalog(resolved).copy()


def catalog_vocabulary(
    frame: pd.DataFrame, knowledge: "EditorialKnowledgeBase | None" = None
) -> Dict[str, Any]:
    """The closed vocabularies the planner is allowed to use.

    ``listening_contexts`` is closed too: it comes from the editorial context
    notes, so the planner cannot invent a context that retrieval has no notes for.
    """
    return {
        "catalog_size": int(len(frame)),
        "genres": sorted(frame["genre_key"].unique().tolist()),
        "moods": sorted(frame["mood_key"].unique().tolist()),
        "artists": sorted(frame["artist"].unique().tolist()),
        "contexts": (
            [note["context"] for note in knowledge.context_notes] if knowledge else []
        ),
        "bpm_range": [int(frame["tempo_bpm"].min()), int(frame["tempo_bpm"].max())],
    }


# --------------------------------------------------------------------------- #
# Editorial knowledge base
# --------------------------------------------------------------------------- #
class EditorialKnowledgeBase:
    """Lexical retriever over data/editorial_notes.json."""

    def __init__(self, payload: Dict):
        self.payload = payload
        self.track_notes: Dict[int, Dict] = {
            int(note["song_id"]): note for note in payload.get("track_notes", [])
        }
        self.genre_notes: Dict[str, Dict] = {
            str(note["genre"]).lower(): note for note in payload.get("genre_notes", [])
        }
        self.context_notes: List[Dict] = payload.get("context_notes", [])
        self._track_tokens: Dict[int, List[str]] = {
            song_id: _tokenize(self._track_text(note))
            for song_id, note in self.track_notes.items()
        }

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, notes_path: Optional[str | Path] = None) -> "EditorialKnowledgeBase":
        resolved = _resolve(notes_path, DEFAULT_NOTES_PATH)
        if not resolved.exists():
            raise FileNotFoundError(f"Editorial notes not found: {resolved}")
        with resolved.open(encoding="utf-8") as handle:
            return cls(json.load(handle))

    # -- text views --------------------------------------------------------
    @staticmethod
    def _track_text(note: Dict) -> str:
        return " ".join(
            [
                " ".join(note.get("mood_descriptors", [])),
                " ".join(note.get("listening_contexts", [])),
                note.get("editorial_note", ""),
            ]
        )

    def track_note_text(self, song_id: int) -> str:
        note = self.track_notes.get(int(song_id))
        return note.get("editorial_note", "") if note else ""

    def track_descriptors(self, song_id: int) -> List[str]:
        note = self.track_notes.get(int(song_id))
        return list(note.get("mood_descriptors", [])) if note else []

    def track_contexts(self, song_id: int) -> List[str]:
        note = self.track_notes.get(int(song_id))
        return list(note.get("listening_contexts", [])) if note else []

    def avoid_when(self, song_id: int) -> List[str]:
        note = self.track_notes.get(int(song_id))
        return list(note.get("avoid_when", [])) if note else []

    # -- retrieval ---------------------------------------------------------
    def match_contexts(self, query_terms: Sequence[str]) -> List[Dict]:
        """Context notes whose name or aliases appear in the query terms."""
        haystack = " ".join(term.lower() for term in query_terms)
        matched: List[Dict] = []
        for note in self.context_notes:
            names = [note["context"]] + list(note.get("aliases", []))
            if any(re.search(rf"\b{re.escape(name)}\b", haystack) for name in names):
                matched.append(note)
        return matched

    def retrieve(
        self,
        query: str,
        plan: Plan,
        top_k: int = 8,
    ) -> Tuple[List[EditorialHit], Dict[int, float], List[Dict]]:
        """Score editorial passages against the query + plan.

        Returns ``(hits, per_track_boost, matched_context_notes)``.
        """
        plan_terms: List[str] = [
            *plan.moods,
            *plan.genres,
            *plan.listening_contexts,
        ]
        query_tokens = _tokenize(query) + _tokenize(" ".join(plan_terms))
        token_counts: Dict[str, int] = {}
        for token in query_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        matched_contexts = self.match_contexts([query, *plan_terms])

        hits: List[EditorialHit] = []
        track_scores: Dict[int, float] = {}

        # 1. Track-level notes, scored by token overlap.
        for song_id, tokens in self._track_tokens.items():
            if not tokens:
                continue
            overlap = sum(token_counts.get(token, 0) for token in set(tokens))
            if overlap <= 0:
                continue
            score = overlap / math.sqrt(len(set(tokens)))
            track_scores[song_id] = track_scores.get(song_id, 0.0) + score
            note = self.track_notes[song_id]
            hits.append(
                EditorialHit(
                    source="track_note",
                    key=f"{note['title']} - {note['artist']}",
                    song_id=song_id,
                    score=round(score, 4),
                    text=note.get("editorial_note", ""),
                )
            )

        # 2. Context notes: strong, intentional signal for situational prompts.
        for note in matched_contexts:
            hits.append(
                EditorialHit(
                    source="context_note",
                    key=note["context"],
                    score=1.0,
                    text=note.get("curation_note", ""),
                )
            )

        # 3. Genre notes for genres the plan or query named.
        for genre, note in self.genre_notes.items():
            named = genre in plan.genres or re.search(
                rf"\b{re.escape(genre)}\b", query.lower()
            )
            if named:
                hits.append(
                    EditorialHit(
                        source="genre_note",
                        key=genre,
                        score=0.8,
                        text=note.get("curation_note", ""),
                    )
                )

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k], track_scores, matched_contexts


# --------------------------------------------------------------------------- #
# Deterministic filtering + blended ranking
# --------------------------------------------------------------------------- #
def _apply_hard_filters(
    frame: pd.DataFrame, plan: Plan
) -> Tuple[pd.DataFrame, List[str]]:
    """Exclusions and numeric bounds. These are never relaxed."""
    filtered = frame
    applied: List[str] = []

    if plan.exclude_genres:
        excluded = {genre.lower() for genre in plan.exclude_genres}
        filtered = filtered[~filtered["genre_key"].isin(excluded)]
        applied.append(f"exclude genre in {sorted(excluded)}")
    if plan.exclude_moods:
        excluded = {mood.lower() for mood in plan.exclude_moods}
        filtered = filtered[~filtered["mood_key"].isin(excluded)]
        applied.append(f"exclude mood in {sorted(excluded)}")
    if plan.exclude_artists:
        excluded = {artist.lower() for artist in plan.exclude_artists}
        filtered = filtered[~filtered["artist_key"].isin(excluded)]
        applied.append(f"exclude artist in {sorted(excluded)}")
    if plan.min_bpm is not None:
        filtered = filtered[filtered["tempo_bpm"] >= plan.min_bpm]
        applied.append(f"tempo_bpm >= {plan.min_bpm}")
    if plan.max_bpm is not None:
        filtered = filtered[filtered["tempo_bpm"] <= plan.max_bpm]
        applied.append(f"tempo_bpm <= {plan.max_bpm}")
    if plan.min_energy is not None:
        filtered = filtered[filtered["energy"] >= plan.min_energy]
        applied.append(f"energy >= {plan.min_energy:g}")
    if plan.max_energy is not None:
        filtered = filtered[filtered["energy"] <= plan.max_energy]
        applied.append(f"energy <= {plan.max_energy:g}")

    return filtered, applied


def _soft_preference_score(
    row: pd.Series,
    plan: Plan,
    context_notes: Sequence[Dict],
) -> Tuple[float, List[str]]:
    """Feature-space score for a surviving track, plus why-matched strings."""
    score = 0.0
    reasons: List[str] = []

    if plan.genres and row["genre_key"] in {g.lower() for g in plan.genres}:
        score += 3.0
        reasons.append(f"requested genre ({row['genre']})")
    if plan.moods and row["mood_key"] in {m.lower() for m in plan.moods}:
        score += 3.0
        reasons.append(f"requested mood ({row['mood']})")

    if plan.acoustic_preference == "prefer":
        score += 1.5 * float(row["acousticness"])
        if row["acousticness"] >= 0.6:
            reasons.append(f"acoustic texture ({row['acousticness']:.2f})")
    elif plan.acoustic_preference == "avoid":
        score += 1.5 * (1.0 - float(row["acousticness"]))
        if row["acousticness"] <= 0.3:
            reasons.append(f"low acousticness ({row['acousticness']:.2f})")

    # Editorial context priors shape ranking directly.
    for note in context_notes:
        preferred = {g.lower() for g in note.get("preferred_genres", [])}
        discouraged = {g.lower() for g in note.get("discouraged_genres", [])}
        if row["genre_key"] in preferred:
            score += 2.0
            reasons.append(f"editorially preferred for {note['context']}")
        if row["genre_key"] in discouraged:
            score -= 2.0
            reasons.append(f"editorially discouraged for {note['context']}")
        bpm_range = note.get("preferred_bpm")
        if bpm_range and len(bpm_range) == 2:
            low, high = int(bpm_range[0]), int(bpm_range[1])
            tempo = int(row["tempo_bpm"])
            if low <= tempo <= high:
                score += 1.0
                reasons.append(f"tempo {tempo} BPM fits {note['context']}")
            else:
                distance = min(abs(tempo - low), abs(tempo - high))
                score -= min(1.5, distance / 60.0)

    # Mild centrality prior so ties resolve stably rather than by file order.
    score += 0.25 * float(row["valence"])
    return score, reasons


def retrieve_candidates(
    plan: Plan,
    query: str,
    catalog: Optional[pd.DataFrame] = None,
    knowledge: Optional[EditorialKnowledgeBase] = None,
    pool_size: int = 10,
) -> RetrievalResult:
    """Run the full Step 2 retrieval and return a typed result."""
    frame = get_catalog() if catalog is None else catalog.copy()
    knowledge = EditorialKnowledgeBase.load() if knowledge is None else knowledge

    hits, track_boosts, context_notes = knowledge.retrieve(query, plan)
    filtered, filters_applied = _apply_hard_filters(frame, plan)
    strict_match_count = int(len(filtered))

    relaxations: List[str] = []
    unsatisfiable = filtered.empty
    if unsatisfiable:
        # Hard constraints are never silently relaxed. The full catalog is still
        # ranked, but only so the agent can *report* the closest tracks as near
        # misses - they are never offered as recommendations.
        relaxations.append(
            "No catalog track satisfies every hard constraint. The constraints were "
            "not relaxed; the closest tracks are reported as near misses only."
        )
        filtered = frame

    max_boost = max(track_boosts.values()) if track_boosts else 0.0
    candidates: List[Candidate] = []
    for _, row in filtered.iterrows():
        song_id = int(row["id"])
        feature_score, reasons = _soft_preference_score(row, plan, context_notes)
        raw_boost = track_boosts.get(song_id, 0.0)
        editorial_score = (
            EDITORIAL_WEIGHT * (raw_boost / max_boost) if max_boost > 0 else 0.0
        )
        if editorial_score > 0.5:
            reasons.append("matched editorial notes")
        candidates.append(
            Candidate(
                song_id=song_id,
                title=str(row["title"]),
                artist=str(row["artist"]),
                genre=str(row["genre"]),
                mood=str(row["mood"]),
                energy=float(row["energy"]),
                tempo_bpm=int(row["tempo_bpm"]),
                valence=float(row["valence"]),
                danceability=float(row["danceability"]),
                acousticness=float(row["acousticness"]),
                feature_score=round(feature_score, 4),
                editorial_score=round(editorial_score, 4),
                total_score=round(feature_score + editorial_score, 4),
                editorial_note=knowledge.track_note_text(song_id),
                why_matched=reasons,
            )
        )

    candidates.sort(key=lambda c: (-c.total_score, c.song_id))
    pool = max(pool_size, plan.track_count * 2)

    return RetrievalResult(
        candidates=[] if unsatisfiable else candidates[:pool],
        near_misses=candidates[:3] if unsatisfiable else [],
        editorial_hits=hits,
        filters_applied=filters_applied or ["none (no hard constraints in plan)"],
        relaxations=relaxations,
        catalog_size=int(len(frame)),
        strict_match_count=strict_match_count,
    )


def format_candidates_for_prompt(candidates: Iterable[Candidate]) -> str:
    """Render the candidate pool as the only track universe the curator may use."""
    lines: List[str] = []
    for candidate in candidates:
        lines.append(
            f"- song_id={candidate.song_id} | title=\"{candidate.title}\" | "
            f"artist=\"{candidate.artist}\" | genre={candidate.genre} | "
            f"mood={candidate.mood} | tempo_bpm={candidate.tempo_bpm} | "
            f"energy={candidate.energy:.2f} | valence={candidate.valence:.2f} | "
            f"danceability={candidate.danceability:.2f} | "
            f"acousticness={candidate.acousticness:.2f}"
        )
        if candidate.editorial_note:
            lines.append(f"    editorial_note: {candidate.editorial_note}")
    return "\n".join(lines)
