"""Pydantic schemas for the agentic music recommender.

Every boundary in the agent pipeline is typed:

* ``Plan``            - Step 1 output (intent planning)
* ``RetrievalResult`` - Step 2 output (multi-source RAG)
* ``CuratedPlaylist`` - Step 3 output (LLM curation, structured)
* ``GuardrailReport`` - Step 4 output (verification)
* ``AgentTrace``      - the full reasoning trace written to logs/agent_traces.json
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AcousticPreference = Literal["prefer", "avoid", "any"]

MIN_CATALOG_BPM = 30
MAX_CATALOG_BPM = 300
MAX_TRACK_COUNT = 20


# --------------------------------------------------------------------------- #
# Step 1 - Intent planning
# --------------------------------------------------------------------------- #
class Plan(BaseModel):
    """Structured interpretation of a natural-language listening request."""

    model_config = ConfigDict(extra="ignore")

    moods: List[str] = Field(
        default_factory=list,
        description="Mood words the listener asked for, lowercase.",
    )
    genres: List[str] = Field(
        default_factory=list,
        description="Genres explicitly requested, lowercase.",
    )
    listening_contexts: List[str] = Field(
        default_factory=list,
        description="Situations the playlist is for, e.g. 'deep focus', 'workout'.",
    )
    exclude_genres: List[str] = Field(
        default_factory=list, description="Genres the listener ruled out."
    )
    exclude_artists: List[str] = Field(
        default_factory=list, description="Artists the listener ruled out."
    )
    exclude_moods: List[str] = Field(
        default_factory=list, description="Moods the listener ruled out."
    )
    min_bpm: Optional[int] = Field(
        default=None, description="Lower tempo bound in BPM, if stated."
    )
    max_bpm: Optional[int] = Field(
        default=None, description="Upper tempo bound in BPM, if stated."
    )
    min_energy: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Lower energy bound, 0-1."
    )
    max_energy: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Upper energy bound, 0-1."
    )
    acoustic_preference: AcousticPreference = Field(
        default="any", description="Whether acoustic texture is wanted, avoided or irrelevant."
    )
    track_count: int = Field(
        default=5, ge=1, le=MAX_TRACK_COUNT, description="How many tracks to return."
    )
    rationale: str = Field(
        default="",
        description="One sentence explaining how the request was interpreted.",
    )
    unsupported_requests: List[str] = Field(
        default_factory=list,
        description="Parts of the request the catalog cannot satisfy, or instructions "
        "that were ignored because they tried to change the agent's behaviour.",
    )

    @field_validator(
        "moods",
        "genres",
        "listening_contexts",
        "exclude_genres",
        "exclude_artists",
        "exclude_moods",
        mode="before",
    )
    @classmethod
    def _normalise_terms(cls, value: Any) -> Any:
        """Accept a bare string or a list; always emit a clean lowercase list."""
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            return value
        cleaned: List[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip().lower()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned

    @field_validator("min_bpm", "max_bpm", mode="before")
    @classmethod
    def _coerce_bpm(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        try:
            bpm = int(round(float(value)))
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("BPM bounds must be numeric") from exc
        return bpm

    @field_validator("track_count", mode="before")
    @classmethod
    def _coerce_track_count(cls, value: Any) -> Any:
        if value is None or value == "":
            return 5
        try:
            count = int(round(float(value)))
        except (TypeError, ValueError) as exc:
            raise ValueError("track_count must be numeric") from exc
        return min(max(count, 1), MAX_TRACK_COUNT)

    @model_validator(mode="after")
    def _check_bounds(self) -> "Plan":
        if self.min_bpm is not None and not (
            MIN_CATALOG_BPM <= self.min_bpm <= MAX_CATALOG_BPM
        ):
            raise ValueError(
                f"min_bpm must be between {MIN_CATALOG_BPM} and {MAX_CATALOG_BPM}"
            )
        if self.max_bpm is not None and not (
            MIN_CATALOG_BPM <= self.max_bpm <= MAX_CATALOG_BPM
        ):
            raise ValueError(
                f"max_bpm must be between {MIN_CATALOG_BPM} and {MAX_CATALOG_BPM}"
            )
        if (
            self.min_bpm is not None
            and self.max_bpm is not None
            and self.min_bpm > self.max_bpm
        ):
            raise ValueError("min_bpm cannot be greater than max_bpm")
        if (
            self.min_energy is not None
            and self.max_energy is not None
            and self.min_energy > self.max_energy
        ):
            raise ValueError("min_energy cannot be greater than max_energy")
        return self

    def constraint_summary(self) -> List[str]:
        """Human-readable list of the hard constraints this plan imposes."""
        parts: List[str] = [f"return exactly {self.track_count} track(s)"]
        if self.genres:
            parts.append(f"prefer genres: {', '.join(self.genres)}")
        if self.moods:
            parts.append(f"prefer moods: {', '.join(self.moods)}")
        if self.exclude_genres:
            parts.append(f"exclude genres: {', '.join(self.exclude_genres)}")
        if self.exclude_artists:
            parts.append(f"exclude artists: {', '.join(self.exclude_artists)}")
        if self.exclude_moods:
            parts.append(f"exclude moods: {', '.join(self.exclude_moods)}")
        if self.min_bpm is not None:
            parts.append(f"tempo >= {self.min_bpm} BPM")
        if self.max_bpm is not None:
            parts.append(f"tempo <= {self.max_bpm} BPM")
        if self.min_energy is not None:
            parts.append(f"energy >= {self.min_energy:g}")
        if self.max_energy is not None:
            parts.append(f"energy <= {self.max_energy:g}")
        if self.acoustic_preference != "any":
            parts.append(f"acoustic preference: {self.acoustic_preference}")
        return parts


# --------------------------------------------------------------------------- #
# Step 2 - Multi-source retrieval
# --------------------------------------------------------------------------- #
class EditorialHit(BaseModel):
    """One retrieved passage from data/editorial_notes.json."""

    source: Literal["track_note", "genre_note", "context_note"]
    key: str
    song_id: Optional[int] = None
    score: float
    text: str


class Candidate(BaseModel):
    """A catalog track that survived deterministic filtering, with its ranking parts."""

    song_id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: int
    valence: float
    danceability: float
    acousticness: float
    feature_score: float = 0.0
    editorial_score: float = 0.0
    total_score: float = 0.0
    editorial_note: str = ""
    why_matched: List[str] = Field(default_factory=list)

    def label(self) -> str:
        return f"#{self.song_id} {self.title} - {self.artist}"


class RetrievalResult(BaseModel):
    """Everything Step 2 produced, including how the filter was relaxed."""

    candidates: List[Candidate] = Field(default_factory=list)
    near_misses: List[Candidate] = Field(
        default_factory=list,
        description="Closest tracks when nothing satisfies the hard constraints. "
        "These are reported, never recommended.",
    )
    editorial_hits: List[EditorialHit] = Field(default_factory=list)
    filters_applied: List[str] = Field(default_factory=list)
    relaxations: List[str] = Field(default_factory=list)
    catalog_size: int = 0
    strict_match_count: int = 0


# --------------------------------------------------------------------------- #
# Step 3 - Curation (the LLM's structured output)
# --------------------------------------------------------------------------- #
class CuratedTrack(BaseModel):
    """One recommendation as returned by the curator model."""

    model_config = ConfigDict(extra="ignore")

    song_id: int = Field(description="The id column from data/songs.csv.")
    title: str = Field(description="Track title, copied verbatim from the catalog.")
    artist: str = Field(description="Artist name, copied verbatim from the catalog.")
    reason: str = Field(
        default="",
        description="One or two sentences grounded in the track's metadata and editorial note.",
    )

    @field_validator("song_id", mode="before")
    @classmethod
    def _coerce_song_id(cls, value: Any) -> Any:
        try:
            return int(str(value).strip().lstrip("#"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"song_id must be an integer, got {value!r}") from exc

    @field_validator("title", "artist", "reason", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> Any:
        return "" if value is None else str(value).strip()


class CuratedPlaylist(BaseModel):
    """The curator model's full structured answer."""

    model_config = ConfigDict(extra="ignore")

    playlist_name: str = Field(default="Custom Mix")
    summary: str = Field(
        default="", description="Two sentences on the arc of the playlist."
    )
    tracks: List[CuratedTrack] = Field(default_factory=list)

    @field_validator("playlist_name", "summary", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> Any:
        return "" if value is None else str(value).strip()


# --------------------------------------------------------------------------- #
# Step 4 - Guardrails
# --------------------------------------------------------------------------- #
class GuardrailViolation(BaseModel):
    """A single failed check, phrased so it can be fed straight back to the model."""

    check: Literal["hallucination", "constraint", "format", "grounding"]
    song_id: Optional[int] = None
    message: str

    def __str__(self) -> str:  # pragma: no cover - formatting helper
        return f"[{self.check}] {self.message}"


class GuardrailReport(BaseModel):
    """Result of running every deterministic check over a candidate playlist."""

    passed: bool = True
    violations: List[GuardrailViolation] = Field(default_factory=list)
    checks_run: List[str] = Field(default_factory=list)

    @property
    def hallucination_violations(self) -> List[GuardrailViolation]:
        return [v for v in self.violations if v.check in ("hallucination", "grounding")]

    @property
    def constraint_violations(self) -> List[GuardrailViolation]:
        return [v for v in self.violations if v.check == "constraint"]

    def feedback(self) -> str:
        """Render violations as correction instructions for the self-correction loop."""
        return "\n".join(f"- {violation}" for violation in self.violations)


# --------------------------------------------------------------------------- #
# Tracing + final output
# --------------------------------------------------------------------------- #
class AttemptTrace(BaseModel):
    """One pass through curate -> verify."""

    attempt: int
    prompt_kind: Literal["initial", "self_correction"]
    raw_model_output: str = ""
    parsed_playlist: Optional[CuratedPlaylist] = None
    guardrail: Optional[GuardrailReport] = None
    parse_error: Optional[str] = None


class Recommendation(BaseModel):
    """One verified track handed back to the UI/CLI."""

    song_id: int
    title: str
    artist: str
    genre: str
    mood: str
    tempo_bpm: int
    energy: float
    reason: str
    editorial_note: str = ""


class AgentResult(BaseModel):
    """The agent's final answer plus everything needed to audit it."""

    query: str
    playlist_name: str
    summary: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    plan: Optional[Plan] = None
    guardrail: Optional[GuardrailReport] = None
    retries: int = 0
    fallback_used: bool = False
    provider: str = "offline"
    notices: List[str] = Field(default_factory=list)
    trace_id: str = ""


class AgentTrace(BaseModel):
    """The full, append-only reasoning record written to logs/agent_traces.json."""

    trace_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    query: str
    provider: str
    model: str = ""
    plan: Optional[Plan] = None
    plan_error: Optional[str] = None
    retrieval: Optional[RetrievalResult] = None
    attempts: List[AttemptTrace] = Field(default_factory=list)
    retries: int = 0
    fallback_used: bool = False
    final_output: Optional[AgentResult] = None
    duration_ms: int = 0
    notices: List[str] = Field(default_factory=list)

    def to_log_dict(self) -> Dict[str, Any]:
        """Compact dict for the JSON log (drops empty keys for readability)."""
        return self.model_dump(mode="json", exclude_none=True)
