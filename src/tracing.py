"""Structured JSON trace logging for every agent run.

Each run appends one :class:`~src.schemas.AgentTrace` object to
``logs/agent_traces.json``, which stays a single valid JSON array so it can be
loaded with ``json.load`` and inspected in the Streamlit UI. The log is capped so
that a long demo session cannot grow it without bound.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.schemas import AgentTrace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACE_PATH = PROJECT_ROOT / "logs" / "agent_traces.json"
MAX_TRACES = 200


def new_trace_id() -> str:
    return f"trace-{uuid.uuid4().hex[:12]}"


def _resolve(path: Optional[str | Path]) -> Path:
    if path is None:
        return DEFAULT_TRACE_PATH
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def read_traces(path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    """Load the trace log, tolerating an absent or corrupt file."""
    resolved = _resolve(path)
    if not resolved.exists():
        return []
    try:
        with resolved.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def append_trace(
    trace: AgentTrace,
    path: Optional[str | Path] = None,
    max_traces: int = MAX_TRACES,
) -> Path:
    """Append one trace and rewrite the log atomically."""
    resolved = _resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    traces = read_traces(resolved)
    traces.append(trace.to_log_dict())
    if len(traces) > max_traces:
        traces = traces[-max_traces:]

    handle_fd, temporary = tempfile.mkstemp(
        dir=str(resolved.parent), prefix=".traces-", suffix=".tmp"
    )
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            json.dump(traces, handle, indent=2, ensure_ascii=False)
        os.replace(temporary, resolved)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return resolved


def summarise_trace(trace: AgentTrace) -> str:
    """One-line human summary used by the CLI."""
    status = "PASS" if (trace.final_output and trace.final_output.guardrail and trace.final_output.guardrail.passed) else "FALLBACK"
    return (
        f"{trace.trace_id} | provider={trace.provider} | retries={trace.retries} | "
        f"fallback={trace.fallback_used} | guardrails={status} | "
        f"{trace.duration_ms}ms"
    )


def render_trace_markdown(trace: AgentTrace) -> str:
    """Render a trace as the step-by-step block embedded in ai_interactions.md."""
    lines: List[str] = []
    lines.append(f"**Trace `{trace.trace_id}`** - provider `{trace.provider}`"
                 f" (`{trace.model}`), {trace.duration_ms} ms")
    lines.append("")
    lines.append(f"**1. User Query** -> `{trace.query}`")
    lines.append("")
    if trace.plan is not None:
        plan = trace.plan
        lines.append("**2. Parsed Plan**")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(plan.model_dump(mode="json"), indent=2))
        lines.append("```")
        lines.append("")
    if trace.retrieval is not None:
        retrieval = trace.retrieval
        lines.append("**3. Multi-Source RAG Retrieval**")
        lines.append("")
        lines.append(f"- Pandas filters: `{'; '.join(retrieval.filters_applied)}`")
        lines.append(
            f"- Catalog rows satisfying every hard constraint: "
            f"{retrieval.strict_match_count}/{retrieval.catalog_size}"
        )
        for note in retrieval.relaxations:
            lines.append(f"- Notice: {note}")
        lines.append("- Editorial passages retrieved:")
        for hit in retrieval.editorial_hits[:5]:
            lines.append(f"  - `[{hit.source}] {hit.key}` (score {hit.score:g})")
        if retrieval.candidates:
            lines.append("- Ranked candidate pool (feature + editorial score):")
            for candidate in retrieval.candidates[:6]:
                lines.append(
                    f"  - `#{candidate.song_id} {candidate.title}` total="
                    f"{candidate.total_score:g} (features {candidate.feature_score:g} + "
                    f"editorial {candidate.editorial_score:g})"
                )
        else:
            lines.append(
                "- Ranked candidate pool: **empty** - no track satisfies the hard "
                "constraints, so no track is eligible."
            )
        if retrieval.near_misses:
            lines.append("- Near misses (reported to the user, never recommended):")
            for candidate in retrieval.near_misses:
                lines.append(
                    f"  - `#{candidate.song_id} {candidate.title}` "
                    f"({candidate.genre}, {candidate.tempo_bpm} BPM, "
                    f"energy {candidate.energy:.2f})"
                )
        lines.append("")

    if not trace.attempts:
        lines.append("**4. Initial Candidate Selection**")
        lines.append("")
        lines.append(
            "- Skipped. With an empty candidate pool no generated answer could pass "
            "verification, so the agent short-circuited and made **zero model calls**."
        )
        lines.append("")

    for attempt in trace.attempts:
        heading = (
            "**4. Initial Candidate Selection**"
            if attempt.prompt_kind == "initial"
            else f"**5. Self-Correction Retry {attempt.attempt - 1}**"
        )
        lines.append(heading)
        lines.append("")
        if attempt.parse_error:
            lines.append(f"- Parse error: `{attempt.parse_error}`")
        if attempt.parsed_playlist is not None:
            for track in attempt.parsed_playlist.tracks:
                lines.append(
                    f"  - `#{track.song_id}` {track.title} - {track.artist}"
                )
        if attempt.guardrail is not None:
            if attempt.guardrail.passed:
                lines.append(
                    f"- Guardrail check: **PASSED** "
                    f"({', '.join(attempt.guardrail.checks_run)})"
                )
            else:
                lines.append("- Guardrail check: **FAILED**")
                for violation in attempt.guardrail.violations:
                    lines.append(f"  - {violation}")
        lines.append("")
    if trace.final_output is not None:
        result = trace.final_output
        lines.append("**6. Final Output**")
        lines.append("")
        lines.append(f"- Playlist: *{result.playlist_name}*")
        lines.append(f"- Retries: {result.retries} | Fallback used: {result.fallback_used}")
        if not result.recommendations:
            lines.append("  - *(no tracks returned)*")
        for recommendation in result.recommendations:
            lines.append(
                f"  - `#{recommendation.song_id}` **{recommendation.title}** - "
                f"{recommendation.artist} ({recommendation.genre}, "
                f"{recommendation.tempo_bpm} BPM)"
            )
        lines.append("")
    return "\n".join(lines)
