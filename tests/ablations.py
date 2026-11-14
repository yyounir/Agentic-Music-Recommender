"""Ablation studies that produce the comparison tables in ``model_card.md``.

Two independent studies:

``--rag``     Before vs after the editorial knowledge source. Six niche prompts are
              run with the full ``data/editorial_notes.json`` and again with an
              empty knowledge base. Quality is scored as precision@3 against
              hand-labelled ideal sets (see ``IDEAL_SETS``), which are authored
              from the catalog's own feature values and are *not* derived from
              the editorial notes - so the comparison is not circular.

``--prompt``  Baseline zero-shot curation vs the specialised few-shot "Music
              Curator" persona. Both variants are scored on their FIRST response,
              before any self-correction, because that is what the prompt itself
              is responsible for.

              With ``GEMINI_API_KEY`` set this runs a real A/B against Gemini:
              variant A uses a terse generic instruction with the whole catalog
              pasted in, variant B uses the project's persona, few-shot examples
              and filtered candidate pool. Without a key it falls back to a
              clearly-labelled SIMULATED baseline (``NaiveCuratorEngine``) that
              reproduces the failure modes an unconstrained prompt exhibits:
              prose-wrapped JSON, the full catalog as its universe, generic
              rationales and no regard for the requested count.

Usage::

    python tests/ablations.py            # both studies
    python tests/ablations.py --rag
    python tests/ablations.py --prompt --engine gemini
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import CurationRequest, OfflineHeuristicEngine, build_engine  # noqa: E402
from src.recommender import MusicCuratorAgent  # noqa: E402
from src.retrieval import EditorialKnowledgeBase, get_catalog, retrieve_candidates  # noqa: E402
from src.schemas import Plan  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "logs" / "ablation_report.json"

EMPTY_KB = {"track_notes": [], "genre_notes": [], "context_notes": []}


# --------------------------------------------------------------------------- #
# Study 1: RAG ablation
# --------------------------------------------------------------------------- #
# Hand-labelled ideal sets, authored from the numeric columns of data/songs.csv
# (tempo, energy, valence, acousticness) rather than from the editorial notes.
IDEAL_SETS: List[Dict[str, Any]] = [
    {
        "prompt": "Something for a deep focus work session.",
        "ideal": {2, 4, 6, 9},
        "basis": "energy <= 0.42 and tempo <= 80 BPM with no vocal hook",
    },
    {
        "prompt": "Three tracks for a late night drive through the city - neon, overcast, not cheerful.",
        "ideal": {3, 8, 16},
        "basis": "valence <= 0.49 with a beat (energy >= 0.46)",
    },
    {
        "prompt": "I just got bad news and I want to sit with it. Genuinely sad songs.",
        "ideal": {11, 13, 16},
        "basis": "valence <= 0.73 with low energy and high acousticness",
    },
    {
        "prompt": "Background music for a dinner party - nothing too loud.",
        "ideal": {7, 10, 12, 15},
        "basis": "energy 0.37-0.79, valence >= 0.71, tempo 90-124 BPM",
    },
    {
        "prompt": "Lift my mood without hyping me up.",
        "ideal": {1, 10, 12, 17},
        "basis": "valence >= 0.81 with energy <= 0.82",
    },
    {
        "prompt": "I need to wind down before sleep.",
        "ideal": {4, 6, 13},
        "basis": "energy <= 0.35, tempo <= 72 BPM, acousticness >= 0.86",
    },
]


def _precision_at_k(selected: Sequence[int], ideal: set, k: int = 3) -> float:
    top = list(selected)[:k]
    if not top:
        return 0.0
    return len(set(top).intersection(ideal)) / len(top)


def run_rag_ablation() -> Dict[str, Any]:
    catalog = get_catalog()
    full_kb = EditorialKnowledgeBase.load()
    empty_kb = EditorialKnowledgeBase(EMPTY_KB)

    # The plan is produced by the same heuristic planner in both arms so that the
    # only variable is whether editorial notes inform retrieval and ranking.
    planner = OfflineHeuristicEngine(full_kb)
    vocabulary = {
        "genres": sorted(catalog["genre"].str.lower().unique().tolist()),
        "moods": sorted(catalog["mood"].str.lower().unique().tolist()),
        "artists": sorted(catalog["artist"].unique().tolist()),
        "bpm_range": [
            int(catalog["tempo_bpm"].min()),
            int(catalog["tempo_bpm"].max()),
        ],
    }

    rows: List[Dict[str, Any]] = []
    for entry in IDEAL_SETS:
        plan = Plan.model_validate(
            json.loads(planner.plan(entry["prompt"], vocabulary))
        )
        plan = plan.model_copy(update={"track_count": 3})

        without = retrieve_candidates(plan, entry["prompt"], catalog, empty_kb)
        with_notes = retrieve_candidates(plan, entry["prompt"], catalog, full_kb)

        top_without = [c.song_id for c in without.candidates[:3]]
        top_with = [c.song_id for c in with_notes.candidates[:3]]

        rows.append(
            {
                "prompt": entry["prompt"],
                "ideal": sorted(entry["ideal"]),
                "ideal_basis": entry["basis"],
                "top3_without_rag": top_without,
                "top3_with_rag": top_with,
                "precision_without_rag": _precision_at_k(top_without, entry["ideal"]),
                "precision_with_rag": _precision_at_k(top_with, entry["ideal"]),
                "editorial_passages_retrieved": len(with_notes.editorial_hits),
            }
        )

    mean_without = sum(r["precision_without_rag"] for r in rows) / len(rows)
    mean_with = sum(r["precision_with_rag"] for r in rows) / len(rows)
    changed = sum(
        1 for r in rows if r["top3_without_rag"] != r["top3_with_rag"]
    )

    return {
        "rows": rows,
        "summary": {
            "prompts": len(rows),
            "mean_precision_at_3_without_rag": mean_without,
            "mean_precision_at_3_with_rag": mean_with,
            "prompts_whose_selection_changed": changed,
        },
    }


def print_rag_report(study: Dict[str, Any]) -> None:
    print("=" * 78)
    print("ABLATION 1 - BEFORE vs AFTER editorial RAG (data/editorial_notes.json)")
    print("=" * 78)
    print()
    for row in study["rows"]:
        print(f"  prompt: {row['prompt']}")
        print(f"    ideal (hand-labelled from CSV features): {row['ideal']}")
        print(
            f"    without notes -> {row['top3_without_rag']}  "
            f"precision@3 = {row['precision_without_rag']:.2f}"
        )
        print(
            f"    with notes    -> {row['top3_with_rag']}  "
            f"precision@3 = {row['precision_with_rag']:.2f}"
        )
        print()
    summary = study["summary"]
    print(
        f"  mean precision@3: {summary['mean_precision_at_3_without_rag']:.2f} "
        f"-> {summary['mean_precision_at_3_with_rag']:.2f}  "
        f"({summary['prompts_whose_selection_changed']}/{summary['prompts']} "
        "prompts changed selection)"
    )
    print()


# --------------------------------------------------------------------------- #
# Study 2: prompt ablation
# --------------------------------------------------------------------------- #
PROMPT_BENCHMARK = [
    "I need three lofi tracks under 90 BPM for a late-night coding session.",
    "Build me a 4-song workout playlist that climbs in intensity, but no metal, "
    "and everything has to be at least 120 BPM.",
    "Five songs for a dinner party with friends - nothing too loud.",
    "It is raining and I am feeling nostalgic. Give me 3 acoustic songs.",
    "Three tracks for a late night drive through the city.",
    "Two genuinely sad songs, not just slow ones.",
]


class NaiveCuratorEngine:
    """SIMULATED zero-shot baseline. Not a language model.

    Reproduces, deterministically, what an unconstrained prompt tends to produce:
    JSON wrapped in prose, the entire catalog treated as the track universe
    instead of the filtered candidate pool, a famous-sounding track that is not
    in the catalog, generic rationales, and no regard for the requested count.
    It exists so the comparison table can be regenerated without an API key; run
    with ``--engine gemini`` for a real A/B.
    """

    provider = "simulated-zero-shot"
    model = "naive-v1"

    def __init__(self, knowledge: Optional[EditorialKnowledgeBase] = None):
        self._planner = OfflineHeuristicEngine(knowledge)

    def plan(self, query: str, vocabulary: Dict[str, Any]) -> str:
        return self._planner.plan(query, vocabulary)

    def curate(self, request: CurationRequest) -> str:
        catalog = get_catalog()
        tracks: List[Dict[str, Any]] = [
            {
                "song_id": 101,
                "title": "Bohemian Rhapsody",
                "artist": "Queen",
                "reason": "A classic that everyone loves.",
            }
        ]
        for _, row in catalog.head(4).iterrows():
            tracks.append(
                {
                    "song_id": int(row["id"]),
                    "title": str(row["title"]),
                    "artist": str(row["artist"]),
                    "reason": "A great pick for this vibe.",
                }
            )
        payload = {
            "playlist_name": "Your Playlist",
            "summary": "Here are some songs you might enjoy!",
            "tracks": tracks,
        }
        return (
            "Sure! Here is a playlist I put together for you:\n\n"
            "```json\n" + json.dumps(payload, indent=2) + "\n```\n"
            "Let me know if you'd like more suggestions!"
        )


class ZeroShotGeminiEngine:
    """Real zero-shot arm: terse instruction, no persona, no few-shot examples."""

    provider = "gemini-zero-shot"

    SYSTEM = (
        "You are a helpful assistant. Recommend songs for the user's request. "
        "Reply with JSON containing playlist_name, summary and tracks, where each "
        "track has song_id, title, artist and reason."
    )

    def __init__(self, inner):
        self.inner = inner
        self.model = inner.model

    def plan(self, query: str, vocabulary: Dict[str, Any]) -> str:
        return self.inner.plan(query, vocabulary)

    def curate(self, request: CurationRequest) -> str:
        catalog = get_catalog()
        listing = "\n".join(
            f"{int(row['id'])}: {row['title']} - {row['artist']} ({row['genre']}, "
            f"{int(row['tempo_bpm'])} BPM)"
            for _, row in catalog.iterrows()
        )
        message = (
            f"User request: {request.query}\n\n"
            f"Here is a song library:\n{listing}\n\n"
            f"Pick {request.plan.track_count} songs."
        )
        return self.inner._generate(  # noqa: SLF001 - deliberate reuse of the transport
            self.SYSTEM, [message], None, 0.3
        )


def _first_attempt_metrics(agent: MusicCuratorAgent, queries: List[str]) -> Dict[str, Any]:
    catalog_ids = set(get_catalog()["id"].tolist())
    parsed_ok = 0
    format_ok = 0
    grounding_ok = 0
    constraint_ok = 0
    total_tracks = 0
    fabricated = 0
    reason_lengths: List[int] = []
    numeric_reasons = 0
    # Capture real rationales so the model card can quote actual output rather
    # than a paraphrase of what the output is like.
    samples: List[Dict[str, Any]] = []

    for query in queries:
        result, trace = agent.recommend(query, log=False)
        if not trace.attempts:
            continue
        attempt = trace.attempts[0]
        report = attempt.guardrail
        if attempt.parsed_playlist is not None:
            parsed_ok += 1
            for index, track in enumerate(attempt.parsed_playlist.tracks):
                total_tracks += 1
                if track.song_id not in catalog_ids:
                    fabricated += 1
                reason_lengths.append(len(track.reason))
                if re.search(r"\d", track.reason):
                    numeric_reasons += 1
                if index == 0:
                    samples.append(
                        {
                            "query": query,
                            "playlist_name": attempt.parsed_playlist.playlist_name,
                            "song_id": track.song_id,
                            "title": track.title,
                            "artist": track.artist,
                            "reason": track.reason,
                        }
                    )
        if report is not None:
            if not any(v.check == "format" for v in report.violations):
                format_ok += 1
            if not any(v.check == "grounding" for v in report.violations):
                grounding_ok += 1
            if not report.constraint_violations:
                constraint_ok += 1

    cases = len(queries)
    return {
        "cases": cases,
        "schema_parse_rate": parsed_ok / cases if cases else 0.0,
        "format_adherence": format_ok / cases if cases else 0.0,
        "constraint_adherence": constraint_ok / cases if cases else 0.0,
        "rationale_grounding_rate": grounding_ok / cases if cases else 0.0,
        "hallucination_rate": fabricated / total_tracks if total_tracks else 0.0,
        "tracks": total_tracks,
        "fabricated_tracks": fabricated,
        "mean_rationale_chars": (
            sum(reason_lengths) / len(reason_lengths) if reason_lengths else 0.0
        ),
        "rationales_citing_a_number": (
            numeric_reasons / total_tracks if total_tracks else 0.0
        ),
        "sample_rationales": samples,
    }


def run_prompt_ablation(prefer: str = "auto") -> Dict[str, Any]:
    knowledge = EditorialKnowledgeBase.load()
    engine, notices = build_engine(prefer=prefer)

    if engine.provider == "gemini":
        baseline_engine: Any = ZeroShotGeminiEngine(engine)
        baseline_label = "Gemini, zero-shot generic prompt, full catalog in context"
        simulated = False
    else:
        baseline_engine = NaiveCuratorEngine(knowledge)
        baseline_label = (
            "SIMULATED zero-shot baseline (offline engine in use) - assumed "
            "generic-prompt behaviour, full catalog as universe. NOTE: the real "
            "Gemini A/B contradicted this simulation on every headline metric; "
            "see model_card.md section 6."
        )
        simulated = True

    # max_retries=0 so each arm is judged on its first response only.
    baseline_agent = MusicCuratorAgent(engine=baseline_engine, max_retries=0)
    specialised_agent = MusicCuratorAgent(engine=engine, max_retries=0)

    return {
        "notices": notices,
        "simulated_baseline": simulated,
        "baseline_label": baseline_label,
        "specialised_label": (
            f"{engine.provider} ({engine.model}), Music Curator persona + "
            "2 few-shot examples + filtered candidate pool"
        ),
        "baseline": _first_attempt_metrics(baseline_agent, PROMPT_BENCHMARK),
        "specialised": _first_attempt_metrics(specialised_agent, PROMPT_BENCHMARK),
    }


def print_prompt_report(study: Dict[str, Any]) -> None:
    print("=" * 78)
    print("ABLATION 2 - BASELINE ZERO-SHOT vs SPECIALISED FEW-SHOT CURATOR")
    print("=" * 78)
    print()
    if study["simulated_baseline"]:
        print("  WARNING: the baseline arm is SIMULATED - it encodes assumed")
        print("           generic-prompt failure modes, not a real model.")
        print("           The real A/B (--engine gemini) contradicted it on every")
        print("           headline metric. Treat these numbers as a hypothesis that")
        print("           was tested and refuted; see model_card.md section 6.")
        print()
    print(f"  A = {study['baseline_label']}")
    print(f"  B = {study['specialised_label']}")
    print()
    metrics = [
        ("Schema parse rate (1st response)", "schema_parse_rate", "pct"),
        ("Format adherence (1st response)", "format_adherence", "pct"),
        ("Constraint adherence (1st response)", "constraint_adherence", "pct"),
        ("Rationale grounding rate", "rationale_grounding_rate", "pct"),
        ("Hallucination rate (1st response)", "hallucination_rate", "pct"),
        ("Mean rationale length (chars)", "mean_rationale_chars", "num"),
        ("Rationales citing a number", "rationales_citing_a_number", "pct"),
    ]
    print(f"  {'METRIC':<38} {'A (zero-shot)':>15} {'B (few-shot)':>15}")
    print("  " + "-" * 70)
    for label, key, kind in metrics:
        a_value = study["baseline"][key]
        b_value = study["specialised"][key]
        if kind == "pct":
            print(f"  {label:<38} {a_value * 100:>14.0f}% {b_value * 100:>14.0f}%")
        else:
            print(f"  {label:<38} {a_value:>15.0f} {b_value:>15.0f}")
    print()


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the model-card ablations.")
    parser.add_argument("--rag", action="store_true", help="Run the RAG ablation only.")
    parser.add_argument(
        "--prompt", action="store_true", help="Run the prompt ablation only."
    )
    parser.add_argument(
        "--engine", choices=["auto", "gemini", "offline"], default="auto"
    )
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)

    run_all = not (args.rag or args.prompt)
    payload: Dict[str, Any] = {}

    if args.rag or run_all:
        study = run_rag_ablation()
        print_rag_report(study)
        payload["rag_ablation"] = study

    if args.prompt or run_all:
        study = run_prompt_ablation(args.engine)
        print_prompt_report(study)
        payload["prompt_ablation"] = study

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"JSON report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
