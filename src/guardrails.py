"""Step 4: deterministic verification of whatever the curator produced.

Nothing here calls a model. Every check is a plain comparison against
``data/songs.csv`` or against the :class:`~src.schemas.Plan`, which is what lets
the hallucination rate be asserted rather than estimated.

Checks run, in order:

``format``         structural sanity - at least one track, no duplicate ids, the
                   requested length (or the full candidate pool when it is smaller).
``hallucination``  every ``song_id`` exists in the catalog, and the ``title`` and
                   ``artist`` the model returned match that row verbatim.
``constraint``     exclusions, BPM bounds and energy bounds from the plan.
``grounding``      each rationale is non-empty and anchors on something concrete
                   from that track's metadata or editorial note.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import pandas as pd

from src.retrieval import EditorialKnowledgeBase
from src.schemas import (
    Candidate,
    CuratedPlaylist,
    CuratedTrack,
    GuardrailReport,
    GuardrailViolation,
    Plan,
    Recommendation,
    RetrievalResult,
)

CHECKS = ("format", "hallucination", "constraint", "grounding")

_GENERIC_PHRASES = re.compile(
    r"^(a |an |the )?(perfect|great|good|nice|amazing|excellent|solid)\b", re.IGNORECASE
)


def _catalog_index(catalog: pd.DataFrame) -> Dict[int, Dict[str, object]]:
    return {
        int(row["id"]): {
            "title": str(row["title"]).strip(),
            "artist": str(row["artist"]).strip(),
            "genre": str(row["genre"]).strip(),
            "mood": str(row["mood"]).strip(),
            "tempo_bpm": int(row["tempo_bpm"]),
            "energy": float(row["energy"]),
            "valence": float(row["valence"]),
            "danceability": float(row["danceability"]),
            "acousticness": float(row["acousticness"]),
        }
        for _, row in catalog.iterrows()
    }


def _grounding_tokens(row: Dict[str, object], editorial_note: str) -> List[str]:
    tokens = [
        str(row["genre"]).lower(),
        str(row["mood"]).lower(),
    ]
    tokens += re.findall(r"[a-z]{4,}", editorial_note.lower())
    return tokens


def verify_playlist(
    playlist: CuratedPlaylist,
    plan: Plan,
    retrieval: RetrievalResult,
    catalog: pd.DataFrame,
    knowledge: Optional[EditorialKnowledgeBase] = None,
) -> GuardrailReport:
    """Run every check and return a report that can be fed back to the model."""
    index = _catalog_index(catalog)
    violations: List[GuardrailViolation] = []

    tracks: Sequence[CuratedTrack] = playlist.tracks

    # ---------------------------------------------------------------- format
    available = len(retrieval.candidates)
    expected = min(plan.track_count, available) if available else 0

    if not tracks:
        if expected > 0:
            violations.append(
                GuardrailViolation(
                    check="format",
                    message=(
                        f"The playlist is empty but {available} candidate track(s) "
                        f"were supplied; return {expected}."
                    ),
                )
            )
    elif expected and len(tracks) != expected:
        violations.append(
            GuardrailViolation(
                check="format",
                message=(
                    f"Returned {len(tracks)} track(s) but the plan asks for "
                    f"{expected} (track_count={plan.track_count}, "
                    f"{available} candidate(s) available)."
                ),
            )
        )

    seen: Dict[int, int] = {}
    for track in tracks:
        seen[track.song_id] = seen.get(track.song_id, 0) + 1
    for song_id, count in seen.items():
        if count > 1:
            violations.append(
                GuardrailViolation(
                    check="format",
                    song_id=song_id,
                    message=f"song_id {song_id} appears {count} times; each track "
                    "may appear at most once.",
                )
            )

    # --------------------------------------------------------- hallucination
    for track in tracks:
        row = index.get(track.song_id)
        if row is None:
            violations.append(
                GuardrailViolation(
                    check="hallucination",
                    song_id=track.song_id,
                    message=(
                        f'song_id {track.song_id} ("{track.title}" by '
                        f'"{track.artist}") does not exist in data/songs.csv. '
                        "Replace it with a track from the candidate list."
                    ),
                )
            )
            continue
        if track.title.strip().casefold() != str(row["title"]).casefold():
            violations.append(
                GuardrailViolation(
                    check="hallucination",
                    song_id=track.song_id,
                    message=(
                        f'Title mismatch for song_id {track.song_id}: you wrote '
                        f'"{track.title}" but the catalog says "{row["title"]}". '
                        "Copy the catalog title verbatim."
                    ),
                )
            )
        if track.artist.strip().casefold() != str(row["artist"]).casefold():
            violations.append(
                GuardrailViolation(
                    check="hallucination",
                    song_id=track.song_id,
                    message=(
                        f'Artist mismatch for song_id {track.song_id}: you wrote '
                        f'"{track.artist}" but the catalog says "{row["artist"]}". '
                        "Copy the catalog artist verbatim."
                    ),
                )
            )

    # ------------------------------------------------------------ constraint
    excluded_genres = {genre.lower() for genre in plan.exclude_genres}
    excluded_moods = {mood.lower() for mood in plan.exclude_moods}
    excluded_artists = {artist.lower() for artist in plan.exclude_artists}

    for track in tracks:
        row = index.get(track.song_id)
        if row is None:
            continue  # already reported as a hallucination
        genre = str(row["genre"]).lower()
        mood = str(row["mood"]).lower()
        artist = str(row["artist"]).lower()
        tempo = int(row["tempo_bpm"])
        energy = float(row["energy"])

        if genre in excluded_genres:
            violations.append(
                GuardrailViolation(
                    check="constraint",
                    song_id=track.song_id,
                    message=f'"{row["title"]}" is {row["genre"]}, which the listener '
                    "excluded. Drop it and pick another candidate.",
                )
            )
        if mood in excluded_moods:
            violations.append(
                GuardrailViolation(
                    check="constraint",
                    song_id=track.song_id,
                    message=f'"{row["title"]}" has excluded mood "{row["mood"]}".',
                )
            )
        if artist in excluded_artists:
            violations.append(
                GuardrailViolation(
                    check="constraint",
                    song_id=track.song_id,
                    message=f'"{row["artist"]}" was excluded by the listener.',
                )
            )
        if plan.min_bpm is not None and tempo < plan.min_bpm:
            violations.append(
                GuardrailViolation(
                    check="constraint",
                    song_id=track.song_id,
                    message=f'"{row["title"]}" is {tempo} BPM, below the requested '
                    f"minimum of {plan.min_bpm} BPM.",
                )
            )
        if plan.max_bpm is not None and tempo > plan.max_bpm:
            violations.append(
                GuardrailViolation(
                    check="constraint",
                    song_id=track.song_id,
                    message=f'"{row["title"]}" is {tempo} BPM, above the requested '
                    f"maximum of {plan.max_bpm} BPM.",
                )
            )
        if plan.min_energy is not None and energy < plan.min_energy:
            violations.append(
                GuardrailViolation(
                    check="constraint",
                    song_id=track.song_id,
                    message=f'"{row["title"]}" has energy {energy:.2f}, below the '
                    f"requested minimum of {plan.min_energy:g}.",
                )
            )
        if plan.max_energy is not None and energy > plan.max_energy:
            violations.append(
                GuardrailViolation(
                    check="constraint",
                    song_id=track.song_id,
                    message=f'"{row["title"]}" has energy {energy:.2f}, above the '
                    f"requested maximum of {plan.max_energy:g}.",
                )
            )

    # ------------------------------------------------------------- grounding
    knowledge = knowledge or EditorialKnowledgeBase.load()
    for track in tracks:
        row = index.get(track.song_id)
        if row is None:
            continue
        reason = track.reason.strip()
        if len(reason) < 15:
            violations.append(
                GuardrailViolation(
                    check="grounding",
                    song_id=track.song_id,
                    message=f'The rationale for "{row["title"]}" is missing or too '
                    "short. Give one or two sentences citing its metadata or "
                    "editorial note.",
                )
            )
            continue
        note = knowledge.track_note_text(track.song_id)
        descriptors = knowledge.track_descriptors(track.song_id)
        contexts = knowledge.track_contexts(track.song_id)
        anchors = _grounding_tokens(row, note) + [
            d.lower() for d in descriptors
        ] + [c.lower() for c in contexts]
        lowered = reason.lower()
        has_number = bool(re.search(r"\d", reason))
        has_anchor = any(anchor and anchor in lowered for anchor in anchors)
        if not has_number and not has_anchor:
            violations.append(
                GuardrailViolation(
                    check="grounding",
                    song_id=track.song_id,
                    message=f'The rationale for "{row["title"]}" cites nothing from '
                    "its metadata or editorial note. Reference its tempo, energy, "
                    "valence, genre, mood or editorial note.",
                )
            )
        elif _GENERIC_PHRASES.match(reason) and not has_number:
            violations.append(
                GuardrailViolation(
                    check="grounding",
                    song_id=track.song_id,
                    message=f'The rationale for "{row["title"]}" opens with generic '
                    "praise. Lead with a concrete feature instead.",
                )
            )

    return GuardrailReport(
        passed=not violations, violations=violations, checks_run=list(CHECKS)
    )


def deterministic_fallback(
    plan: Plan, retrieval: RetrievalResult, knowledge: EditorialKnowledgeBase
) -> CuratedPlaylist:
    """Top-ranked candidates with template rationales, used when retries run out.

    This path involves no model at all, so it cannot hallucinate and cannot
    violate a hard constraint (the candidates already passed Pandas filtering).
    """
    chosen: List[Candidate] = retrieval.candidates[: plan.track_count]
    tracks = [
        CuratedTrack(
            song_id=candidate.song_id,
            title=candidate.title,
            artist=candidate.artist,
            reason=(
                f"Deterministic fallback pick: {candidate.genre} at "
                f"{candidate.tempo_bpm} BPM, energy {candidate.energy:.2f}, "
                f"valence {candidate.valence:.2f}. "
                + (knowledge.track_note_text(candidate.song_id).split(".")[0] + "." if knowledge.track_note_text(candidate.song_id) else "")
            ).strip(),
        )
        for candidate in chosen
    ]
    return CuratedPlaylist(
        playlist_name="Verified Fallback Selection",
        summary=(
            "The curator model could not produce a playlist that passed "
            "verification, so these are the top-ranked tracks from deterministic "
            "catalog filtering blended with editorial retrieval."
        ),
        tracks=tracks,
    )


def to_recommendations(
    playlist: CuratedPlaylist,
    catalog: pd.DataFrame,
    knowledge: EditorialKnowledgeBase,
) -> List[Recommendation]:
    """Join verified tracks back onto catalog rows for display."""
    index = _catalog_index(catalog)
    recommendations: List[Recommendation] = []
    for track in playlist.tracks:
        row = index.get(track.song_id)
        if row is None:
            continue
        recommendations.append(
            Recommendation(
                song_id=track.song_id,
                title=str(row["title"]),
                artist=str(row["artist"]),
                genre=str(row["genre"]),
                mood=str(row["mood"]),
                tempo_bpm=int(row["tempo_bpm"]),
                energy=float(row["energy"]),
                reason=track.reason,
                editorial_note=knowledge.track_note_text(track.song_id),
            )
        )
    return recommendations
