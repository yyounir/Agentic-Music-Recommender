# AI Interactions, Agent Traces & Reflection

This document covers the stretch features attempted in the upgrade of
`ai110-module3show-musicrecommendersimulation-starter` into the **Agentic Music
Recommender & Verification Engine**:

- **SF8 — Agentic Workflow** (§3)
- **SF10 — Design Pattern** (§4)
- Saved multi-step agent reasoning traces (§1)
- Reflection on AI collaboration and system design (§2)

---

## 1. Agent Reasoning Traces

Every run of the agent appends a complete reasoning record to
[`logs/agent_traces.json`](logs/agent_traces.json), following the chain:

```text
User Query
  -> Parsed Plan
  -> RAG Retrieved Context
  -> Initial Candidate Selection
  -> Guardrail Check Result
  -> Self-Correction Retry (if triggered)
  -> Final Output
```

The log is a single valid JSON array, written atomically and capped at the 200
most recent traces. The same trace is rendered live in the Streamlit UI under
**Agent reasoning trace**, and on the CLI by every `--demo`, `--guardrail-demo`
and `--query` run.

Four representative traces follow, reproduced verbatim from real runs. Traces B
and D use `FaultInjectingEngine`, which deliberately corrupts the curator's
response so the detection and recovery path can be demonstrated — the failure is
injected, but every line of detection, feedback and recovery is production code.

Regenerate any of these with:

```bash
python -m src.main --demo
python -m src.main --guardrail-demo
python -m src.main --query "Give me 5 songs above 200 BPM."
```

---

### Trace A - niche mood request, clean pass

**Trace `trace-97722f6b9087`** - provider `offline` (`heuristic-v1`), 14 ms

**1. User Query** -> `It is raining and I am feeling nostalgic. Give me 3 acoustic songs that are wistful without being depressing.`

**2. Parsed Plan**

```json
{
  "moods": [
    "nostalgic"
  ],
  "genres": [],
  "listening_contexts": [
    "nostalgia"
  ],
  "exclude_genres": [],
  "exclude_artists": [],
  "exclude_moods": [],
  "min_bpm": null,
  "max_bpm": null,
  "min_energy": null,
  "max_energy": null,
  "acoustic_preference": "prefer",
  "track_count": 3,
  "rationale": "Parsed request as context: nostalgia; moods: nostalgic",
  "unsupported_requests": []
}
```

**3. Multi-Source RAG Retrieval**

- Pandas filters: `none (no hard constraints in plan)`
- Catalog rows satisfying every hard constraint: 17/17
- Editorial passages retrieved:
  - `[context_note] nostalgia` (score 1)
  - `[track_note] Desert Postcard - Juniper Trail` (score 0.8575)
  - `[track_note] Library Rain - Paper Lanterns` (score 0.3381)
  - `[track_note] Paper Kites - Wild Fern` (score 0.3288)
  - `[track_note] Velvet Weekend - The Marigold Club` (score 0.1768)
- Ranked candidate pool (feature + editorial score):
  - `#11 Desert Postcard` total=9.45 (features 6.95 + editorial 2.5)
  - `#17 Paper Kites` total=5.2586 (features 4.3 + editorial 0.9586)
  - `#16 Blue Hour Train` total=4.0775 (features 4.0775 + editorial 0)
  - `#7 Coffee Shop Stories` total=2.5125 (features 2.5125 + editorial 0)
  - `#9 Focus Flow` total=2.3175 (features 2.3175 + editorial 0)
  - `#4 Library Rain` total=2.2923 (features 1.3067 + editorial 0.9856)

**4. Initial Candidate Selection**

  - `#11` Desert Postcard - Juniper Trail
  - `#17` Paper Kites - Wild Fern
  - `#16` Blue Hour Train - Harborlight
- Guardrail check: **PASSED** (format, hallucination, constraint, grounding)

**6. Final Output**

- Playlist: *Nostalgia Set*
- Retries: 0 | Fallback used: False
  - `#11` **Desert Postcard** - Juniper Trail (country, 96 BPM)
  - `#17` **Paper Kites** - Wild Fern (folk, 101 BPM)
  - `#16` **Blue Hour Train** - Harborlight (blues, 84 BPM)


---

### Trace B - hallucinated track caught by the verifier, then self-corrected

**Trace `trace-810ce2800495`** - provider `offline+fault-injection` (`heuristic-v1`), 6 ms

**1. User Query** -> `Give me 3 chill songs for a rainy afternoon.`

**2. Parsed Plan**

```json
{
  "moods": [
    "chill"
  ],
  "genres": [],
  "listening_contexts": [],
  "exclude_genres": [],
  "exclude_artists": [],
  "exclude_moods": [],
  "min_bpm": null,
  "max_bpm": null,
  "min_energy": null,
  "max_energy": null,
  "acoustic_preference": "any",
  "track_count": 3,
  "rationale": "Parsed request as moods: chill",
  "unsupported_requests": []
}
```

**3. Multi-Source RAG Retrieval**

- Pandas filters: `none (no hard constraints in plan)`
- Catalog rows satisfying every hard constraint: 17/17
- Editorial passages retrieved:
  - `[track_note] Library Rain - Paper Lanterns` (score 0.3381)
  - `[track_note] Coffee Shop Stories - Slow Stereo` (score 0.3244)
  - `[track_note] Blue Hour Train - Harborlight` (score 0.1644)
  - `[track_note] Paper Kites - Wild Fern` (score 0.1644)
- Ranked candidate pool (feature + editorial score):
  - `#4 Library Rain` total=5.65 (features 3.15 + editorial 2.5)
  - `#6 Spacewalk Thoughts` total=3.1625 (features 3.1625 + editorial 0)
  - `#2 Midnight Coding` total=3.14 (features 3.14 + editorial 0)
  - `#7 Coffee Shop Stories` total=2.5768 (features 0.1775 + editorial 2.3993)
  - `#17 Paper Kites` total=1.4207 (features 0.205 + editorial 1.2157)
  - `#16 Blue Hour Train` total=1.2882 (features 0.0725 + editorial 1.2157)

**4. Initial Candidate Selection**

  - `#999` Neon Cathedral - The Phantom Orchestra
  - `#4` Library Rain - Paper Lanterns
  - `#6` Spacewalk Thoughts - Orbit Bloom
- Guardrail check: **FAILED**
  - [hallucination] song_id 999 ("Neon Cathedral" by "The Phantom Orchestra") does not exist in data/songs.csv. Replace it with a track from the candidate list.

**5. Self-Correction Retry 1**

  - `#4` Library Rain - Paper Lanterns
  - `#6` Spacewalk Thoughts - Orbit Bloom
  - `#2` Midnight Coding - LoRoom
- Guardrail check: **PASSED** (format, hallucination, constraint, grounding)

**6. Final Output**

- Playlist: *Chill Mix*
- Retries: 1 | Fallback used: False
  - `#4` **Library Rain** - Paper Lanterns (lofi, 72 BPM)
  - `#6` **Spacewalk Thoughts** - Orbit Bloom (ambient, 60 BPM)
  - `#2` **Midnight Coding** - LoRoom (lofi, 78 BPM)


---

### Trace C - unsatisfiable constraint, explicit empty result

**Trace `trace-d23eb0a81574`** - provider `offline` (`heuristic-v1`), 5 ms

**1. User Query** -> `Give me 5 songs above 200 BPM.`

**2. Parsed Plan**

```json
{
  "moods": [],
  "genres": [],
  "listening_contexts": [],
  "exclude_genres": [],
  "exclude_artists": [],
  "exclude_moods": [],
  "min_bpm": 200,
  "max_bpm": null,
  "min_energy": null,
  "max_energy": null,
  "acoustic_preference": "any",
  "track_count": 5,
  "rationale": "Parsed request as tempo: 200..- BPM",
  "unsupported_requests": []
}
```

**3. Multi-Source RAG Retrieval**

- Pandas filters: `tempo_bpm >= 200`
- Catalog rows satisfying every hard constraint: 0/17
- Notice: No catalog track satisfies every hard constraint. The constraints were not relaxed; the closest tracks are reported as near misses only.
- Editorial passages retrieved:
  - `[track_note] Velvet Weekend - The Marigold Club` (score 0.1768)
  - `[track_note] Desert Postcard - Juniper Trail` (score 0.1715)
  - `[track_note] Thunder Bloom - Iron Orchard` (score 0.1715)
  - `[track_note] Storm Runner - Voltline` (score 0.169)
  - `[track_note] Library Rain - Paper Lanterns` (score 0.169)
- Ranked candidate pool: **empty** - no track satisfies the hard constraints, so no track is eligible.
- Near misses (reported to the user, never recommended):
  - `#15 Velvet Weekend` (funk, 112 BPM, energy 0.79)
  - `#11 Desert Postcard` (country, 96 BPM, energy 0.54)
  - `#10 Rooftop Lights` (indie pop, 124 BPM, energy 0.76)

**4. Initial Candidate Selection**

- Skipped. With an empty candidate pool no generated answer could pass verification, so the agent short-circuited and made **zero model calls**.

**6. Final Output**

- Playlist: *No Matching Tracks*
- Retries: 0 | Fallback used: False
  - *(no tracks returned)*


---

### Trace D - retries exhausted, deterministic fallback

**Trace `trace-4f9a4fdca9f6`** - provider `offline+fault-injection` (`heuristic-v1`), 11 ms

**1. User Query** -> `Three chill songs for a rainy afternoon.`

**2. Parsed Plan**

```json
{
  "moods": [
    "chill"
  ],
  "genres": [],
  "listening_contexts": [],
  "exclude_genres": [],
  "exclude_artists": [],
  "exclude_moods": [],
  "min_bpm": null,
  "max_bpm": null,
  "min_energy": null,
  "max_energy": null,
  "acoustic_preference": "any",
  "track_count": 3,
  "rationale": "Parsed request as moods: chill",
  "unsupported_requests": []
}
```

**3. Multi-Source RAG Retrieval**

- Pandas filters: `none (no hard constraints in plan)`
- Catalog rows satisfying every hard constraint: 17/17
- Editorial passages retrieved:
  - `[track_note] Library Rain - Paper Lanterns` (score 0.3381)
  - `[track_note] Coffee Shop Stories - Slow Stereo` (score 0.3244)
  - `[track_note] Blue Hour Train - Harborlight` (score 0.1644)
  - `[track_note] Paper Kites - Wild Fern` (score 0.1644)
- Ranked candidate pool (feature + editorial score):
  - `#4 Library Rain` total=5.65 (features 3.15 + editorial 2.5)
  - `#6 Spacewalk Thoughts` total=3.1625 (features 3.1625 + editorial 0)
  - `#2 Midnight Coding` total=3.14 (features 3.14 + editorial 0)
  - `#7 Coffee Shop Stories` total=2.5768 (features 0.1775 + editorial 2.3993)
  - `#17 Paper Kites` total=1.4207 (features 0.205 + editorial 1.2157)
  - `#16 Blue Hour Train` total=1.2882 (features 0.0725 + editorial 1.2157)

**4. Initial Candidate Selection**

  - `#999` Neon Cathedral - The Phantom Orchestra
  - `#4` Library Rain - Paper Lanterns
  - `#6` Spacewalk Thoughts - Orbit Bloom
- Guardrail check: **FAILED**
  - [hallucination] song_id 999 ("Neon Cathedral" by "The Phantom Orchestra") does not exist in data/songs.csv. Replace it with a track from the candidate list.

**5. Self-Correction Retry 1**

  - `#999` Neon Cathedral - The Phantom Orchestra
  - `#4` Library Rain - Paper Lanterns
  - `#6` Spacewalk Thoughts - Orbit Bloom
- Guardrail check: **FAILED**
  - [hallucination] song_id 999 ("Neon Cathedral" by "The Phantom Orchestra") does not exist in data/songs.csv. Replace it with a track from the candidate list.

**5. Self-Correction Retry 2**

  - `#999` Neon Cathedral - The Phantom Orchestra
  - `#4` Library Rain - Paper Lanterns
  - `#6` Spacewalk Thoughts - Orbit Bloom
- Guardrail check: **FAILED**
  - [hallucination] song_id 999 ("Neon Cathedral" by "The Phantom Orchestra") does not exist in data/songs.csv. Replace it with a track from the candidate list.

**5. Self-Correction Retry 3**

  - `#999` Neon Cathedral - The Phantom Orchestra
  - `#4` Library Rain - Paper Lanterns
  - `#6` Spacewalk Thoughts - Orbit Bloom
- Guardrail check: **FAILED**
  - [hallucination] song_id 999 ("Neon Cathedral" by "The Phantom Orchestra") does not exist in data/songs.csv. Replace it with a track from the candidate list.

**6. Final Output**

- Playlist: *Verified Fallback Selection*
- Retries: 3 | Fallback used: True
  - `#4` **Library Rain** - Paper Lanterns (lofi, 72 BPM)
  - `#6` **Spacewalk Thoughts** - Orbit Bloom (ambient, 60 BPM)
  - `#2` **Midnight Coding** - LoRoom (lofi, 78 BPM)


---
## 2. Reflection on AI Collaboration & System Design

### (a) How AI was used during design and debugging

I used an AI coding agent (Claude, via Claude Code) as a design partner and
implementation pair across the whole upgrade, not as an autocomplete. The work
split roughly four ways.

**Schema-first design.** Before writing any logic I worked with the AI to settle
the Pydantic boundaries in `src/schemas.py` — `Plan`, `RetrievalResult`,
`CuratedPlaylist`, `GuardrailReport`, `AgentTrace`. Deciding *what each step is
allowed to hand the next one* turned out to be the architectural decision that
made everything else easy. Once `Plan` existed with validators on BPM bounds,
energy bounds and track count, a whole category of bug ("the model said
`min_bpm=150, max_bpm=90`") became impossible rather than something to defend
against downstream.

**Prompt authoring.** The curator persona in `src/prompts.py` went through several
rounds. The first version was a paragraph of description. It produced the right
*kind* of answer but not reliably the right *shape*, so the AI suggested replacing
description with demonstration: two complete worked examples, including one that
shows what a correct exclusion looks like. The measured effect is in §6 of
[`model_card.md`](model_card.md) — format adherence on the first response moved
from 17% to 100%, and rationale grounding from 0% to 100%.

**Debugging.** Two bugs are worth recording because they were both found by
running the system rather than by reading it:

- The natural-language track-count parser initially read *"I want 3 K-pop and trap
  songs"* as five tracks, because the regex only allowed two words between the
  number and the noun. The fix allows a bounded gap but explicitly excludes digits
  and the token `bpm`, so that *"everything has to be at least 120 BPM and 4 songs"*
  can never read `120` as a track count. This was caught by case `A2` in the
  evaluation harness, not by a unit test I had thought to write.
- The deterministic engine crashed on the first real query because I had written
  `any(re.search(...))` — `re.search` returns a `Match` or `None`, neither of which
  is iterable. A trivial bug, but it is the reason the whole pipeline now has a
  smoke test in `--demo` that runs before anything is documented.

**What only the live model could find.** The system was built and tested against
the deterministic engine first, then run against Gemini. Four things surfaced that
no amount of offline testing would have:

- **The planner invented a value outside a closed vocabulary.** Asked for a
  "late-night coding session", Gemini set
  `listening_contexts=['late-night coding session']` — plausible, and matching none
  of the eight contexts the editorial notes define. It was nearly harmless, because
  retrieval also matches context notes against the raw query text, so the right
  note was still retrieved. But the plan was carrying a value nothing downstream
  could use. I had typed the field as `List[str]` and written the *vocabulary* only
  for genres and moods. The fix was to pass the closed context list in the prompt;
  the lesson was that "typed" and "constrained" are not the same thing.
- **User-facing text needs to be specified as user-facing.** `unsupported_requests`
  entries are shown directly to the listener, and Gemini filled one with the bare
  fragment `instrumental / no vocals preference`, which rendered as a notice
  reading exactly that. The schema said what the field *was* but not what it would
  be *used for*. The prompt now requires complete sentences and gives a good and a
  bad example.
- **The self-correction loop fired for real.** `gemini-3.1-flash-lite` returned the
  same `song_id` twice on one prompt. The duplicate check existed because it was
  cheap to write, not because I had seen a model do it. It caught the first
  genuine, un-injected violation in the project's life.
- **A dead planner is worse than a wrong one.** This is the finding I would keep if
  I could keep only one. When the free-tier daily quota ran out mid-benchmark, the
  planner returned nothing, and my code degraded to an *empty* plan. An empty plan
  has no constraints; the constraint guardrail checks the answer against the plan;
  so *"Give me 5 songs above 200 BPM"* returned five tracks between 66 and 124 BPM,
  with all four guardrails reporting **PASS**. The verification layer I had spent
  most of the project building was structurally incapable of catching it, because
  **Step 4 can only enforce what Step 1 recorded**. I had written the degradation
  path early, when it looked like defensive hygiene, and never asked what it would
  actually produce. Planner failure now falls back to the rule-based heuristic
  planner — which needs no network and does extract `min_bpm=200` — and only to an
  unconstrained plan if that fails too, in which case the user is explicitly
  warned. The regression test simulates that exact 429.

**Evaluation design.** The AI pushed hardest here, and correctly. My instinct was
to assert "the recommendations look right". What went in instead was a benchmark
where each case carries *named, individually-scored assertions* split into
`constraint` and `behaviour` kinds, so "Constraint Adherence: 100% (14/14)" means
something specific and a regression points at the exact assertion that broke.

### (b) An AI suggestion I accepted, and why

**Accepted: short-circuit unsatisfiable requests instead of retrying them.**

My first implementation of "no track matches" followed the obvious path from the
spec: when the Pandas filter returned zero rows, fall back to ranking the whole
catalog and let the curator pick from it. Running
`--query "Give me 5 songs above 200 BPM"` showed what that actually produced:

```text
attempt 1 ... GUARDRAIL -> "Velvet Weekend" is 112 BPM, below the requested minimum of 200 BPM.
attempt 2 ... (identical five violations)
attempt 3 ... (identical five violations)
attempt 4 ... (identical five violations)
-> fallback used, guardrails FAILED, five tracks returned, every one of them
   violating the user's only stated constraint
```

Four model calls to produce an answer that was wrong, labelled `FAILED`, and
shown to the user anyway. The AI's point was that this is not a retry problem at
all — it is a *satisfiability* problem, and no amount of regeneration can fix it.
I accepted the redesign: hard constraints are never relaxed, an empty candidate
pool means the agent makes **zero model calls**, and the honest answer is an
explicit empty result plus the three closest tracks clearly labelled as near
misses rather than recommendations. Trace C above shows the result.

I accepted it because it is the behaviour I would want as a user. "Nothing in this
catalog goes above 168 BPM — the closest is Thunder Bloom at 168" is a useful
answer. Five songs at half the requested tempo, stamped with a red FAILED badge,
is not.

**Also accepted: fault injection as a first-class component.** The deterministic
engine can never hallucinate, because it only ever selects from the candidate pool.
That is good for correctness and terrible for testing — it meant the entire
self-correction loop was dead code I could not demonstrate. `FaultInjectingEngine`
wraps any engine and corrupts its first *n* curation responses on demand, which is
what makes `python tests/eval_harness.py --inject-faults` and the
`--guardrail-demo` command possible, and what lets the test suite assert that a
fabricated track is caught on attempt 1 and absent from the final output.

### (c) An AI suggestion I rejected or revised, and why

**Rejected: letting the model name songs and post-filtering the result.**

An early suggestion was to let the curator generate freely from the whole catalog
(or, in one variant, from its own knowledge of music) and then drop anything that
failed verification. It is simpler to build and it reads well in a demo.

I rejected it for two reasons.

First, **post-filtering gets the ordering of trust backwards.** If the model
returns five tracks and two are invalid, dropping them leaves a three-track
playlist that no longer matches what was asked for, and the summary describing the
playlist's "arc" now describes tracks that are not in it. The system has to
*re-ask*, not quietly truncate. That is why the loop feeds the exact violation
strings back and demands a complete replacement answer, and why `format` is a
guardrail in its own right rather than something fixed up silently.

Second, **grounding should be structural rather than corrective.** Restricting the
input so the model is only ever *shown* the legal candidates, and then proving the
output against `songs.csv`, bounds the failure in a way that a prompt instruction
plus a post-filter does not. The candidate pool is the real control; the persona
and the verifier are the second and third lines.

**A correction I owe this section.** When I first wrote the paragraph above, my
evidence was the simulated baseline in model-card §6, which fabricated "Bohemian
Rhapsody — Queen" when handed the full catalog and a generic prompt. Running the
real Gemini-vs-Gemini A/B later overturned that evidence: given the whole catalog
and a two-sentence instruction, the live model fabricated **nothing** across 19
tracks, honoured every exclusion and BPM bound, and returned the right track count
every time. The dramatic gaps my simulation predicted — 17% format adherence, 20%
hallucination — did not replicate at all.

So the architectural argument still holds, but it is weaker and differently shaped
than I claimed. The candidate pool and the verifier bound the worst case at
essentially no runtime cost, and the same model produced a duplicate `song_id` on a
benchmark prompt that same day, so it is not incapable of malformed output. But I
cannot claim they *prevented* hallucinations that a plain prompt would have caused,
because on this evidence a plain prompt did not cause any. They are insurance, and
I over-sold them on the strength of a baseline I had written myself.

That is the real lesson, and it is a methodological one: **a strawman you implement
confirms whatever you believed when you implemented it.** The honest comparison
cost 24 API calls and overturned the headline claim of a section I had already
written. Both tables are kept in the model card, the simulated one explicitly
labelled, because the gap between them is more instructive than either alone.

**Revised rather than rejected: the grounding check.** The first version flagged
any rationale that did not contain a digit. That is cheap and it does catch "this
song is a great pick", but it also rejects a perfectly good sentence like *"its
rain-streaked, resigned tone suits a wet night"*. I revised it to accept either a
numeric citation **or** a token drawn from that specific track's genre, mood,
editorial note or mood descriptors — so the check asks "is this anchored in
something real about *this* track", which is the actual requirement, rather than
"does this contain a number", which was a proxy for it.

### (d) Current limitations and future improvements

**Limitations I would flag to anyone building on this.**

1. **The catalog is 17 songs and most genres have exactly one representative.**
   Several benchmark cases are effectively unanswerable at full quality — the
   night-drive prompt tops out at precision@3 = 0.33 because there is one synthwave
   track in existence here. Retrieval quality cannot exceed catalog coverage.
2. **The Gemini benchmark is one run on a free-tier key.** 10/10 with 0%
   hallucination is real, but it is a single pass over 10 prompts, on a mix of
   models forced by rate limiting, with no seed variation. `Average Retries: 0.00`
   means *these ten prompts did not provoke a violation that day* — not that
   violations are rare. Free-tier quota (20 requests/day on `gemini-3.6-flash`,
   two per benchmark case) makes repeated measurement genuinely difficult.
3. **The editorial knowledge base is one person's taste, applied at a fixed
   weight.** §5 of the model card shows it helps on average (mean precision@3
   0.33 → 0.78) and hurts on at least one prompt. It is a new bias surface the base
   project did not have, and the system gives the user no way to turn it down.
4. **The guardrails prove grounding, not quality.** "GUARDRAILS: PASS" means the
   tracks exist and satisfy the stated constraints. It does not mean the playlist
   is good, and the badge is prominent enough that a user could reasonably over-read
   it.
5. **The offline planner is regular expressions.** It covers the benchmark's
   phrasings and degrades predictably outside them.

**Improvements, in the order I would do them.**

1. Run and publish the Gemini benchmark across several seeds, so the guardrail
   firing rate against a real model is a measured number rather than an inference.
2. Replace lexical retrieval over the editorial notes with embeddings, so
   *"music for grinding through a hard problem"* reaches the `deep focus` note
   without sharing a word with it.
3. Grow the catalog to the point where most requests have several defensible
   answers, which is the precondition for evaluating ranking at all.
4. Replace my hand-labelled ideal sets with multi-rater human judgments, so the
   RAG benefit is measured against something other than the taste of the person who
   wrote both the notes and the labels.
5. Expose diversity as a user control, to counteract the preference bias that
   heavy genre/mood weighting introduces.

---

## 3. Agentic Workflow (SF8)

**What task did you give the agent?**

Upgrade a single-file Pandas recommender into a production-shaped agentic system:
a four-step `Plan → Retrieve → Act → Verify` workflow over the Gemini API with
Pydantic-typed boundaries, a second retrieval source, guardrail verification with
a self-correction loop, structured trace logging, a Streamlit UI and CLI, a unit
test suite, a benchmark harness, and documentation written from real measured
outputs rather than estimates.

**Prompts used (the load-bearing ones):**

- *"Transform the baseline recommender into a multi-step agentic workflow with
  multi-source RAG and specialised persona prompting, using Pydantic for
  structured outputs at every step."*
- *"Run deterministic checks on the model's playlist before returning it: verify
  every recommended song_id and title+artist exists verbatim in songs.csv, verify
  tracks satisfy user constraints, and feed the exact validation errors back for up
  to 3 self-correction retries."*
- *"Make the retrieved editorial notes actively shape candidate selection and
  ranking, not just get tacked onto the final output."*
- *"Build an evaluation harness over 10 benchmark prompts including adversarial
  cases — impossible BPM constraints, nonexistent genres, prompt-injection
  attempts, empty results — and report pass rate, constraint adherence,
  hallucination rate and average retries."*
- *"Run the demo, tests and harness to capture real outputs, then write the docs
  from those outputs."*

**What did the agent generate or change?**

New: `src/schemas.py`, `src/retrieval.py`, `src/prompts.py`, `src/llm_client.py`,
`src/guardrails.py`, `src/tracing.py`, `data/editorial_notes.json`,
`tests/eval_harness.py`, `tests/ablations.py`, `architecture.mmd`,
`run_app.py`, `logs/agent_traces.json`.
Rewritten: `src/main.py` (Streamlit UI + CLI), `README.md`, `model_card.md`, this
file, `requirements.txt`.
Extended in place: `src/recommender.py` (the baseline scorer is untouched;
`MusicCuratorAgent` was added beneath it) and `tests/test_recommender.py` (the five
starter tests are unchanged; 46 more were added).

**What did you verify or fix manually?**

- Every number quoted in `README.md` and `model_card.md` was produced by running
  the command shown next to it; nothing is estimated.
- The claim *"editorial notes change the ranking"* was checked directly rather than
  assumed — the before/after lists in §5 of the model card come from
  `python tests/ablations.py --rag`, and the one prompt where RAG makes things
  **worse** is reported rather than dropped.
- The Streamlit app was exercised headlessly with `streamlit.testing.v1.AppTest`
  to confirm it renders, accepts a query and produces metrics without exceptions.
- The zero-shot arm in model-card §6 began as a *simulation*, built because no API
  key was available, and was labelled as simulated everywhere it appeared rather
  than passed off as a Gemini-vs-Gemini comparison. When a key arrived, the real
  A/B was run — and it contradicted the simulation on every headline metric. The
  real numbers now lead §6.1 and the simulated ones are kept in §6.2 as a worked
  example of a baseline confirming its author's assumptions. The easy move would
  have been to quietly delete the table that no longer held.
- Claims in the docs were re-checked against reruns rather than left as written.
  A README example quoting a Gemini run was regenerated after a prompt fix so the
  pasted output matches the current code, and the superseded run is described in a
  note instead of silently replaced.
- The two guardrail demos that need a failing model response are labelled
  "fault-injected" so no reader mistakes an injected failure for an observed one.

---

## 4. Design Pattern (SF10)

**Which design pattern did you use?**

Primarily **Strategy**, with supporting **Decorator** and **Chain of
Responsibility**.

**How did AI help you brainstorm or implement it?**

The problem that forced the pattern was practical: the project had to run with no
`GEMINI_API_KEY` (for tests, for CI, for anyone cloning the repo) and with one (for
real use), and the two paths had to be the same system rather than a real
implementation plus a stub. The AI's framing was that the *generation back-end* is
the only thing that varies, so it should be the only thing behind an interface —
everything else (planning contract, retrieval, persona contract, guardrails, retry
loop, fallback, tracing) is invariant and should be written once.

**How does the pattern appear in the final code?**

- **Strategy** — `src/llm_client.py` defines a two-method `Engine` protocol
  (`plan`, `curate`). `GeminiEngine` implements it against the Gemini API with
  Pydantic response schemas; `OfflineHeuristicEngine` implements it with
  deterministic rules. `MusicCuratorAgent` depends only on the protocol, and
  `build_engine()` selects the strategy at runtime, emitting a user-visible notice
  on any downgrade. The engine used is recorded in every trace, so no output is
  ambiguous about which path produced it.
- **Decorator** — `FaultInjectingEngine` wraps *any* `Engine`, delegates both
  methods, and corrupts the first *n* curation responses. Because it satisfies the
  same protocol it can be dropped in anywhere an engine goes: the guardrail demo,
  the benchmark's `--inject-faults` mode, the Streamlit sidebar checkbox, and four
  unit tests. `ZeroShotGeminiEngine` in `tests/ablations.py` is a second decorator,
  reusing the Gemini transport with a deliberately generic prompt so the prompt
  ablation changes exactly one variable.
- **Chain of Responsibility** — `verify_playlist()` runs four independent checks
  (`format`, `hallucination`, `constraint`, `grounding`), each appending typed
  `GuardrailViolation` objects to one report. No check short-circuits the others,
  so a single pass reports *every* problem at once and the self-correction prompt
  can ask for all of them to be fixed together rather than discovering them one
  retry at a time.
