# 🎧 Model Card: Agentic Music Recommender & Verification Engine

**Version:** 1.0 · **Date:** 2026-11-13 · **Base project:**
`ai110-module3show-musicrecommendersimulation-starter`

---

## 1. Model Name

**Agentic Music Recommender & Verification Engine (AMR-V 1.0)**

This is not a trained model. It is a *system*: a four-step agentic workflow built
around a general-purpose LLM (Google Gemini, `gemini-3.6-flash` by default, with
automatic fallback to `gemini-3.5-flash` and `gemini-3.1-flash-lite`), a
17-track catalog, a hand-authored editorial knowledge base, and a deterministic
verification layer. The system's behaviour is defined as much by its prompts,
retrieval and guardrails as by the underlying model.

---

## 2. Intended Use

**What it does.** Turns a natural-language listening request
("three lofi tracks under 90 BPM for a late-night coding session") into a short,
ordered playlist drawn exclusively from `data/songs.csv`, with a one-to-two
sentence rationale per track grounded in that track's metadata and editorial
notes, plus a full audit trail of how it got there.

**Who it is for.** Students and engineers studying agentic workflows, retrieval-
augmented generation and LLM guardrails. It is a **classroom and portfolio
artifact, not a production recommender**: the catalog is 17 songs, there is no
user account, no listening history, no personalisation and no audio.

**Assumptions about the user.** That they state preferences in plain language;
that constraints they state (tempo, excluded genres, playlist length) are *hard*
requirements rather than hints; and that they would rather be told "nothing in the
catalog matches" than handed a near-miss presented as a match.

**Out of scope.** Commercial recommendation, music discovery beyond this catalog,
any claim about artists, lyrics, chart performance or release dates (the system
has none of that data and the grounding guardrail actively discourages the model
from inventing it), and any use where a wrong recommendation carries real cost.

---

## 3. How the Model Works

Imagine asking a knowledgeable record-shop clerk for something to listen to, and
imagine that behind the clerk stands a fact-checker who reads every answer before
it leaves the shop.

**Step 1 — understand the request.** The system reads your sentence and fills in a
form: which moods, which genres, what situation, what you ruled out, any tempo or
energy limits, how many songs. The form has strict fields, so vague or impossible
answers get caught at the door. Anything in your message that looks like an
instruction to the system itself ("ignore your rules") is written down as *refused*
rather than followed.

**Step 2 — find the candidates.** Two sources are consulted. The first is the
spreadsheet of songs: anything that breaks a hard limit you set is removed
immediately, by arithmetic, with no model involved. The second is a set of written
curation notes — what each track feels like, what it is for, what to avoid it for,
and what kinds of music suit situations like "deep focus" or "heartbreak". Both
sources feed one ranking, so the notes change *which* songs are considered, not
just what gets said about them.

**Step 3 — curate.** A model playing a constrained "music curator" role is shown
the request, the retrieved notes, and the candidate list — and nothing else. It is
told, with worked examples, to pick from that list only, to copy titles and artists
exactly, and to justify each pick with something concrete: a tempo, an energy
value, a mood, a line from the notes.

**Step 4 — verify, and correct.** Before anything is shown, four checks run. Does
every song actually exist in the spreadsheet, with that exact title and artist? Did
it respect your exclusions and limits? Is the list the right length with no
repeats? Does each explanation cite something real? If any check fails, the exact
complaints are handed back to the model and it tries again, up to three times. If
it still cannot produce a clean answer, the system discards the model's output
entirely and returns the top-ranked candidates with template explanations —
answers that cannot be wrong because no model wrote them.

**What changed from the starter.** The starter compared a hard-coded preference
dictionary against each row and printed the arithmetic. Everything above —
natural language in, the second knowledge source, the curator persona, the four
checks, the correction loop, the fallback, the trace log — is new. The original
scoring engine is still present and still tested, and runs via
`python -m src.main --baseline`.

---

## 4. Data

### 4.1 `data/songs.csv` — the catalog (unchanged from the starter)

17 tracks. Columns: `id, title, artist, genre, mood, energy, tempo_bpm, valence,
danceability, acousticness`.

- **Genres (14):** pop, lofi, rock, ambient, jazz, synthwave, indie pop, country,
  reggae, classical, metal, funk, blues, folk.
- **Moods (13):** happy, chill, intense, relaxed, focused, moody, nostalgic,
  playful, serene, defiant, sensual, melancholic, hopeful.
- **Tempo range:** 60–168 BPM. **Energy range:** 0.22–0.96.

Most genres are represented by exactly **one** track. That is the single most
important fact about this dataset: for many requests there is only one defensible
answer, and for some there is none.

### 4.2 `data/editorial_notes.json` — new secondary knowledge source

Hand-authored for this upgrade, written against the exact tracks in the CSV:

- **17 track notes** — mood descriptors ("rain-streaked", "glacial", "golden-hour"),
  listening contexts ("debugging session", "processing bad news"), a prose
  editorial note, `avoid_when`, and `pairs_well_with`.
- **14 genre notes** — how each genre functions in *this* catalog.
- **8 context notes** — "deep focus", "workout", "late night drive", "wind down",
  "hosting", "heartbreak", "mood lift", "nostalgia" — each with aliases, preferred
  and discouraged genres, and a preferred BPM band.

**What this data does not contain:** audio, lyrics, language, release year,
popularity, artist biography, subgenre, instrumentation, or anything about the
listener. Musical taste is far richer than ten columns and a paragraph.

---

## 5. Before vs. After RAG

Does the editorial source actually improve recommendations, or does it just make
the prose nicer? Reproduce with `python tests/ablations.py --rag`.

**Method.** Six niche prompts are run twice: once with the full editorial
knowledge base and once with an empty one. The *only* variable is whether the
editorial layer informs retrieval and ranking — the planner, the catalog and the
scoring code are identical in both arms. Quality is scored as **precision@3**
against ideal sets hand-labelled from the CSV's numeric columns (tempo, energy,
valence, acousticness) and **not** from the editorial notes, so the metric is not
circular. The labelling basis for each prompt is recorded in `tests/ablations.py`.

| Niche prompt | Top-3 **without** editorial RAG | Top-3 **with** editorial RAG | P@3 before | P@3 after |
|---|---|---|---:|---:|
| "Something for a deep focus work session." | Focus Flow, Island Rhythm, Sunrise City | Focus Flow, Midnight Coding, Spacewalk Thoughts | 0.33 | **1.00** |
| "Late night drive through the city — neon, overcast, not cheerful." | Island Rhythm, Paper Kites, Gym Hero | Night Drive Loop, Island Rhythm, Paper Kites | 0.00 | **0.33** |
| "I just got bad news and I want to sit with it. Genuinely sad songs." | Blue Hour Train, Island Rhythm, Sunrise City | Blue Hour Train, Glass Cathedral, Desert Postcard | 0.33 | **1.00** |
| "Background music for a dinner party — nothing too loud." | Island Rhythm, Sunrise City, Paper Kites | Coffee Shop Stories, Velvet Weekend, Island Rhythm | 0.33 | **1.00** |
| "Lift my mood without hyping me up." | Island Rhythm, Sunrise City, Paper Kites | Velvet Weekend, Island Rhythm, Paper Kites | **1.00** | 0.67 |
| "I need to wind down before sleep." | Island Rhythm, Sunrise City, Paper Kites | Glass Cathedral, Library Rain, Midnight Coding | 0.00 | **0.67** |
| **Mean precision@3** | | | **0.33** | **0.78** |

**Reading the table.**

- **Selection changed on 6/6 prompts.** The editorial layer is not cosmetic.
- **The failure mode it fixes is specific.** Without curation notes, mood prompts
  collapse onto the same three high-valence tracks (`Island Rhythm`, `Sunrise
  City`, `Paper Kites`) regardless of what was asked, because raw valence is the
  only signal the features offer for "how does this feel". Three different prompts —
  "wind down before sleep", "background music for a dinner party" and "lift my mood
  without hyping me up" — returned the *identical* top-3 list before RAG
  (`Island Rhythm, Sunrise City, Paper Kites`). The system could not tell those
  requests apart. Afterwards the sleep list (`Glass Cathedral, Library Rain,
  Midnight Coding`) shares no track with either of the other two.
- **One honest regression.** "Lift my mood without hyping me up" drops from 1.00
  to 0.67: the `mood lift` context note promotes `Velvet Weekend` (funk, high
  danceability) into first place, displacing `Sunrise City`. That is a defensible
  curatorial call and a measurable precision loss. Editorial data is opinion, and
  opinion that helps on average can hurt on a specific prompt.
- **The night-drive prompt is catalog-limited, not retrieval-limited.** It improves
  from 0.00 to 0.33 because the catalog contains exactly one synthwave track; the
  other two slots have no good answer available at any retrieval quality.

---

## 6. Baseline Zero-Shot vs. Specialised Few-Shot

Reproduce with `python tests/ablations.py --prompt`.

**Method.** Both arms answer the same six prompts and are scored on their **first
response only**, before any self-correction, because the first response is what the
prompt itself is responsible for. Scoring uses the production guardrails.

- **Arm A — baseline zero-shot:** a terse, generic instruction ("you are a helpful
  assistant, reply with JSON"), no persona, no few-shot examples, and the **whole
  catalog** pasted in rather than a filtered candidate pool.
- **Arm B — specialised few-shot:** the shipped `CURATOR_SYSTEM_PROMPT` persona
  with its absolute rules, two worked few-shot examples, and the filtered,
  pre-ranked candidate pool as the only legal track universe.

### 6.1 Real Gemini-vs-Gemini A/B (`gemini-3.1-flash-lite`, 6 prompts)

```bash
GEMINI_MODEL=gemini-3.1-flash-lite python tests/ablations.py --prompt --engine gemini
```

| Metric (first response, before any correction) | A — zero-shot | B — specialised few-shot |
|---|---:|---:|
| Schema parse rate † | 100% | 100% |
| Format adherence (right count, no duplicates) | 100% | 100% |
| Constraint adherence (exclusions, BPM, energy) | 100% | 100% |
| Rationale grounding (cites real metadata or notes) | 100% | 100% |
| Hallucination rate (tracks absent from `songs.csv`) | 0% | 0% |
| **Mean rationale length** | **88 chars** | **142 chars** |
| **Rationales citing a concrete numeric feature** | **37%** | **100%** |

† Both arms were called with `response_mime_type="application/json"`, so the
parse rate is a property of the API call, not of the prompt. Only the rows in
bold represent a measured prompt effect.

**Reading the table — and it is not the result I expected.**

- **On correctness, the specialised prompt made no measurable difference.** Given
  the whole 17-track catalog and a two-sentence generic instruction,
  `gemini-3.1-flash-lite` returned the right number of tracks, honoured
  exclusions and BPM bounds, and fabricated nothing across 19 tracks. The persona,
  the few-shot examples and the filtered candidate pool bought **zero** measurable
  correctness on these six prompts.
- **The measurable effect is specificity.** Rationales citing a concrete number
  went from 37% to 100%, and mean length from 88 to 142 characters. The
  zero-shot arm wrote *"a relaxing track that fits the mood"*; the specialised arm
  wrote *"78 BPM with 0.42 energy and a muted Rhodes figure that never resolves"*.
  That is a real difference in usefulness, and it is the one the prompt earned.
- **This does not retire the candidate pool or the guardrails, but it does
  reframe them.** Six prompts, one run, one small model, zero observed
  hallucinations — that is weak evidence of safety, not proof of it. The same model
  returned a duplicate `song_id` on a benchmark prompt the same day (§7), so it is
  not incapable of malformed output. The guardrails cost nothing per request and
  bound the worst case; what this ablation shows is that on a modern model they are
  insurance rather than a load-bearing correction mechanism.

### 6.2 What a strawman baseline predicted, and why it was wrong

Before an API key was available, Arm A was a **simulated** baseline
(`NaiveCuratorEngine` in `tests/ablations.py`) built to reproduce the failure modes
an unconstrained prompt "obviously" exhibits. Run
`python tests/ablations.py --prompt` without a key and it still produces:

| Metric | A — *simulated* zero-shot | B — specialised few-shot |
|---|---:|---:|
| Format adherence | 17% | 100% |
| Constraint adherence | 67% | 100% |
| Rationale grounding | 0% | 100% |
| Hallucination rate | 20% | 0% |

**Every dramatic gap in that table failed to replicate against a real model.** The
simulation assumed a generic prompt would ignore the requested track count, drift
from stated constraints, and reach for a famous song outside the catalog
("Bohemian Rhapsody — Queen"). A current model did none of those things. The
simulated baseline was not a measurement of a weak prompt; it was a measurement of
my assumptions about one.

This table is kept, clearly labelled, because it is the most useful methodological
lesson in the project: **a baseline you implement yourself will confirm whatever
you believed when you wrote it.** The real comparison was cheap — 24 API calls —
and it overturned the headline claim.

---

## 7. Guardrail Behaviour

Four checks run on every candidate playlist, in `src/guardrails.py`. None involves
a model.

| Check | What it verifies | On failure |
|---|---|---|
| `format` | Right number of tracks (or the whole pool when it is smaller), no duplicate ids, schema-valid | Violation text names the expected count |
| `hallucination` | Every `song_id` exists in `data/songs.csv`, **and** the `title` and `artist` match that row verbatim | Violation names the id, the claimed title/artist and the catalog's actual values |
| `constraint` | Excluded genres / moods / artists absent; BPM and energy within the plan's bounds | Violation names the track and the specific bound it broke |
| `grounding` | Each rationale is non-trivial and cites a real feature, mood, genre or phrase from that track's editorial note | Violation asks for a concrete feature |

**Self-correction.** On any failure the violations are fed back verbatim, together
with the candidate pool, for up to **three** retries.

**Fallback.** If all four attempts fail, the model's output is discarded entirely
and `deterministic_fallback()` returns the top-ranked candidates with template
rationales. That output is itself verified before return. It cannot hallucinate
(the ids come from the catalog) and cannot violate a hard constraint (the
candidates already passed Pandas filtering).

**Unsatisfiable requests.** When deterministic filtering leaves zero rows, no
generated answer could pass verification. The agent short-circuits: **zero model
calls**, an explicit empty result, and a notice naming the unmet constraint plus
the three closest tracks, clearly labelled as near misses.

**Prompt injection.** The planner treats user text strictly as data, records
embedded instructions in `unsupported_requests` instead of following them, and
surfaces them as notices. If that pattern matching were bypassed, the
hallucination check remains the backstop: a track not in `songs.csv` cannot be
returned regardless of what the prompt says.

**Measured on the 10-prompt benchmark** (`python tests/eval_harness.py`):

| | Offline, clean | Offline, injected faults | **Real Gemini** |
|---|---|---|---|
| Pass rate | 10/10 | 10/10 | **10/10** |
| Constraint adherence | 100% (14/14) | 100% (14/14) | **100% (14/14)** |
| Hallucination rate at output | 0% (0/28) | 0% (0/28) | **0% (0/30)** |
| Average retries | 0.00 | 0.90 | 0.00 |
| Fallbacks used | 0/10 | 0/10 | 0/10 |

In the fault-injected run, 9 of 10 cases received a fabricated track in the first
curator response; all 9 were caught and repaired in a single retry. (The tenth is
unsatisfiable and never reaches the model.)

The Gemini column is a **mixed-model run**: free-tier rate limits meant the
fallback chain fired on every case, so `gemini-3.6-flash` answered two cases and
`gemini-3.1-flash-lite` the rest. Two findings from running against a live model
are worth recording because offline testing could not have produced them:

- **The self-correction loop fired for real.** On the niche-mood case `M2`,
  `gemini-3.1-flash-lite` returned the same `song_id` twice. The `format` check
  caught it and the retry returned a clean three-track playlist. That check was
  written because it was cheap, not because a model had ever failed it.
- **A dead planner was more dangerous than a wrong one.** When the free-tier
  **daily** quota (20 requests/day) was exhausted, every model returned `429` and
  the planner produced nothing. The agent degraded to an *unconstrained* plan, and
  since the constraint guardrail can only check what the plan recorded, *"5 songs
  above 200 BPM"* came back with five tracks between 66 and 124 BPM — a confident,
  guardrail-approved, completely wrong answer. **Verification is bounded by
  planning: Step 4 cannot catch what Step 1 dropped.** Planner failure now falls
  back to the deterministic heuristic planner rather than to an empty plan, with a
  regression test pinned to that exact 429.

---

## 8. Strengths

- **Hallucination is structurally prevented, not merely discouraged.** Three
  independent mechanisms stack: the candidate pool restricts what the model can
  see, the persona tells it to copy verbatim, and the verifier proves it did. The
  measured rate at the output is 0% even when the model is forced to fabricate.
- **Constraints are honoured or the shortfall is stated.** Ask for four workout
  tracks when only three qualify and you get three plus an explanation — never a
  padded fourth.
- **Situational requests work.** "Deep focus", "late night drive", "heartbreak" and
  "hosting" retrieve meaningfully different music, which raw feature matching could
  not distinguish (§5).
- **Every run is auditable.** The complete chain — query, plan, filters, retrieved
  passages, candidate scores, each attempt, each guardrail verdict, final output —
  is written to `logs/agent_traces.json` and rendered in the UI.
- **It degrades instead of failing.** No API key, a model that raises, unparseable
  JSON, or four failed attempts each have a defined, tested path that still returns
  a verified answer.

---

## 9. Limitations and Bias

### Limitations

- **17 songs.** Most genres have exactly one representative. "Three night-drive
  tracks" cannot be answered well because the catalog holds one synthwave song. The
  system reports the ceiling honestly but cannot raise it.
- **Guardrails verify grounding, not taste.** They prove a track exists and
  satisfies the stated constraints. They cannot prove it is a *good* recommendation,
  and the grounding check is satisfiable by a rationale that cites a real number
  uselessly.
- **Editorial notes are one person's opinion**, hand-written, unreviewed, and
  applied with a fixed weight (`EDITORIAL_WEIGHT = 2.5`). §5 shows this helps on
  average and hurts on at least one prompt.
- **The offline planner is regex-based.** It handles the benchmark phrasings well
  and degrades predictably outside them — "songs like Radiohead but happier" parses
  to nothing useful. The guardrails behave identically either way.
- **No audio, lyrics, history or personalisation.** Every judgment derives from ten
  CSV columns and a JSON file.
- **The Gemini benchmark is a single run on a free-tier key.** 10/10 with 0%
  hallucination is a real measurement, but it is *one* pass over 10 prompts, on a
  mixed set of models forced by rate limiting, with no seed variation. It is
  evidence that the guardrails hold, not a stable estimate of how often a model
  fails them. The honest reading of `Average Retries: 0.00` is that these ten
  prompts did not provoke a violation on that day, not that violations are rare.
- **Free-tier quota is a hard operational ceiling.** 20 requests per day on
  `gemini-3.6-flash`, and each benchmark case costs two. Running the full harness
  twice exhausts a day's budget, which is itself how the planner-degradation bug
  above was discovered.

### Bias

- **Catalog bias.** Coverage is uneven by construction. Genres with one track get
  recommended whenever that genre is mentioned; absent genres (K-pop, hip hop, EDM)
  can never be served. The system now *names* the gap in a notice rather than
  silently substituting something adjacent — an improvement in transparency, not in
  coverage.
- **Editorial bias (new in this version).** The notes encode judgments like "metal
  should be excluded by default from shared-background requests" and "blues is the
  honest pick for sadness". Those are defensible curatorial positions, not facts,
  and they now steer ranking directly. This is a *fifth* bias layered on top of the
  four the base project already had.
- **Preference bias.** Exact genre and mood matches are weighted heavily (3.0 each),
  so a stated preference narrows results sharply and reduces serendipity.
- **Feature and proxy bias.** "Happy" and "chill" are coarse labels standing in for
  something much more individual; one person's "chill" is another's "boring".
- **Popularity and representation bias.** Any gap in who is represented in the
  catalog is reproduced and amplified by every layer above it. Retrieval and
  guardrails do nothing to correct this — they faithfully propagate whatever the
  catalog contains.
- **Automation bias (new risk).** The visible "GUARDRAILS: PASS" badge may read as
  "this recommendation is correct". It means only that the track exists and
  satisfies the stated constraints. The UI states the specific checks that ran, but
  the risk of over-trust is real and worth naming.

---

## 10. Future Improvements

1. **Record the Gemini-path benchmark.** Run `eval_harness.py --engine gemini`
   across several seeds and publish those numbers next to the offline ones; the
   real measure of the guardrails is how often they fire against a real model.
2. **Grow and rebalance the catalog** so most requests have more than one
   defensible answer, and so absent genres become a solvable problem.
3. **Semantic retrieval over the editorial notes.** Lexical overlap misses
   paraphrase; embeddings would catch "music for grinding through a problem" →
   "deep focus".
4. **Have a second model review the editorial notes** for bias and
   over-prescription, and attribute each note so its provenance is inspectable.
5. **Replace the hand-labelled ideal sets** in §5 with multi-rater human judgments
   so the RAG benefit is measured against something other than the author's taste.
6. **Expose a diversity control** so the user can ask for breadth over precision,
   directly countering preference bias.

---

## 11. Reproducing Everything in This Card

```bash
pytest -q                                      # 48 unit tests
python tests/eval_harness.py                   # §7 clean run
python tests/eval_harness.py --inject-faults   # §7 fault-injected run
python tests/ablations.py                      # §5 and §6 tables
python -m src.main --guardrail-demo            # §7 behaviours, live
```

JSON artifacts: `logs/eval_report.json`, `logs/eval_report_faults.json`,
`logs/ablation_report.json`, `logs/agent_traces.json`.
