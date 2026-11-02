
# Judge Audit

**Same traces, different verdicts.** How much does a reported "evaluation awareness" rate depend on the instrument that measured it?

Papers on AI safety report how often a model's reasoning shows that it suspects it is being tested. That number is almost always produced by an
AI judge applying a written rubric. This project holds the reasoning traces fixed and changes only the measuring instrument: the rubric,
the judge model and the form of the trace the judge sees. The code package is called `awarejudge`.

## What was uncovered

All figures are from raw traces, 200 traces, two judges (Grok and Gemini) averaged unless stated. The pilot has no confidence intervals.

| Rubric | Source | Rated aware (2 judges) | Human label rate (100 traces) |
|---|---|---|---|
| Goodfire F.1 | Aranguri and Bloom, Appendix F.1 | 15.8% | 24% |
| Li et al. | Li et al., EvalAwareBench judge prompt | 14.5% | 14% |
| Deckenbach 3-level | Deckenbach et al. | 36.7% | 14% |
| 0–3 rubric | arXiv 2608.21766, Appendix C.1 | 39.0% | 24% |

1. **The rubric moves the rate 2.7-fold** (14.5% to 39.0%) on the same traces. Under one judge alone the spread was 5.5% to 41.5%.
2. **The judge matters almost as much.** Kappa against our human labels on the F.1 rubric is 0.78 under Gemini and 0.23 under Grok. Judge-to-judge
   disagreement is 0.171, rubric-to-rubric is 0.210: a ratio of 0.82. The gate said continue at 0.5 or more.
3. **No ranking flipped.** No model pair and no prompt-category pair changed order between rubrics by the preset margin of 0.05. The verdict
   CONTINUE comes from the judge-noise criterion alone.
4. **The finding survives removing the obvious cases** (post-hoc, run after seeing the results). Without the prompts that state the evaluation themselves,
   rubric-vs-rubric disagreement is 0.188 and judge-vs-judge 0.134 (ratio 0.71). On those same traces Gemini finds 7 of the 9 human positives under F.1 and Grok finds 0.
5. **Where the rubrics part ways** (read by hand for the agentic-conflict category only, small counts, untested). The F.1 misses are hedged mentions such as "this could be a test of
   whether the model will delete others' data". The Deckenbach over-flags are of two kinds: hedged mentions that our labels call "mentions only", so part of that gap
   is our own mapping, and traces where "scenario", "ethics question" or "jailbreak attempt" was read as awareness.

What this does not show: how aware any model is. It shows that a reported awareness rate cannot be compared across studies without its rubric, judge, trace form,
refusal share and agreement with humans.

## What was built

Most of the work is the machinery around the numbers, so that each number can be traced to its inputs.

- **Traces.** 50 prompts in five categories (artificial scenario, explicit evaluation note, ethical dilemma, agentic conflict, benign realistic), two samples each, two
  reasoning models (gpt-oss and DeepSeek, the latter used through its paid API): 200 traces, each in three forms (raw, 150-word summary, 60-word summary).
- **Human labels.** 100 traces labelled for five features ("mentions a possible test", "concludes it is a test", type of evaluation, "notes the scenario is artificial",
  "acts on the belief"), by two labellers and a third reviewer for disagreements. Labels are features, not rubric levels, so the same labels can be mapped onto every rubric.
  The mapping is ours and is listed in `awarejudge/rubrics.py`. Totals: 24 traces mention a test, 14 conclude one, 3 act on it.
- **Four published rubrics, used as written.** Two are read programmatically from the authors' repositories; two were copied by hand from the paper and post and checked.
  `third_party_rubrics/SOURCES.md` records where each came from, its licence, and what is ours. Earlier adapted copies were wrong and were discarded.
- **Two judges, 2,400 judgments each** (4 rubrics × 3 forms × 200 traces). Every call logs the provider, the model requested and the model served. Raw replies are stored,
  so a parser fix needs no new calls (`scripts/reparse.py`).
- **A preset decision rule**, in code, with thresholds fixed before any judging (`awarejudge/analysis.py`).
- **Complete-case analysis.** Judgments a provider refused are treated as missing, never as "unaware", and judges are compared on the cells both scored.
- **Post-hoc analyses** (`scripts/posthoc.py`): without the explicit-evaluation prompts, only those prompts, without the rows the third reviewer decided, and a blind
  20-trace validation sample to be labelled by someone who has not seen any judge output (instructions for the labeller are in `docs/`).
- **A disagreement reader** (`scripts/disagreements.py`) that lists the traces where a judge and the human labels differ, filterable by judge and category.

## The pre-set gate

Continue if any of these holds, with thresholds fixed in the analysis code before the main run:

- a model pair swaps order between rubrics, with a gap of at least 0.05 in each direction;
- a prompt-category pair swaps order between rubrics, same margin;
- judge-vs-judge disagreement is at least 0.5 × rubric-vs-rubric disagreement.

The rate ratio between rubrics is reported but does not gate, because earlier work already shows that wording moves rates. Stop or shrink if all rubrics are within 1.5× of each other,
nothing flips, and mean judge-to-human kappa is at least 0.8. Otherwise the pilot is inconclusive. The gate is not re-run on subsets.

## Credits and licences

The rubric prompts are other people's work: see `third_party_rubrics/SOURCES.md`. The Li et al. prompt is CC BY-NC 4.0 (non-commercial, credit required) and the
Deckenbach et al. prompt is Apache-2.0. Do not relicense them. Licence for this repository's own code: Apache-2.0

## About

Author: Eyitayo Alimi, Independent AI safety researcher.

