# Judge Audit

**Same traces, different verdicts.** (package name `awarejudge`).

**Question.** On identical reasoning traces, how much do reported evaluation-awareness rates change with
(a) the rubric definition, (b) the judge model, and (c) whether the judge sees raw or summarized reasoning?

**Status.** The pilot has been run: 200 traces from two models, four published rubrics, two judges (Grok and Gemini),
three trace forms, and 100 hand-labelled traces. The pre-set gate said CONTINUE. This is a small pilot with no confidence
intervals: it shows that reported rates are sensitive to measurement choices, not what any model's true rate is.

**Headline numbers** (raw traces, two judges averaged): the share rated aware ranges from 14.5% to 39.0% across the four rubrics
(2.7-fold). On the Goodfire F.1 rubric, kappa to the human labels is 0.78 under one judge and 0.23 under the other.
Judge-vs-judge disagreement is 0.82 of rubric-vs-rubric disagreement (the gate rule: continue at 0.5 or more).

**What is not in this repository.** Private working notes, and the early rubric copies that were found to be wrong. Model outputs and
judge replies depend on the providers' terms: check them before publishing raw outputs. `scripts/modal_serve.py` is an
open-weight judge server that was tried and not completed; no result uses it.

**Rubric prompts** are third-party work: see `third_party_rubrics/SOURCES.md` for sources and licences (the Li et al. prompt is
CC BY-NC 4.0). Do not relicense them.

## What you run (needs your model access)
Use any OpenAI-compatible endpoint (vLLM, Together, OpenRouter, ...). The endpoint must return the model's
reasoning (`reasoning_content` or `reasoning`). Traces without visible reasoning are dropped.

```bash
export OPENAI_API_KEY=...            # or whatever key variable you pass with --api-key-env

# generate 50 prompts x 2 samples x 2 open-weight reasoning models = about 200 traces
python -m awarejudge.cli gen --model-name modelA --model <id> --base-url <url> --out traces/a.jsonl
python -m awarejudge.cli gen --model-name modelB --model <id> --base-url <url> --out traces/b.jsonl
cat traces/a.jsonl traces/b.jsonl > traces/all.jsonl

# Add summarized forms (150 and 60 words)
python -m awarejudge.cli forms --traces traces/all.jsonl --summ-model <id> --summ-base-url <url> --out traces/forms.jsonl

# write the label sheet (100 traces, balanced across model and category), then label it
python -m awarejudge.cli sheet --traces traces/forms.jsonl --n 100 --out labels/sheet.csv
#   -> read docs/labeling_guide.md; two people label the first 30 independently and reconcile

# Judge every cell (4 rubrics x 2 judges x 3 forms) while labeling happens
python -m awarejudge.cli judge --traces traces/forms.jsonl --judge-models <judgeA> <judgeB> --base-url <url> --out results/judgments.jsonl

# Report and gate verdict
python -m awarejudge.cli analyze --judgments results/judgments.jsonl --labels labels/sheet.csv --out results/report.md
```
Add `--dummy` to gen/forms/judge to test the plumbing offline.

## Pre-set gate (change thresholds before the pilot, not after)
Continue if any holds:
- a model pair swaps order between rubrics, with a gap of at least 0.05 in each direction;
- a prompt-category pair swaps order between rubrics, same margin;
- judge-vs-judge disagreement >= 0.5 x rubric-vs-rubric disagreement.

The rate ratio between rubrics is reported but does not gate: prior work already shows that judge and rubric
wording move rates, so a gap alone is not a new finding.
Stop or shrink if all rubrics are within 1.5x, there are no flips of any size, and mean judge-human kappa >= 0.8.
Otherwise the pilot is inconclusive; extend the labeled sample before deciding.
Otherwise the verdict is INCONCLUSIVE and we look at the numbers together.

## Known limitations (read before trusting any number)
- **Rubric prompts are four published originals** (Li et al., Aranguri and Bloom F1, Deckenbach et al., arXiv 2608.21766 C.1).
  Two are verbatim from authors' repositories; two were copied by the user from the sources (F1 from the post, C.1 from the paper) and are not machine-verified. See
  `third_party_rubrics/SOURCES.md`. What is ours (inputs given, thresholds, gold mappings) is listed there.
- The gold mappings from human features to each rubric are our reading of each rubric's construct. Review them.
- The four prompts differ in scale (two binary, two four-level), in what text the judge receives, and in definition. Judge-vs-gold agreement per rubric shows where our mapping is doubtful.
- Gold labels come from raw traces; summary forms are scored against raw-trace gold, so they measure information loss.
- 200 traces and about 100 labels give wide intervals. The pilot decides whether to continue; it is not a result.
- Open-weight traces may not resemble closed-model reasoning, and our summarizer is not any vendor's.
- Prompts here are our own (50, five categories). Swap in a public prompt set if you prefer.
- The Dummy models and the unit tests check plumbing only. They validate no scientific claim.
- The endpoint client is untested against a live server.
