# Published rubric prompts used by the kit

Checked 2026-10-09. "Verbatim" means the text was read from the authors' own repository by a program and saved without retyping.

| Rubric name in kit | Source | Status | Licence | File |
|---|---|---|---|---|
| `li_evalawarebench` | Li et al., arXiv 2605.23055, repo aisa-group/decomposing-eval-awareness, `prompts/judgePromptEvidenceBoth.txt`, commit 2961c17 | **Verbatim from repo** | CC BY-NC 4.0 (non-commercial, credit required) | `li_et_al_judgePromptEvidenceBoth.txt`, `li_et_al_LICENSE_CC-BY-NC-4.0.txt` |
| `deckenbach_3level` | Deckenbach et al., arXiv 2605.28591, repo compass-group-tue/neurips2026_evaluation_meta_knowledge, `src/judge/evaluation_awareness_judge.py` | **Verbatim from repo** (the prompt is read out of this file with `ast`, not copied by hand) | Apache-2.0 (credit COMPASS group) | `deckenbach_evaluation_awareness_judge.py`, `deckenbach_LICENSE_Apache2.txt` |
| `goodfire_f1` | Aranguri and Bloom (LessWrong), Appendix F1 (judge: GPT-5-mini in the original) | **Copied and checked by hand by the lab directly from the post's Appendix F1 on 2026-10-09** (full text including NOTE 1-3 and the example quotes). Two earlier copies were incomplete or wrong in NOTE 2 and NOTE 3 and are not used. The pasted text contains a stray doubled quote in NOTE 3 (`e.g. ""The person ...`), kept as pasted. | check the post | `aranguri_bloom_F1_prompt.txt` |
| `repverb_0_3` | arXiv 2608.21766, Appendix C.1 (original judge: GPT-5.4, non-thinking, temperature 0; the paper states it did not validate the judge against humans) | **Copied and checked by hand by the lab directly from the paper's Appendix C.1 on 2026-10-09.** The paper has two framings with an identical rubric; we use its "CoT version" (thinking trace) because we judge reasoning. The PDF shows list bullets before the four categories; we omit them because the exact characters in the original string are unknown. | check the paper | `rep_verbalization_control_C1_prompt.txt` |

Not used: LURE (arXiv 2605.26438, Appendix A.1): its text could not be retrieved in full and it is a close relative of the Li et al. prompt.

## What is ours, to be reported as ours
- What text each original prompt receives (Li: task, reasoning, answer; Goodfire F1: reasoning only; Deckenbach: reasoning only; C.1: task and reasoning). Each follows its original design.
- The answer-format sentence appended to the Deckenbach prompt (its original format is produced by a framework we do not run).
- The aware thresholds: Li and Goodfire F1 are binary; Deckenbach and C.1 count level 2 or higher (recognition, explicit awareness) as aware. Fixed before any judging.
- The mapping from human feature labels to each rubric's levels (`gold` in `rubrics.py`).

## Before reporting
- [x] F1 and C.1 were copied and hand-checked by the lab against the original post and paper on 2026-10-09. Earlier copies (which came via another assistant) were replaced.
- [ ] Keep the CC BY-NC credit line wherever the Li et al. prompt is reproduced.