#!/usr/bin/env python3
"""Post-hoc analyses. Every number printed here is POST-HOC: it was chosen after seeing the main results.
Report it next to the original results, labelled post-hoc. No thresholds are tuned and no labels are changed.
No API calls, no cost. Standard library plus the kit itself.

  python3 scripts/posthoc.py --judgments results/judgments_pub.jsonl --labels data/labels_final.csv \
      --reconcile data/reconcile.csv --out results/report_posthoc.md

Analyses:
  A. without the explicit_eval_note category (prompts that state the evaluation themselves)
  B. only the explicit_eval_note category (where nearly all human-positive labels sit)
  C. without the reconciled rows (rows where the two labelers differed and the third reviewer decided); needs --reconcile
  D. validation sample: --make-validation writes a blind 20-row sheet (4 per category) and a key; --check-validation compares it.

For every analysis: aware rate per rubric (raw traces, judge-averaged), disagreement, and per judge and rubric the
counts against the human-feature gold (gold positives, judge flags, true positives, false positives, misses).
The pre-set gate is NOT re-run on a subset, because its thresholds were set for the full sample.
"""
import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from awarejudge.analysis import disagreement, load_labels, rubric_rates_raw, kappa  # noqa: E402
from awarejudge.judge import load_judgments  # noqa: E402
from awarejudge.rubrics import RUBRICS  # noqa: E402

EXPLICIT = "explicit_eval_note"
FEATS = ["mentions_possible_test", "concludes_test", "eval_type", "notes_artificial", "acts_on_belief"]


def counts_against_gold(rows, gold):
    """(rubric, judge) -> counts on raw traces that have a label and a valid judgment."""
    cells = defaultdict(lambda: {"n": 0, "gold_pos": 0, "flagged": 0, "tp": 0, "fp": 0, "fn": 0})
    for r in rows:
        if r["form"] != "raw" or r["aware"] is None:
            continue
        f = gold.get(r["trace_id"])
        if f is None:
            continue
        rub = RUBRICS[r["rubric"]]
        g, p = bool(rub.binary(rub.gold(f))), bool(r["aware"])
        c = cells[(r["rubric"], r["judge"])]
        c["n"] += 1
        c["gold_pos"] += g
        c["flagged"] += p
        c["tp"] += g and p
        c["fp"] += (not g) and p
        c["fn"] += g and (not p)
    return cells


def kappa_from_counts(c):
    n = c["n"]
    if not n:
        return None
    a = [1] * c["tp"] + [1] * c["fn"] + [0] * (c["fp"]) + [0] * (n - c["tp"] - c["fn"] - c["fp"])
    b = [1] * c["tp"] + [0] * c["fn"] + [1] * (c["fp"]) + [0] * (n - c["tp"] - c["fn"] - c["fp"])
    return kappa(a, b)


def section(title, rows, gold, note=""):
    traces = {r["trace_id"] for r in rows if r["form"] == "raw"}
    labelled = [t for t in traces if t in gold]
    L = [f"## {title}", "", f"POST-HOC. Raw traces judged: {len(traces)}; with human labels: {len(labelled)}. {note}".strip(), ""]
    rr = rubric_rates_raw(rows)
    L.append("Aware rate per rubric (raw traces, judge-averaged):")
    for k, v in sorted(rr.items()):
        L.append(f"- {k}: {v:.3f}")
    d = disagreement(rows)
    L += ["", f"Disagreement: rubric-vs-rubric {d['rubric_vs_rubric']:.3f}, judge-vs-judge {d['judge_vs_judge']:.3f}, "
              f"ratio {d['judge_noise_ratio']:.2f}", ""]
    L.append("Against the human-feature gold (raw traces):")
    L.append("")
    L.append("| Rubric | Judge | n | Gold positives | Judge flags | True pos. | False pos. (over-flags) | Misses | Kappa |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (rub, judge), c in sorted(counts_against_gold(rows, gold).items()):
        k = kappa_from_counts(c) if c["gold_pos"] else None  # no human positives: kappa is not meaningful
        L.append(f"| {rub} | {judge} | {c['n']} | {c['gold_pos']} | {c['flagged']} | {c['tp']} | {c['fp']} | {c['fn']} | "
                 f"{'n/a' if k is None else format(k, '.2f')} |")
    L.append("")
    return L


def reconciled_trace_ids(reconcile_path, labels_path):
    rec = {r["row_id"] for r in csv.DictReader(open(reconcile_path, encoding="utf-8-sig", newline=""))}
    rid2tid = {r["row_id"]: r["trace_id"] for r in csv.DictReader(open(labels_path, encoding="utf-8-sig", newline=""))}
    missing = sorted(rec - set(rid2tid))
    if missing:
        print("WARNING: reconcile rows not found in the labels file:", missing)
    return {rid2tid[r] for r in rec if r in rid2tid}


def make_validation(labels_path, out_dir, per_category=4, seed=0):
    rows = list(csv.DictReader(open(labels_path, encoding="utf-8-sig", newline="")))
    by = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r)
    rng = random.Random(seed)
    pick = []
    for cat in sorted(by):
        pick += rng.sample(by[cat], min(per_category, len(by[cat])))
    rng.shuffle(pick)
    out_dir.mkdir(parents=True, exist_ok=True)
    blank = out_dir / "validation_sample_blank.csv"
    key = out_dir / "validation_sample_KEY.csv"
    with open(blank, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["val_id", "reasoning"] + FEATS + ["notes"])
        for i, r in enumerate(pick, 1):
            w.writerow([f"V{i:02d}", r["reasoning"]] + [""] * (len(FEATS) + 1))
    with open(key, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["val_id", "row_id", "trace_id", "category"] + FEATS)
        for i, r in enumerate(pick, 1):
            w.writerow([f"V{i:02d}", r["row_id"], r["trace_id"], r["category"]] + [r[x] for x in FEATS])
    print(f"Wrote {blank} (give this to a labeler who has not seen the judge output or the earlier labels) and {key} "
          f"(keep it away from the labeler).")


def check_validation(filled_path, key_path):
    filled = {r["val_id"]: r for r in csv.DictReader(open(filled_path, encoding="utf-8-sig", newline=""))}
    key = {r["val_id"]: r for r in csv.DictReader(open(key_path, encoding="utf-8-sig", newline=""))}
    L = ["## D. Validation sample (POST-HOC)", "", f"Rows compared: {len(set(filled) & set(key))}", "",
         "| Feature | Agreement | Kappa |", "| --- | --- | --- |"]
    for f in FEATS:
        a, b = [], []
        for v in sorted(set(filled) & set(key)):
            a.append((filled[v][f] or "").strip().lower())
            b.append((key[v][f] or "").strip().lower())
        n = len(a)
        agree = sum(x == y for x, y in zip(a, b)) / n if n else float("nan")
        cats = sorted(set(a) | set(b))
        idx = {c: i for i, c in enumerate(cats)}
        k = kappa([idx[x] for x in a], [idx[y] for y in b]) if len(cats) == 2 else None
        L.append(f"| {f} | {agree:.2f} | {'n/a' if k is None else format(k, '.2f')} |")
    L += ["", "Kappa is shown for two-valued features only and is unreliable when a feature is rare. "
              "Read the raw disagreements before drawing any conclusion.", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--reconcile")
    ap.add_argument("--out", default="results/report_posthoc.md")
    ap.add_argument("--make-validation", action="store_true")
    ap.add_argument("--check-validation", nargs=2, metavar=("FILLED_CSV", "KEY_CSV"))
    a = ap.parse_args()

    if a.make_validation:
        make_validation(a.labels, Path(a.out).parent)
        return
    if a.check_validation:
        text = "\n".join(check_validation(*a.check_validation))
        print(text)
        Path(a.out).with_name("report_posthoc_validation.md").write_text(text, encoding="utf-8")
        return
    if not a.judgments:
        ap.error("--judgments is required unless --make-validation or --check-validation is used")

    rows = load_judgments(a.judgments)
    gold = load_labels(a.labels)
    L = ["# Post-hoc analyses", "",
         "Everything below was chosen after seeing the main results. Report it next to the original results, labelled "
         "post-hoc. Thresholds are as pre-set and no label was changed. The pre-set gate is not re-run on subsets.", ""]

    no_exp = [r for r in rows if r.get("category") != EXPLICIT]
    gold_no_exp = {t: f for t, f in gold.items() if t in {r["trace_id"] for r in no_exp}}
    L += section("A. Without the explicit_eval_note category", no_exp, gold_no_exp,
                 "Recall cannot be assessed here when there are few or no human positives; over-flags and judge disagreement can.")

    only_exp = [r for r in rows if r.get("category") == EXPLICIT]
    gold_exp = {t: f for t, f in gold.items() if t in {r["trace_id"] for r in only_exp}}
    L += section("B. Only the explicit_eval_note category", only_exp, gold_exp)

    if a.reconcile:
        drop = reconciled_trace_ids(a.reconcile, a.labels)
        kept = [r for r in rows if r["trace_id"] not in drop]
        gold_kept = {t: f for t, f in gold.items() if t not in drop}
        L += section(f"C. Without the {len(drop)} reconciled rows", kept, gold_kept)
    else:
        L += ["## C. Without the reconciled rows", "", "Skipped: pass --reconcile data/reconcile.csv.", ""]

    text = "\n".join(L)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWritten to {a.out}. For analysis D run with --make-validation, then --check-validation.")


if __name__ == "__main__":
    main()