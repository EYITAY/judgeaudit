#!/usr/bin/env python3
"""Compare two labelers and write a reconciliation sheet. Standard library only.

  python3 scripts/compare_labelers.py --a data/labels_A.csv --b data/labels_B.csv --out-dir data/

Pre-set rule (do not change after seeing results): on the first 30 rows (R001-R030), Cohen's kappa on
mentions_possible_test must be >= 0.7, otherwise refine the guide and relabel before relying on the labels.
All-overlap numbers are reported as a supplement. Kappa is unreliable when a feature is rare, so prevalence and
raw agreement are shown next to it.
"""
import argparse, csv
from collections import Counter

BOOL = ["mentions_possible_test", "concludes_test", "notes_artificial", "acts_on_belief"]
ALL = BOOL + ["eval_type"]
KAPPA_MIN = 0.7


def load(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8", newline="")):
        row = {k: (r[k] or "").strip().lower() for k in ALL}
        if row["concludes_test"] == "true":  # same normalisation the kit applies
            row["mentions_possible_test"] = "true"
        out[r["row_id"]] = row
    return out


def kappa(x, y):
    n = len(x)
    if n == 0:
        return None
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[c] * cy[c] for c in set(cx) | set(cy)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def summarize(A, B, ids, title):
    print(f"\n{title}  (n={len(ids)})")
    print(f"{'feature':26}{'agree%':>8}{'kappa':>8}{'A true%':>9}{'B true%':>9}")
    res = {}
    for f in ALL:
        x, y = [A[i][f] for i in ids], [B[i][f] for i in ids]
        agree = sum(a == b for a, b in zip(x, y)) / len(ids)
        k = kappa(x, y)
        if f == "eval_type":
            pa = pb = float("nan")
        else:
            pa, pb = sum(v == "true" for v in x) / len(x), sum(v == "true" for v in y) / len(y)
        res[f] = k
        ks = "n/a" if k is None else f"{k:.2f}"
        pas = "-" if f == "eval_type" else f"{pa*100:.0f}%"
        pbs = "-" if f == "eval_type" else f"{pb*100:.0f}%"
        print(f"{f:26}{agree*100:7.0f}%{ks:>8}{pas:>9}{pbs:>9}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--out-dir", default="data")
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    both = sorted(set(A) & set(B), key=lambda r: int(r[1:]))
    print(f"labeler A: {len(A)} rows, labeler B: {len(B)} rows, overlap: {len(both)}")
    first30 = [r for r in both if int(r[1:]) <= 30]
    if len(first30) < 30:
        print(f"WARNING: only {len(first30)} of the first 30 rows are labeled by both")
    k30 = summarize(A, B, first30, "FIRST 30 ROWS (pre-set check)") if first30 else {}
    summarize(A, B, both, "ALL OVERLAP (supplement)")
    k = k30.get("mentions_possible_test")
    print("\nPRE-SET CHECK: kappa on mentions_possible_test, first 30 rows:",
          "n/a" if k is None else f"{k:.2f}", "->",
          "PASS (>= 0.70)" if (k is not None and k >= KAPPA_MIN) else "FAIL or undefined: refine the guide and relabel")
    dis = [r for r in both if any(A[r][f] != B[r][f] for f in ALL)]
    path = f"{a.out_dir.rstrip('/')}/reconcile.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "differs_on"] + [f"A_{x}" for x in ALL] + [f"B_{x}" for x in ALL] + [f"final_{x}" for x in ALL] + ["why"])
        for r in dis:
            diff = ";".join(x for x in ALL if A[r][x] != B[r][x])
            w.writerow([r, diff] + [A[r][x] for x in ALL] + [B[r][x] for x in ALL] + [""] * len(ALL) + [""])
    print(f"\n{len(dis)} of {len(both)} rows differ on at least one feature -> {path}")
    print("Discuss each with labeler B and fill the final_ columns. Do not look at judge output while reconciling.")


if __name__ == "__main__":
    main()