#!/usr/bin/env python3
"""Build the final label file from two labelers plus the reconciled rows. Standard library only.

  python3 scripts/finalize_labels.py --a data/labels_A.csv --b data/labels_B.csv \
      --reconcile data/reconcile.csv --out data/labels_final.csv

Rows where A and B agree on every feature keep that value. Rows in reconcile.csv take the filled-in final_ columns.
Only the features where A and B differ need a final_ value; the others may stay blank.
Stops with a clear message if any reconcile row is incomplete or invalid, so nothing is silently guessed.
"""
import argparse, csv, sys

BOOL = ["mentions_possible_test", "concludes_test", "notes_artificial", "acts_on_belief"]
FEATS = ["mentions_possible_test", "concludes_test", "eval_type", "notes_artificial", "acts_on_belief"]
ET = {"none", "alignment_safety", "capability", "other_or_unspecified"}


def read(path):
    return {r["row_id"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig", newline=""))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--reconcile", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, B = read(a.a), read(a.b)
    rec = read(a.reconcile)
    errs, rows = [], []
    for rid in sorted(set(A) & set(B), key=lambda r: int(r[1:])):
        base = A[rid]
        vals = {f: (base[f] or "").strip().lower() for f in FEATS}
        same = all((A[rid][f] or "").strip().lower() == (B[rid][f] or "").strip().lower() for f in FEATS)
        if not same:
            r = rec.get(rid)
            if r is None:
                errs.append(f"{rid}: labelers differ but the row is not in reconcile.csv"); continue
            fin = {}
            for f in FEATS:
                v = (r.get("final_" + f) or "").strip().lower()
                if not v:  # blank is fine only where both labelers already agree
                    av, bv = (A[rid][f] or "").strip().lower(), (B[rid][f] or "").strip().lower()
                    if av == bv:
                        v = av
                    else:
                        errs.append(f"{rid}: fill in final_{f} (A said {av}, B said {bv})")
                fin[f] = v
            if any(e.startswith(rid + ":") for e in errs):
                continue
            for f in BOOL:
                if fin[f] not in ("true", "false"):
                    errs.append(f"{rid}: final_{f} must be true or false, got {fin[f]!r}")
            if fin["eval_type"] not in ET:
                errs.append(f"{rid}: final_eval_type {fin['eval_type']!r} invalid")
            vals = fin
        if vals["concludes_test"] == "true":
            vals["mentions_possible_test"] = "true"
        if vals["mentions_possible_test"] == "false" and vals["concludes_test"] == "false" and vals["eval_type"] != "none":
            errs.append(f"{rid}: no test mention but eval_type is {vals['eval_type']}")
        if vals["acts_on_belief"] == "true" and vals["mentions_possible_test"] == "false":
            errs.append(f"{rid}: acts_on_belief needs a test mention")
        rows.append({"row_id": rid, "trace_id": base["trace_id"], "model": base["model"], "category": base["category"],
                     "reasoning": base["reasoning"], **vals, "notes": (rec.get(rid, {}).get("why") or base.get("notes", "")).strip()})
    if errs:
        print(f"{len(errs)} problems, nothing written:")
        for e in errs[:30]:
            print("  -", e)
        sys.exit(1)
    cols = ["row_id", "trace_id", "model", "category", "reasoning"] + FEATS + ["notes"]
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    n_rec = sum(1 for r in rows if r["row_id"] in rec)
    print(f"{len(rows)} final labels written to {a.out} ({n_rec} reconciled, {len(rows) - n_rec} agreed)")


if __name__ == "__main__":
    main()