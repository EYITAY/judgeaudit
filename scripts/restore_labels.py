#!/usr/bin/env python3
"""Turn a labeler's returned sheet back into the kit's label CSV.

The labeler works on the blind workbook (row_id, reasoning, five label columns, notes) and never sees the model
or category. In Excel: open the Labels sheet, File > Save As > CSV UTF-8, then run:

  python3 scripts/restore_labels.py --returned labeler1.csv --key label_key_PRIVATE.csv --out data/labels_A.csv

Standard library only. Prints problems instead of hiding them. Never prints reasoning text.
"""
import argparse, csv, sys

FEATS = ["mentions_possible_test", "concludes_test", "eval_type", "notes_artificial", "acts_on_belief"]
EVAL_TYPES = {"none", "alignment_safety", "capability", "other_or_unspecified"}
TRUE, FALSE = {"true", "1", "yes", "y", "t"}, {"false", "0", "no", "n", "f"}


def norm_bool(v):
    s = (v or "").strip().lower()
    return "true" if s in TRUE else "false" if s in FALSE else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--returned", required=True); ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = {r["row_id"]: r for r in csv.DictReader(open(a.key, encoding="utf-8-sig", newline=""))}
    out, problems, blank = [], [], 0
    for r in csv.DictReader(open(a.returned, encoding="utf-8-sig", newline="")):
        rid = (r.get("row_id") or "").strip()
        if rid not in key:
            problems.append(f"{rid or '(no id)'}: row_id not in key"); continue
        vals = {k: (r.get(k) or "").strip() for k in FEATS}
        if not any(vals.values()):
            blank += 1; continue
        row = {"row_id": rid, "trace_id": key[rid]["trace_id"], "model": key[rid]["model"], "category": key[rid]["category"],
               "reasoning": r.get("reasoning", ""), "notes": (r.get("notes") or "").strip()}
        ok = True
        for k in ("mentions_possible_test", "concludes_test", "notes_artificial", "acts_on_belief"):
            b = norm_bool(vals[k])
            if b is None:
                problems.append(f"{rid}: {k} must be true/false, got {vals[k]!r}"); ok = False
            row[k] = b
        et = vals["eval_type"].lower()
        if et not in EVAL_TYPES:
            problems.append(f"{rid}: eval_type {vals['eval_type']!r} not one of {sorted(EVAL_TYPES)}"); ok = False
        row["eval_type"] = et
        if ok:
            if row["concludes_test"] == "true" and row["mentions_possible_test"] != "true":
                problems.append(f"{rid}: concludes_test true needs mentions_possible_test true (kit will normalize)")
            if row["mentions_possible_test"] == "false" and row["concludes_test"] == "false" and et != "none":
                problems.append(f"{rid}: no test mention but eval_type is {et} (must be none)")
            if row["acts_on_belief"] == "true" and row["mentions_possible_test"] == "false" and row["concludes_test"] == "false":
                problems.append(f"{rid}: acts_on_belief needs a test mention")
            out.append(row)
    cols = ["row_id", "trace_id", "model", "category", "reasoning"] + FEATS + ["notes"]
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(out)
    print(f"{len(out)} labeled rows written to {a.out}; {blank} blank rows skipped; {len(problems)} problems")
    for p in problems[:40]:
        print("  -", p)
    sys.exit(1 if problems and not out else 0)


if __name__ == "__main__":
    main()