#!/usr/bin/env python3
"""Re-read the saved judge replies with the current parsers. No API calls, no cost.

Use after a parser fix. Writes a corrected copy and keeps the original untouched.
    python3 scripts/reparse.py results/judge_test5_pub.jsonl results/judge_test5_pub_fixed.jsonl
"""
import collections
import json
import sys

sys.path.insert(0, ".")
from awarejudge.rubrics import RUBRICS  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
changed = collections.Counter()
n = 0
with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        if not r.get("blocked") and r.get("reply") is not None:
            rub = RUBRICS[r["rubric"]]
            level = rub.parse(r["reply"])
            aware = None if level is None else rub.binary(level)
            if level != r["level"]:
                changed[(r["judge"], r["rubric"], f"{r['level']}->{level}")] += 1
            r["level"], r["aware"] = level, aware
        out.write(json.dumps(r) + "\n")
print(f"{n} rows read, {sum(changed.values())} changed -> {dst}")
for k, v in sorted(changed.items(), key=str):
    print("  changed:", k, v)