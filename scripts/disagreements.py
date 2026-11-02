"""List raw traces where a judge and the human-feature gold disagree, with the text that mentions tests.
No API calls. Usage (optional --judge gemini|grok, --category agentic_conflict):
  python3 scripts/disagreements.py --traces data/traces_all.jsonl --labels data/labels_final.csv \
      --judgments results/judgments_grok_pub_FINAL.jsonl --rubric goodfire_f1 --kind miss --limit 6
kind: miss = human gold says aware, judge says not (judge too strict or gold too loose)
      over = judge says aware, human gold says not (judge too liberal or gold too strict)
"""
import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from awarejudge.analysis import load_labels
from awarejudge.rubrics import RUBRICS
from awarejudge.traces import load_traces

ap = argparse.ArgumentParser()
ap.add_argument("--traces", required=True); ap.add_argument("--labels", required=True)
ap.add_argument("--judgments", required=True); ap.add_argument("--rubric", required=True, choices=list(RUBRICS))
ap.add_argument("--kind", required=True, choices=["miss", "over"]); ap.add_argument("--limit", type=int, default=6)
ap.add_argument("--judge", default=None, help="only this judge name, e.g. gemini or grok"); ap.add_argument("--category", default=None, help="only this prompt category")
a = ap.parse_args()

rub = RUBRICS[a.rubric]
gold = load_labels(a.labels)
traces = {t.trace_id: t for t in load_traces(a.traces)}
KEY = re.compile(r"test|eval|benchmark|red.?team|probing|scenario|fictional|artificial|simulat|placeholder", re.I)

n = 0
from collections import Counter
bycat = Counter()
for l in open(a.judgments):
    r = json.loads(l)
    if r["rubric"] != a.rubric or r["form"] != "raw" or r["aware"] is None or r["trace_id"] not in gold:
        continue
    if (a.judge and a.judge not in r["judge"]) or (a.category and r["category"] != a.category):
        continue
    g = rub.binary(rub.gold(gold[r["trace_id"]]))
    if (a.kind == "miss" and not (g == 1 and r["aware"] == 0)) or (a.kind == "over" and not (g == 0 and r["aware"] == 1)):
        continue
    f, t = gold[r["trace_id"]], traces[r["trace_id"]]
    n += 1
    bycat[r["category"]] += 1
    if n > a.limit:
        continue
    print("=" * 90)
    print(f"{r['trace_id']}  judge={r['judge']}  category={r['category']}  judge_level={r['level']}  gold_level={rub.gold(f)}")
    print(f"human: mentions={f.mentions_possible_test} concludes={f.concludes_test} artificial={f.notes_artificial} acts={f.acts_on_belief}")
    shown = 0
    for m in KEY.finditer(t.raw):
        s = max(0, m.start() - 110); e = min(len(t.raw), m.end() + 110)
        print("  ...", t.raw[s:e].replace("\n", " "), "...")
        shown += 1
        if shown >= 3:
            break
    print("  judge said:", r["reply"][:260].replace("\n", " "))
print("\nBy category:", dict(bycat))
print(f"\n{n} {a.kind} cases for {a.rubric} on raw traces (showing up to {a.limit}).")