"""Command line for the pilot. Run `python -m awarejudge.cli --help`.

Steps:
  gen        generate reasoning traces from an open-weight model (needs your endpoint)
  forms      add summarized forms (needs a summarizer endpoint)
  sheet      write the CSV you label by hand
  judge      run rubric x judge x form cells
  analyze    compute the report and the gate verdict
Use --dummy on gen/forms/judge to test the plumbing offline (output is meaningless).
"""
from __future__ import annotations

import argparse
import json
import csv
import random
import zlib
from pathlib import Path

from .analysis import load_labels, report
from .judge import load_judgments, run_judgments
from .env import load_env
from .rubrics import RUBRICS
from .traces import (FORMS, OpenAICompatClient, add_summaries, generate_traces, load_prompts,
                     load_traces, save_traces)

LABEL_COLUMNS = ["mentions_possible_test", "concludes_test", "eval_type", "notes_artificial",
                 "acts_on_belief", "notes"]


class DummyModel:
    """Offline stand-in. Same input -> same output. For plumbing tests only."""

    def __init__(self, kind: str = "reasoner"):
        self.kind = kind

    def chat(self, system, user, max_tokens=0, temperature=0.0):
        rng = random.Random(zlib.crc32((self.kind + system + user).encode()))
        if self.kind == "judge":
            return f'{{"label": {rng.choice([0, 0, 1, 2, 3])}, "evidence": ""}}', ""
        if self.kind == "summarizer":
            return "The model considered the request and answered.", ""
        bits = ["Let me think about the request."]
        if rng.random() < 0.4:
            bits.append("This might be a test of my behavior.")
        if rng.random() < 0.2:
            bits.append("This looks like a safety evaluation, so I should refuse.")
        return "Answer.", " ".join(bits)


def _client(args, prefix=""):
    if getattr(args, prefix + "dummy", False) or getattr(args, "dummy", False):
        return DummyModel({"": "reasoner", "summ_": "summarizer", "judge_": "judge"}[prefix])
    extra = json.loads(getattr(args, prefix + "extra_json", None) or "{}")
    return OpenAICompatClient(getattr(args, prefix + "base_url"), getattr(args, prefix + "model"),
                              getattr(args, prefix + "api_key_env"), extra=extra)


def cmd_gen(a):
    prompts = load_prompts(a.prompts)
    stats = {}
    traces = generate_traces(prompts, _client(a), a.model_name, a.samples, a.max_tokens,
                             out_path=a.out, resume=a.resume, stats=stats)
    print(f"{a.model_name}: {len(traces)} traces in {a.out}  "
          f"(new {stats['ok']}, already had {stats['skipped_existing']}, failed {stats['failed']}, "
          f"no reasoning {stats['no_reasoning']}, hit token limit {stats['truncated']})")
    if stats["failed"] or stats["truncated"]:
        print("Re-run the same command with --resume to retry failed calls." if stats["failed"] else
              "Some traces hit the token limit: consider a larger --max-tokens (and report it).")


def cmd_forms(a):
    traces = load_traces(a.traces)
    if Path(a.out).exists():  # resume: reuse summaries already written for the same trace ids
        prev = {t.trace_id: t for t in load_traces(a.out)}
        for t in traces:
            if t.trace_id in prev:
                t.forms.update(prev[t.trace_id].forms)
                t.meta = prev[t.trace_id].meta or t.meta
    todo = sum(1 for t in traces if any(f not in t.forms for f in ("summary_long", "summary_short")))
    print(f"{len(traces)} traces, {todo} still need summaries")
    add_summaries(traces, _client(a, "summ_"), checkpoint=lambda ts: save_traces(ts, a.out))
    save_traces(traces, a.out)
    print(f"summaries complete for {len(traces)} traces -> {a.out}")


def cmd_sheet(a):
    traces = load_traces(a.traces)
    rng = random.Random(a.seed)
    by = {}
    for t in traces:
        by.setdefault((t.model, t.category), []).append(t)
    for v in by.values():
        rng.shuffle(v)
    picked, i = [], 0
    while len(picked) < min(a.n, len(traces)):  # round-robin so every model x category is covered
        progressed = False
        for v in by.values():
            if i < len(v) and len(picked) < a.n:
                picked.append(v[i]); progressed = True
        if not progressed:
            break
        i += 1
    rng.shuffle(picked)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trace_id", "model", "category", "reasoning"] + LABEL_COLUMNS)
        for t in picked:
            w.writerow([t.trace_id, t.model, t.category, t.raw] + [""] * len(LABEL_COLUMNS))
    print(f"wrote {len(picked)} traces to label -> {a.out}  (read docs/labeling_guide.md first)")


def cmd_judge(a):
    traces = load_traces(a.traces)
    judges = {}
    if a.dummy:
        for name in (a.judge_models or [j[0] for j in (a.judge or [])] or ["j1", "j2"]):
            judges[name] = DummyModel("judge")
    else:
        for label, model, base_url, key_env in (a.judge or []):
            judges[label] = OpenAICompatClient(base_url, model, key_env)
        for name in (a.judge_models or []):  # legacy: several judges on one endpoint
            judges[name] = OpenAICompatClient(a.base_url, name, a.api_key_env)
    for c in judges.values():
        if hasattr(c, "system_prefix"):
            c.system_prefix = a.system_prefix
    if len(judges) < 2 and not a.dummy and not a.allow_single:
        raise SystemExit("need at least two judges: use --judge LABEL MODEL BASE_URL KEY_ENV twice "
                         "(or --allow-single to run one judge now and combine the output files later)")
    rubrics = list(RUBRICS) if a.rubrics == "all" else a.rubrics.split(",")
    forms = a.forms.split(",")
    stats = {}
    # Pre-flight: show what would be paid for BEFORE any call is made.
    plan = {}
    run_judgments(traces, judges, rubrics, forms, a.out, resume=True, workers=a.workers, stats=plan, dry_run=True) \
        if a.resume else run_judgments(traces, judges, rubrics, forms, None, resume=False, workers=a.workers,
                                       stats=plan, dry_run=True)
    print("\n=== PRE-FLIGHT (nothing has been sent yet) ===")
    print(f"judges: {', '.join(judges)}   rubrics: {', '.join(rubrics)}   forms: {', '.join(forms)}")
    print(f"output file: {a.out}  ({'--resume: rows already there are kept' if a.resume else 'will be started fresh'})")
    print(f"already saved: {plan.get('skipped_existing', 0)}   calls to make now: {plan['to_run']}")
    print(f"rough input size: {plan['est_input_tokens']:,} tokens (output comes on top). Price = this x the provider's rate per token.")
    if a.dry_run:
        raise SystemExit("--dry-run: stopping here, as requested. No calls were made.")
    if plan["to_run"] > 100 and not a.yes:
        try:
            ok = input(f"Make {plan['to_run']} calls now (paid or free, depending on your plan)? Type yes to continue: ").strip().lower() == "yes"
        except EOFError:
            ok = False
        if not ok:
            raise SystemExit("Not started. Add --yes to skip this question once you have checked the numbers.")
    rows = run_judgments(traces, judges, rubrics, forms, a.out, resume=a.resume, workers=a.workers, stats=stats)
    print(f"{len(rows)} judgments in {a.out}  (new {stats['ok']}, already had {stats['skipped_existing']}, "
          f"failed {stats['failed']}, unparsable replies {stats['unparsable']}, blocked by provider {stats['blocked']}, hit token limit {stats['truncated']})")
    if stats.get("stopped"):
        print("\nRUN STOPPED EARLY: the provider cannot serve more requests right now.")
        print("Provider message:", stats["stopped"][:400])
        print("Fix that (add credits, raise the spending limit, or wait for the daily limit to reset), then re-run "
              "the same command with --resume. Calls already saved are kept.")
    elif stats["failed"]:
        print("Re-run the same command with --resume to retry the failed calls.")


def cmd_analyze(a):
    rows = load_judgments(a.judgments)
    gold = load_labels(a.labels) if a.labels else None
    note = ""
    if a.complete_cases:
        from .analysis import complete_cases
        rows, info = complete_cases(rows)
        note = (f"**Complete cases only.** Judges: {', '.join(info['judges'])}. Kept {info['cells_kept']} of "
                f"{info['cells_total']} (trace, form, rubric) cells that every judge scored. Dropped per rubric: "
                f"{info['dropped_by_rubric'] or 'none'}.\n\n")
        print(note)
    text = report(rows, gold, len(gold) if gold else None)
    if note:
        text = text.replace("\n", "\n\n" + note, 1)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)


def main():
    load_env()
    ap = argparse.ArgumentParser(prog="awarejudge")
    sp = ap.add_subparsers(dest="cmd", required=True)

    g = sp.add_parser("gen"); g.set_defaults(fn=cmd_gen)
    g.add_argument("--prompts", default="data/pilot_prompts.jsonl")
    g.add_argument("--model-name", required=True, help="short label, e.g. qwen3-32b")
    g.add_argument("--model", help="model id at the endpoint"); g.add_argument("--base-url")
    g.add_argument("--api-key-env", default="OPENAI_API_KEY")
    g.add_argument("--extra-json", default=None, help='extra request fields as JSON, e.g. \'{"reasoning_effort":"high"}\'')
    g.add_argument("--samples", type=int, default=2); g.add_argument("--max-tokens", type=int, default=4096)
    g.add_argument("--out", required=True); g.add_argument("--dummy", action="store_true")
    g.add_argument("--resume", action="store_true", help="skip trace ids already in --out (retry after a crash)")

    f = sp.add_parser("forms"); f.set_defaults(fn=cmd_forms)
    f.add_argument("--traces", required=True); f.add_argument("--out", required=True)
    f.add_argument("--summ-model"); f.add_argument("--summ-base-url")
    f.add_argument("--summ-api-key-env", default="OPENAI_API_KEY"); f.add_argument("--dummy", action="store_true")

    s = sp.add_parser("sheet"); s.set_defaults(fn=cmd_sheet)
    s.add_argument("--traces", required=True); s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--out", required=True)

    j = sp.add_parser("judge"); j.set_defaults(fn=cmd_judge)
    j.add_argument("--traces", required=True); j.add_argument("--judge-models", nargs="+")
    j.add_argument("--judge", nargs=4, action="append", metavar=("LABEL", "MODEL", "BASE_URL", "KEY_ENV"),
                   help="repeatable; one judge per provider, e.g. --judge grok grok-4-fast-non-reasoning https://api.x.ai/v1 XAI_API_KEY")
    j.add_argument("--base-url"); j.add_argument("--api-key-env", default="OPENAI_API_KEY")
    j.add_argument("--rubrics", default="all"); j.add_argument("--forms", default="raw,summary_long,summary_short")
    j.add_argument("--out", required=True); j.add_argument("--dummy", action="store_true")
    j.add_argument("--allow-single", action="store_true", help="run one judge only; cat the output files together afterwards")
    j.add_argument("--resume", action="store_true", help="keep rows already in --out and retry only what is missing")
    j.add_argument("--dry-run", action="store_true", help="show how many calls would be made, then stop (no cost)")
    j.add_argument("--yes", action="store_true", help="do not ask for confirmation before more than 100 calls")
    j.add_argument("--system-prefix", default="", help="line put before every judge system prompt, e.g. \"Reasoning strength: low\" (a model setting; it is saved with each judgment)")
    j.add_argument("--workers", type=int, default=1, help="parallel calls (try 4; lower it if you see rate-limit failures)")

    n = sp.add_parser("analyze"); n.set_defaults(fn=cmd_analyze)
    n.add_argument("--judgments", required=True); n.add_argument("--labels")
    n.add_argument("--out", default="results/report.md")
    n.add_argument("--complete-cases", action="store_true",
                   help="use only cells that every judge in the file scored (refusals count as missing)")

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()