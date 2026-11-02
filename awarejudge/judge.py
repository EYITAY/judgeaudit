"""Run rubric x judge x trace-form cells and record one judgment per cell."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .rubrics import RUBRICS
from .traces import FORMS, ChatModel, Trace

ACCOUNT_RE = re.compile(r"Your team|credits|spending limit|insufficient|billing|quota|per day|\(TPD\)|\(RPD\)|"
                        r"invalid api key|incorrect api key|unauthorized|HTTP 401", re.I)


class StopRun(RuntimeError):
    """The provider says the account cannot serve requests (credits, spending cap, daily limit, bad key).
    Every further call would fail the same way, so the run stops. Nothing is written for these calls."""


def parse_label(reply: str, n_levels: int) -> int | None:
    """Generic {"label": k} parser. Used only by the endpoint check scripts, not by the published rubrics."""
    m = re.search(r'"label"\s*:\s*(\d+)', reply) or re.search(r"\b(\d)\b", reply)
    if not m:
        return None
    v = int(m.group(1))
    return v if 0 <= v < n_levels else None


def run_judgments(traces: list[Trace], judges: dict[str, ChatModel], rubric_names: list[str],
                  forms: list[str], out_path: str | Path | None = None, resume: bool = False,
                  workers: int = 1, stats: dict | None = None, dry_run: bool = False) -> list[dict]:
    """Score every trace x form x rubric x judge. With out_path each row is appended the moment it arrives, so a
    crash loses at most the calls in flight. With resume, rows already in the file are kept and skipped; failed
    calls were never written, so they are retried. workers > 1 runs calls in parallel (each call uses its own copy
    of the client so metadata is not mixed). Failures are counted in `stats`, never silently dropped."""
    import copy
    from concurrent.futures import ThreadPoolExecutor, as_completed
    stats = stats if stats is not None else {}
    stats.update({"ok": 0, "skipped_existing": 0, "failed": 0, "unparsable": 0, "blocked": 0, "truncated": 0, "failed_keys": []})
    rows, done = [], set()
    path = Path(out_path) if out_path else None
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if resume and path.exists():
            rows = load_judgments(path)
            done = {(r["trace_id"], r["form"], r["rubric"], r["judge"]) for r in rows}
        elif path.exists():
            path.unlink()
    jobs = []
    for t in traces:
        for form in forms:
            if form != "raw" and form not in t.forms:
                continue
            text = t.text(form)
            for rn in rubric_names:
                for jn in judges:
                    if (t.trace_id, form, rn, jn) in done:
                        stats["skipped_existing"] += 1
                    else:
                        jobs.append((t, form, text, rn, jn))
    stats["to_run"] = len(jobs)
    stats["est_input_tokens"] = sum(len(RUBRICS[j[3]].render(j[0].prompt, j[2], j[0].answer)) for j in jobs) // 4
    if dry_run:                       # plan only: no judge is called, nothing is written
        return rows

    def one(job):
        t, form, text, rn, jn = job
        rub = RUBRICS[rn]
        judge = copy.copy(judges[jn])
        try:
            reply, _ = judge.chat(rub.system, rub.render(t.prompt, text, t.answer), max_tokens=rub.max_tokens,
                                  temperature=0.0)
        except RuntimeError as e:
            if ACCOUNT_RE.search(str(e)):
                raise StopRun(str(e)) from e
            if "HTTP 403" not in str(e) and "HTTP 400" not in str(e):
                raise
            # The provider refused this exact input (content filter). Retrying cannot help, so the cell is recorded
            # as blocked, counted, and left out of the rates; it is never silently dropped.
            return {"trace_id": t.trace_id, "model": t.model, "category": t.category, "form": form, "rubric": rn,
                    "judge": jn, "level": None, "aware": None, "blocked": True, "reply": str(e)[:400],
                    "judge_meta": {}}
        level = rub.parse(reply)
        jm = getattr(judge, "last_meta", None) or {}
        return {"trace_id": t.trace_id, "model": t.model, "category": t.category, "form": form, "rubric": rn,
                "judge": jn, "level": level, "aware": None if level is None else rub.binary(level),
                "reply": reply,
                "judge_meta": {k: jm.get(k) for k in ("provider", "requested_model", "served_model", "temperature",
                                                      "finish_reason", "system_prefix", "utc")}}

    def record(row):
        rows.append(row)
        stats["ok"] += 1
        if row.get("blocked"):
            stats["blocked"] += 1
        elif row["aware"] is None:
            stats["unparsable"] += 1
        if (row["judge_meta"] or {}).get("finish_reason") == "length":
            stats["truncated"] += 1
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        if stats["ok"] % 100 == 0:
            print(f"  {stats['ok']} of {len(jobs)} judgments done", flush=True)

    def fail(job, e):
        key = (job[0].trace_id, job[1], job[3], job[4])
        stats["failed"] += 1
        stats["failed_keys"].append(key)
        print(f"  FAILED {key}: {str(e)[:300]}", flush=True)

    if workers <= 1:
        for job in jobs:
            try:
                record(one(job))
            except StopRun as e:
                stats["stopped"] = str(e)
                break
            except Exception as e:  # noqa: BLE001
                fail(job, e)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(one, j): j for j in jobs}
            for fu in as_completed(futs):
                try:
                    record(fu.result())
                except StopRun as e:
                    if "stopped" not in stats:
                        stats["stopped"] = str(e)
                        for other in futs:
                            other.cancel()          # calls not yet started are dropped; running ones finish
                except Exception as e:  # noqa: BLE001
                    if not fu.cancelled():
                        fail(futs[fu], e)
    return rows


def load_judgments(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]