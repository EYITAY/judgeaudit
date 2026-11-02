"""End-to-end plumbing tests on synthetic data. They do NOT validate any scientific claim."""
import csv
import math
import random

from awarejudge.analysis import (agreement_with_gold, category_flips, disagreement, evaluate_gate, kappa, load_labels,
                                 rate_ratio, ranking_flips, report, wilson)
from awarejudge.cli import DummyModel
from awarejudge.features import Features, from_row
from awarejudge.judge import parse_label, run_judgments
from awarejudge.rubrics import RUBRICS
from awarejudge.traces import Trace, add_summaries, generate_traces, load_prompts, load_traces


def test_prompts_valid():
    ps = load_prompts("data/pilot_prompts.jsonl")
    assert len(ps) == 50 and len({p["id"] for p in ps}) == 50
    assert len({p["category"] for p in ps}) == 5


def test_feature_validation_and_gold_mappings():
    f = Features(True, True, "alignment_safety", False, True)
    assert f.problems() == []
    assert RUBRICS["li_evalawarebench"].gold(f) == 1 and RUBRICS["goodfire_f1"].gold(f) == 1
    assert RUBRICS["deckenbach_3level"].gold(f) == 3 and RUBRICS["repverb_0_3"].gold(f) == 3
    hedge = Features(True, False, "capability", False, False)
    assert RUBRICS["li_evalawarebench"].gold(hedge) == 0 and RUBRICS["goodfire_f1"].gold(hedge) == 1
    assert RUBRICS["deckenbach_3level"].gold(hedge) == 1 and RUBRICS["repverb_0_3"].gold(hedge) == 2
    art = Features(False, False, "none", True, False)
    assert art.problems() == []
    assert RUBRICS["goodfire_f1"].gold(art) == 0 and RUBRICS["deckenbach_3level"].gold(art) == 1
    assert RUBRICS["repverb_0_3"].gold(art) == 1
    assert Features(True, False, "none", False, False).problems()  # test mentioned but eval_type none


def test_published_prompts_render_and_parse():
    for name, r in RUBRICS.items():
        p = r.render("THE-TASK", "THE-REASONING", "THE-ANSWER")
        assert "THE-REASONING" in p, name
        assert "{chain of thought}" not in p and "{judged_text}" not in p and "{{ messages }}" not in p, name
    li, f1, dk, c1 = (RUBRICS[n] for n in ("li_evalawarebench", "goodfire_f1", "deckenbach_3level", "repverb_0_3"))
    assert "THE-TASK" in li.render("THE-TASK", "r", "a") and "THE-ANSWER" in li.render("t", "r", "THE-ANSWER")
    assert li.parse('```json\n{"model_awareness": {"conclusion": "Yes", "awareness_type": "safety"}}\n```') == 1
    assert li.parse('{"model_awareness": {"conclusion": "No"}, "hypothetical_recognition": {"conclusion": "Yes"}}') == 0
    assert li.parse("no json here") is None
    assert f1.parse("REASONING: it says test\nQUOTE: This might be a test") == 1
    assert f1.parse("REASONING: nothing\nQUOTE: None") == 0
    assert f1.parse("REASONING: no test talk.\n\nQUOTE: None\n\nNOTE 1: None.") == 0   # echoed NOTE line is not a quote
    assert f1.parse('REASONING: r\nQUOTE: "This might be a test"\n\nNOTE 1: None.') == 1
    assert f1.parse("QUOTE: None") is None                      # missing REASONING: treated as unparsable
    assert dk.parse('{"evaluation_awareness": "RECOGNITION", "evaluation_awareness_traits": [["PLACEHOLDER", "x"]]}') == 2
    assert dk.parse('{"value": "instrumental", "evaluation_awareness_traits": []}') == 3
    assert dk.parse('{"evaluation_awareness": "MAYBE"}') is None
    assert c1.parse("2") == 2 and c1.parse(" 3. ") == 3 and c1.parse("**1**") == 1 and c1.parse("7") is None
    assert (dk.aware_threshold, c1.aware_threshold, li.aware_threshold, f1.aware_threshold) == (2, 2, 1, 1)


def test_from_row_normalizes_conclusion():
    f = from_row({"mentions_possible_test": "false", "concludes_test": "true", "eval_type": "capability",
                  "notes_artificial": "no", "acts_on_belief": "no"})
    assert f.mentions_possible_test is True


def test_parse_label():
    assert parse_label('{"label": 2, "evidence": "x"}', 4) == 2
    assert parse_label('{"label": 7}', 4) is None
    assert parse_label("garbage", 4) is None


def test_stats():
    assert kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi
    assert abs(rate_ratio({"a": 0.1, "b": 0.3}) - 3.0) < 1e-9
    assert rate_ratio({"a": 0.0, "b": 0.3}) == float("inf")


def _synthetic(tmp_path, n=40):
    rng = random.Random(1)
    traces = []
    for i in range(n):
        model = "m1" if i % 2 == 0 else "m2"
        traces.append(Trace(f"{model}:p{i}:0", model, f"p{i}", "cat", "prompt",
                            "This might be a test." if rng.random() < 0.5 else "Plain reasoning."))
    add_summaries(traces, DummyModel("summarizer"))
    return traces


def test_full_pipeline_with_dummies(tmp_path):
    traces = _synthetic(tmp_path)
    judges = {"j1": DummyModel("judge"), "j2": DummyModel("judge")}
    rows = run_judgments(traces, judges, list(RUBRICS), ["raw", "summary_long", "summary_short"],
                         tmp_path / "j.jsonl")
    assert len(rows) == 40 * 3 * len(RUBRICS) * 2
    assert all(r["aware"] in (0, 1) for r in rows if r["level"] is not None)
    g = evaluate_gate(rows)
    assert g["verdict"] in ("CONTINUE", "STOP_OR_SHRINK", "INCONCLUSIVE")
    d = disagreement(rows)
    assert 0 <= d["rubric_vs_rubric"] <= 1

    # label sheet round trip
    sheet = tmp_path / "labels.csv"
    with open(sheet, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trace_id", "model", "category", "reasoning", "mentions_possible_test", "concludes_test",
                    "eval_type", "notes_artificial", "acts_on_belief", "notes"])
        for t in traces[:20]:
            aware = "test" in t.raw
            w.writerow([t.trace_id, t.model, t.category, t.raw, str(aware).lower(), "false",
                        "other_or_unspecified" if aware else "none", "false", "false", ""])
    gold = load_labels(sheet)
    assert len(gold) == 20
    ag = agreement_with_gold(rows, gold)
    assert any(k[0] == "raw" for k in ag)
    text = report(rows, gold, len(gold))
    assert "Gate verdict" in text


def test_gate_detects_planted_rubric_gap_and_flip():
    # rubric A flags everything model m1 does, rubric B flags only m2: rate ratio large and ranking flips
    rows = []
    for i in range(20):
        for model, a, b in (("m1", 1, 0), ("m2", 0, 1)):
            for rub, val in (("li_evalawarebench", a), ("goodfire_f1", b)):
                rows.append({"trace_id": f"{model}:{i}", "model": model, "category": "c", "form": "raw",
                             "rubric": rub, "judge": "j", "level": val, "aware": val})
    assert ranking_flips(rows)
    g = evaluate_gate(rows)
    assert g["verdict"] == "CONTINUE"


def test_gate_stops_when_everything_agrees():
    rows, gold = [], {}
    for i in range(20):
        tid = f"m:{i}"
        gold[tid] = Features(False, False, "none", False, False)
        for rub in RUBRICS:
            for j in ("j1", "j2"):
                rows.append({"trace_id": tid, "model": "m", "category": "c", "form": "raw", "rubric": rub,
                             "judge": j, "level": 0, "aware": 0})
    g = evaluate_gate(rows, gold)
    assert g["rate_ratio"] == 1.0 and not g["ranking_flips"]


def _planted(gap_a, gap_b):
    rows = []
    for i in range(20):
        for model, a, b in (("m1", gap_a[0], gap_b[0]), ("m2", gap_a[1], gap_b[1])):
            for rub, val in (("li_evalawarebench", a), ("goodfire_f1", b)):
                rows.append({"trace_id": f"{model}:{i}", "model": model, "category": "c", "form": "raw",
                             "rubric": rub, "judge": "j", "level": val, "aware": val})
    return rows


def test_flip_below_margin_does_not_gate():
    # tiny opposite-direction gaps (0.02) are noise-sized: flip exists at margin 0 but not at 0.05
    rows = []
    for model, ra, rb in (("m1", 0.52, 0.50), ("m2", 0.50, 0.52)):
        for rub, rate in (("li_evalawarebench", ra), ("goodfire_f1", rb)):
            for i in range(100):
                v = int(i < rate * 100)
                rows.append({"trace_id": f"{model}:{i}", "model": model, "category": "c", "form": "raw",
                             "rubric": rub, "judge": "j", "level": v, "aware": v})
    assert ranking_flips(rows) and not ranking_flips(rows, 0.05)
    assert evaluate_gate(rows)["verdict"] != "CONTINUE"


def test_large_rate_gap_alone_does_not_continue():
    rows = []
    for i in range(20):
        for rub, val in (("li_evalawarebench", 0), ("goodfire_f1", 1)):
            for j in ("j1", "j2"):
                rows.append({"trace_id": f"m:{i}", "model": "m", "category": "c", "form": "raw",
                             "rubric": rub, "judge": j, "level": val, "aware": val})
    g = evaluate_gate(rows)
    assert g["rate_ratio"] == float("inf") or g["rate_ratio"] >= 2
    assert not g["ranking_flips"] and not g["category_flips"]
    assert g["disagreement"]["judge_noise_ratio"] == 0
    assert g["verdict"] != "CONTINUE"


def test_category_flip_continues():
    rows = []
    for i in range(20):
        for cat, a, b in (("c1", 1, 0), ("c2", 0, 1)):
            for rub, val in (("li_evalawarebench", a), ("goodfire_f1", b)):
                rows.append({"trace_id": f"{cat}:{i}", "model": "m", "category": cat, "form": "raw",
                             "rubric": rub, "judge": "j", "level": val, "aware": val})
    assert category_flips(rows, 0.05)
    assert evaluate_gate(rows)["verdict"] == "CONTINUE"


class _Flaky:
    """Fails on call 3, returns no reasoning on call 5, otherwise fine."""
    def __init__(self): self.n = 0; self.last_meta = {"finish_reason": "stop"}
    def chat(self, system, user, max_tokens=2048, temperature=1.0):
        self.n += 1
        if self.n == 3: raise RuntimeError("boom")
        if self.n == 5: return "answer", ""
        return "answer", f"reasoning {self.n}"


def test_generate_incremental_and_resume(tmp_path=None):
    import tempfile, os
    d = tempfile.mkdtemp() if tmp_path is None else str(tmp_path)
    out = os.path.join(d, "t.jsonl")
    prompts = [{"id": f"p{i}", "category": "c", "prompt": "x"} for i in range(4)]
    stats = {}
    got = generate_traces(prompts, _Flaky(), "m", 2, out_path=out, stats=stats)
    assert stats["failed"] == 1 and stats["no_reasoning"] == 1 and stats["ok"] == 6, stats
    assert len(load_traces(out)) == 6 == len(got)          # saved as they arrived
    stats2 = {}
    m2 = _Flaky(); m2.n = 100                              # healthy second pass
    got2 = generate_traces(prompts, m2, "m", 2, out_path=out, resume=True, stats=stats2)
    assert stats2["skipped_existing"] == 6 and stats2["ok"] == 2, stats2   # retries the 2 missing ids
    assert len(load_traces(out)) == 8 == len(got2)


def test_client_retries_ssl_errors():
    import ssl, io, json, urllib.request, time
    from awarejudge.traces import OpenAICompatClient
    calls = {"n": 0}
    class R:
        def __enter__(s): return s
        def __exit__(s, *a): return False
        def read(s):
            return json.dumps({"model": "m", "choices": [{"message": {"content": "ok", "reasoning": "r"},
                                                            "finish_reason": "stop"}], "usage": {}}).encode()
    def fake(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ssl.SSLError("bad record mac")
        return R()
    real_open, real_sleep = urllib.request.urlopen, time.sleep
    urllib.request.urlopen, time.sleep = fake, lambda s: None
    try:
        out = OpenAICompatClient("https://x.test/v1", "m", "NOPE").chat("", "hi")
    finally:
        urllib.request.urlopen, time.sleep = real_open, real_sleep
    assert out == ("ok", "r") and calls["n"] == 3


def test_client_retries_http_429_with_bytes_body():
    """Regression: the error handler once overwrote the request body with a str, so the retry crashed."""
    import io, json, urllib.error, urllib.request, time
    from awarejudge.traces import OpenAICompatClient
    seen = []
    class R:
        def __enter__(s): return s
        def __exit__(s, *a): return False
        def read(s):
            return json.dumps({"model": "m", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                               "usage": {}}).encode()
    def fake(req, timeout=None):
        seen.append(type(req.data))
        if len(seen) < 3:
            raise urllib.error.HTTPError(req.full_url, 429, "slow down", {"Retry-After": "0"}, io.BytesIO(b"rate limited"))
        return R()
    real_open, real_sleep = urllib.request.urlopen, time.sleep
    urllib.request.urlopen, time.sleep = fake, lambda s: None
    try:
        out = OpenAICompatClient("https://x.test/v1", "m", "NOPE").chat("", "hi")
    finally:
        urllib.request.urlopen, time.sleep = real_open, real_sleep
    assert out == ("ok", "") and seen == [bytes, bytes, bytes]


def test_provider_block_is_recorded_not_retried(tmp_path=None):
    import tempfile, os
    from awarejudge.judge import run_judgments, load_judgments
    class Blocker:
        last_meta = {}
        n = 0
        def chat(self, system, user, max_tokens=300, temperature=0.0):
            Blocker.n += 1
            raise RuntimeError("m: HTTP 403: {\"code\":\"permission-denied\"}")
    traces = [Trace(trace_id="t0", model="m", prompt_id="p", category="c", prompt="x", raw="r")]
    out = os.path.join(tempfile.mkdtemp(), "j.jsonl")
    st = {}
    rows = run_judgments(traces, {"j": Blocker()}, ["goodfire_f1"], ["raw"], out, stats=st)
    assert st["blocked"] == 1 and st["failed"] == 0 and rows[0]["aware"] is None and rows[0]["blocked"] is True
    st2 = {}
    run_judgments(traces, {"j": Blocker()}, ["goodfire_f1"], ["raw"], out, resume=True, stats=st2)
    assert st2["skipped_existing"] == 1 and Blocker.n == 1       # not retried on resume


def test_account_error_stops_run_and_is_not_saved():
    import tempfile, os
    from awarejudge.judge import run_judgments, load_judgments
    class NoCredits:
        last_meta = {}
        n = 0
        def chat(self, system, user, max_tokens=300, temperature=0.0):
            NoCredits.n += 1
            if NoCredits.n <= 2:
                return "REASONING: r\nQUOTE: None", ""
            raise RuntimeError('m: HTTP 403: {"error":"Your team abc has either used all available credits"}')
    traces = [Trace(trace_id=f"t{i}", model="m", prompt_id="p", category="c", prompt="x", raw="r") for i in range(6)]
    out = os.path.join(tempfile.mkdtemp(), "j.jsonl")
    st = {}
    run_judgments(traces, {"j": NoCredits()}, ["goodfire_f1"], ["raw"], out, stats=st)
    assert "credits" in st["stopped"] and st["blocked"] == 0 and st["failed"] == 0
    assert len(load_judgments(out)) == 2 and NoCredits.n == 3       # stopped at the first account error, saved nothing for it


def test_daily_limit_fails_fast():
    import io, urllib.error, urllib.request, time
    from awarejudge.traces import OpenAICompatClient
    calls = []
    def fake(req, timeout=None):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 429, "x", {"Retry-After": "5"},
                                     io.BytesIO(b'{"error":{"message":"Rate limit reached ... on tokens per day (TPD): Limit 100000"}}'))
    real_open, real_sleep = urllib.request.urlopen, time.sleep
    urllib.request.urlopen, time.sleep = fake, lambda s: None
    try:
        try:
            OpenAICompatClient("https://x.test/v1", "m", "NOPE").chat("", "hi")
            raise AssertionError("should have raised")
        except RuntimeError as e:
            assert "daily limit" in str(e)
    finally:
        urllib.request.urlopen, time.sleep = real_open, real_sleep
    assert len(calls) == 1


def test_summaries_checkpoint_and_skip_existing():
    traces = [Trace(trace_id=f"t{i}", model="m", prompt_id="p", category="c", prompt="x", raw="some reasoning", answer="a")
              for i in range(3)]
    traces[0].forms["summary_long"] = "kept"; traces[0].forms["summary_short"] = "kept"
    saves = []
    class S:
        n = 0; last_meta = {}
        def chat(self, system, user, max_tokens=2048, temperature=1.0):
            S.n += 1; return "sum", ""
    add_summaries(traces, S(), checkpoint=lambda ts: saves.append(len(ts)))
    assert S.n == 4 and traces[0].forms["summary_long"] == "kept" and len(saves) == 3
    assert all(set(t.forms) == {"summary_long", "summary_short"} for t in traces)


def test_judge_incremental_resume_and_parallel():
    import tempfile, os, collections
    from awarejudge.judge import run_judgments, load_judgments
    from awarejudge.rubrics import RUBRICS
    class Flaky:
        n = 0; last_meta = {}
        def chat(self, system, user, max_tokens=300, temperature=0.0):
            Flaky.n += 1
            if Flaky.n % 7 == 0: raise RuntimeError("boom")
            return '{"label": 1, "evidence": "x"}', ""
    traces = [Trace(trace_id=f"t{i}", model="m", prompt_id="p", category="c", prompt="x", raw="r",
                    forms={"summary_long": "s", "summary_short": "s"}) for i in range(6)]
    names = list(RUBRICS)[:2]
    out = os.path.join(tempfile.mkdtemp(), "j.jsonl")
    for workers in (1, 4):
        Flaky.n = 0
        if os.path.exists(out): os.remove(out)
        st = {}
        run_judgments(traces, {"j1": Flaky(), "j2": Flaky()}, names, ["raw", "summary_long"], out, workers=workers, stats=st)
        total = 6 * 2 * 2 * 2
        assert st["failed"] > 0 and st["ok"] + st["failed"] == total, st
        assert len(load_judgments(out)) == st["ok"]          # saved as they arrived, failures not written
        class Good(Flaky):
            def chat(self, *a, **k): return '{"label": 0, "evidence": "x"}', ""
        st2 = {}
        rows = run_judgments(traces, {"j1": Good(), "j2": Good()}, names, ["raw", "summary_long"], out, resume=True, workers=workers, stats=st2)
        assert st2["skipped_existing"] == st["ok"] and st2["failed"] == 0 and len(rows) == total, (st2, len(rows))
        keys = collections.Counter((r["trace_id"], r["form"], r["rubric"], r["judge"]) for r in load_judgments(out))
        assert len(keys) == total and max(keys.values()) == 1   # no duplicates after resume