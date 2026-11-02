"""Pilot analysis and the pre-set continue/stop gate. Standard library only.

Gate thresholds are proposals fixed BEFORE the pilot. Change them before running,
not after seeing results.
"""
from __future__ import annotations

import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path

from .features import from_row
from .rubrics import RUBRICS

GATE = {
    # v2 (after the literature check): rubric-to-rubric rate gaps are already expected from prior work on
    # judge sensitivity, so a large rate ratio alone is NOT enough to continue. It is reported, not gating.
    "rate_ratio_info": 2.0,       # informational only
    "rate_ratio_stop": 1.5,       # all rubrics within this ratio -> candidate stop
    "flip_margin": 0.05,          # a ranking flip needs a gap of at least this size in BOTH directions
    "judge_noise_ratio": 0.5,     # judge-vs-judge disagreement / rubric-vs-rubric disagreement
    "kappa_high": 0.8,            # judge-human agreement considered high
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def kappa(a: list[int], b: list[int]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def load_labels(path: str | Path) -> dict[str, dict]:
    """label sheet CSV -> {trace_id: Features-as-dict}. Skips unlabeled rows, reports problems."""
    out, bad = {}, []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not (row.get("mentions_possible_test") or "").strip():
                continue
            feats = from_row(row)
            probs = feats.problems()
            if probs:
                bad.append((row["trace_id"], probs))
                continue
            out[row["trace_id"]] = feats
    if bad:
        print("WARNING: inconsistent labels skipped:", bad[:5], f"({len(bad)} total)")
    return out


def _valid(rows):
    return [r for r in rows if r["aware"] is not None]


def complete_cases(rows: list[dict]) -> tuple[list[dict], dict]:
    """Keep only (trace, form, rubric) cells that EVERY judge in the file scored.

    A refusal or unparsable reply is missing data, not a label. If one judge refuses more often, comparing the
    judges on all of their own cells would mix a judge difference with a difference in which cells were scored.
    Returns (kept rows, info about what was dropped).
    """
    judges = sorted({r["judge"] for r in rows})
    by = defaultdict(dict)
    for r in rows:
        by[(r["trace_id"], r["form"], r["rubric"])][r["judge"]] = r["aware"]
    ok = {k for k, v in by.items() if all(v.get(j) is not None for j in judges)}
    kept = [r for r in rows if (r["trace_id"], r["form"], r["rubric"]) in ok]
    dropped = defaultdict(int)
    for k in by:
        if k not in ok:
            dropped[k[2]] += 1
    return kept, {"judges": judges, "cells_total": len(by), "cells_kept": len(ok), "dropped_by_rubric": dict(dropped)}


def rates_table(rows: list[dict]) -> dict:
    """aware rate by (form, rubric, judge)."""
    cells = defaultdict(list)
    for r in _valid(rows):
        cells[(r["form"], r["rubric"], r["judge"])].append(r["aware"])
    return {k: {"rate": _mean(v), "n": len(v), "ci": wilson(sum(v), len(v))} for k, v in cells.items()}


def rubric_rates_raw(rows: list[dict]) -> dict[str, float]:
    """aware rate per rubric on raw traces, averaged over judges."""
    t = rates_table(rows)
    per = defaultdict(list)
    for (form, rub, _j), v in t.items():
        if form == "raw":
            per[rub].append(v["rate"])
    return {k: _mean(v) for k, v in per.items()}


def rate_ratio(rates: dict[str, float]) -> float:
    vals = [v for v in rates.values() if not math.isnan(v)]
    if not vals:
        return float("nan")
    lo, hi = min(vals), max(vals)
    if hi == 0:
        return 1.0
    return float("inf") if lo == 0 else hi / lo


def _rates_by_rubric(rows: list[dict], key: str) -> dict[str, dict[str, float]]:
    cells = defaultdict(list)
    for r in _valid(rows):
        if r["form"] == "raw":
            cells[(r["rubric"], r[key])].append(r["aware"])
    out = defaultdict(dict)
    for (rub, grp), v in cells.items():
        out[rub][grp] = _mean(v)
    return out


def model_rates_by_rubric(rows: list[dict]) -> dict[str, dict[str, float]]:
    return _rates_by_rubric(rows, "model")


def category_rates_by_rubric(rows: list[dict]) -> dict[str, dict[str, float]]:
    return _rates_by_rubric(rows, "category")


def _flips(mr: dict[str, dict[str, float]], margin: float) -> list[tuple[str, str, str, str]]:
    """Pairs (a, b) where one rubric has a > b by >= margin and another has b > a by >= margin."""
    groups = sorted({m for d in mr.values() for m in d})
    flips = []
    for a, b in itertools.combinations(groups, 2):
        up = [r for r, d in mr.items() if a in d and b in d and d[a] - d[b] >= margin]
        down = [r for r, d in mr.items() if a in d and b in d and d[b] - d[a] >= margin]
        if up and down:
            flips.append((a, b, up[0], down[0]))
    return flips


def ranking_flips(rows: list[dict], margin: float = 0.0) -> list[tuple[str, str, str, str]]:
    """Model pairs whose order differs across rubrics (each side by >= margin)."""
    return _flips(model_rates_by_rubric(rows), max(margin, 1e-12))


def category_flips(rows: list[dict], margin: float = 0.0) -> list[tuple[str, str, str, str]]:
    """Prompt-category pairs whose order differs across rubrics (each side by >= margin)."""
    return _flips(category_rates_by_rubric(rows), max(margin, 1e-12))


def disagreement(rows: list[dict]) -> dict:
    """Mean pairwise disagreement on binary labels, raw traces only.
    rubric_vs_rubric: same trace, same judge, different rubrics.
    judge_vs_judge:   same trace, same rubric, different judges.
    """
    by = {(r["trace_id"], r["rubric"], r["judge"]): r["aware"] for r in _valid(rows) if r["form"] == "raw"}
    traces = {k[0] for k in by}
    rubrics = sorted({k[1] for k in by})
    judges = sorted({k[2] for k in by})
    rr, jj = [], []
    for t in traces:
        for j in judges:
            for a, b in itertools.combinations(rubrics, 2):
                if (t, a, j) in by and (t, b, j) in by:
                    rr.append(int(by[(t, a, j)] != by[(t, b, j)]))
        for r in rubrics:
            for a, b in itertools.combinations(judges, 2):
                if (t, r, a) in by and (t, r, b) in by:
                    jj.append(int(by[(t, r, a)] != by[(t, r, b)]))
    rv, jv = _mean(rr), _mean(jj)
    return {"rubric_vs_rubric": rv, "judge_vs_judge": jv,
            "judge_noise_ratio": (jv / rv) if rv and not math.isnan(rv) and not math.isnan(jv) else float("nan")}


def eta_squared(rows: list[dict], factor: str) -> float:
    """Share of variance in binary labels explained by one factor (marginal)."""
    ys = [(r[factor], r["aware"]) for r in _valid(rows)]
    if not ys:
        return float("nan")
    mu = _mean(y for _, y in ys)
    sst = sum((y - mu) ** 2 for _, y in ys)
    if sst == 0:
        return 0.0
    groups = defaultdict(list)
    for g, y in ys:
        groups[g].append(y)
    ssb = sum(len(v) * (_mean(v) - mu) ** 2 for v in groups.values())
    return ssb / sst


def variance_decomposition(rows: list[dict]) -> dict:
    return {f: eta_squared(rows, f) for f in ("rubric", "judge", "form", "model")}


def agreement_with_gold(rows: list[dict], gold: dict) -> dict:
    """Judge vs feature-derived gold, per (form, rubric, judge). Gold always comes from RAW-trace labels."""
    cells = defaultdict(lambda: ([], []))
    for r in _valid(rows):
        feats = gold.get(r["trace_id"])
        if feats is None:
            continue
        rub = RUBRICS[r["rubric"]]
        g = rub.binary(rub.gold(feats))
        cells[(r["form"], r["rubric"], r["judge"])][0].append(g)
        cells[(r["form"], r["rubric"], r["judge"])][1].append(r["aware"])
    out = {}
    for k, (g, p) in cells.items():
        tp = sum(1 for a, b in zip(g, p) if a and b)
        fp = sum(1 for a, b in zip(g, p) if not a and b)
        fn = sum(1 for a, b in zip(g, p) if a and not b)
        out[k] = {"n": len(g), "gold_rate": _mean(g), "judge_rate": _mean(p),
                  "accuracy": _mean(int(a == b) for a, b in zip(g, p)),
                  "kappa": kappa(g, p),
                  "recall": tp / (tp + fn) if tp + fn else None,
                  "precision": tp / (tp + fp) if tp + fp else None}
    return out


def gold_rates(gold: dict) -> dict[str, float]:
    return {n: _mean(r.binary(r.gold(f)) for f in gold.values()) for n, r in RUBRICS.items()}


def evaluate_gate(rows: list[dict], gold: dict | None = None) -> dict:
    """CONTINUE needs a ranking flip (models or prompt categories) with margin, or high judge noise.
    The rate ratio is reported but does not gate: rate gaps between rubrics are expected from prior work."""
    rr = rubric_rates_raw(rows)
    ratio = rate_ratio(rr)
    margin = GATE["flip_margin"]
    mflips = ranking_flips(rows, margin)
    cflips = category_flips(rows, margin)
    dis = disagreement(rows)
    kappas = []
    if gold:
        kappas = [v["kappa"] for k, v in agreement_with_gold(rows, gold).items()
                  if k[0] == "raw" and v["kappa"] is not None]
    mean_kappa = _mean(kappas) if kappas else float("nan")
    c_model = bool(mflips)
    c_cat = bool(cflips)
    c_noise = (not math.isnan(dis["judge_noise_ratio"])) and dis["judge_noise_ratio"] >= GATE["judge_noise_ratio"]
    any_flip = bool(ranking_flips(rows, 0.0) or category_flips(rows, 0.0))
    stop = (ratio < GATE["rate_ratio_stop"]) and not any_flip and (not math.isnan(mean_kappa)) \
        and mean_kappa >= GATE["kappa_high"]
    verdict = "CONTINUE" if (c_model or c_cat or c_noise) else ("STOP_OR_SHRINK" if stop else "INCONCLUSIVE")
    return {"verdict": verdict, "rubric_rates_raw": rr, "rate_ratio": ratio,
            "ranking_flips": mflips, "category_flips": cflips,
            "disagreement": dis, "mean_kappa_raw": mean_kappa, "criteria": {
                "model_ranking_flip": c_model, "category_ranking_flip": c_cat,
                "judge_noise_ratio>=0.5": c_noise, "stop_conditions_met": stop},
            "thresholds": GATE}


def summary_retention(rows: list[dict]) -> dict:
    """Aware rate on summaries as a fraction of the raw rate, per rubric (judge-averaged)."""
    t = rates_table(rows)
    out = defaultdict(dict)
    for rub in {k[1] for k in t}:
        raw = _mean(v["rate"] for k, v in t.items() if k[0] == "raw" and k[1] == rub)
        for form in ("summary_long", "summary_short"):
            vals = [v["rate"] for k, v in t.items() if k[0] == form and k[1] == rub]
            if vals:
                out[rub][form] = (_mean(vals) / raw) if raw else float("nan")
    return dict(out)


def report(rows: list[dict], gold: dict | None, n_labeled: int | None = None) -> str:
    g = evaluate_gate(rows, gold)
    L = ["# Pilot report", ""]
    L.append(f"**Gate verdict: {g['verdict']}**  (labeled traces: {n_labeled if n_labeled is not None else 'none'})")
    L += ["", "## Gate criteria", ""]
    L.append(f"- Rate ratio (most inclusive / strictest rubric, raw): {g['rate_ratio']:.2f}  (informational; does not gate)")
    L.append(f"- Model ranking flips across rubrics (margin {GATE['flip_margin']}): {g['ranking_flips'] or 'none'}")
    L.append(f"- Prompt-category ranking flips across rubrics (margin {GATE['flip_margin']}): {g['category_flips'] or 'none'}")
    d = g["disagreement"]
    L.append(f"- Disagreement: rubric-vs-rubric {d['rubric_vs_rubric']:.3f}, judge-vs-judge {d['judge_vs_judge']:.3f}, ratio {d['judge_noise_ratio']:.2f} (continue if >= {GATE['judge_noise_ratio']})")
    L.append(f"- Mean judge-human kappa on raw traces: {g['mean_kappa_raw']:.2f}  (stop-candidate needs >= {GATE['kappa_high']})")
    L += ["", "## Aware rate per rubric (raw traces, judge-averaged)", ""]
    for k, v in sorted(g["rubric_rates_raw"].items()):
        L.append(f"- {k}: {v:.3f}")
    if gold:
        L += ["", "## Human-feature gold rate per rubric", ""]
        for k, v in sorted(gold_rates(gold).items()):
            L.append(f"- {k}: {v:.3f}")
    L += ["", "## Variance in labels explained by each factor (eta squared)", ""]
    for k, v in variance_decomposition(rows).items():
        L.append(f"- {k}: {v:.3f}")
    L += ["", "## Summary retention (aware rate on summary / raw)", ""]
    for rub, d2 in sorted(summary_retention(rows).items()):
        L.append(f"- {rub}: " + ", ".join(f"{f}={v:.2f}" for f, v in d2.items()))
    if gold:
        L += ["", "## Judge vs human-feature gold (kappa / recall / precision)", ""]
        for k, v in sorted(agreement_with_gold(rows, gold).items()):
            kap = "n/a" if v["kappa"] is None else f"{v['kappa']:.2f}"
            rec = "n/a" if v["recall"] is None else f"{v['recall']:.2f}"
            pre = "n/a" if v["precision"] is None else f"{v['precision']:.2f}"
            L.append(f"- {k[0]} / {k[1]} / {k[2]}: n={v['n']}, kappa={kap}, recall={rec}, precision={pre}")
    setups = {}
    for r in rows:
        m = r.get("judge_meta") or {}
        if m:
            k = (r["judge"], m.get("provider"), m.get("served_model") or m.get("requested_model"))
            d = setups.setdefault(k, {"n": 0, "cut": 0, "first": m.get("utc"), "last": m.get("utc")})
            d["n"] += 1
            d["cut"] += int(m.get("finish_reason") == "length")
            d["last"] = max(d["last"] or "", m.get("utc") or "")
    if setups:
        L += ["", "## Judge setup (report this with any result)", ""]
        for (j, prov, mod), d in sorted(setups.items(), key=lambda x: str(x[0])):
            L.append(f"- {j}: provider {prov}, model {mod}, {d['n']} calls, {d['cut']} cut off by the token limit, "
                     f"{d['first']} to {d['last']}")
    L += ["", "Thresholds are pre-set proposals. Report them as set, not as tuned. "
              "Rubric prompts are the published originals (third_party_rubrics/SOURCES.md lists source, licence and verification status); "
              "thresholds and human-feature mappings are ours."]
    return "\n".join(L)