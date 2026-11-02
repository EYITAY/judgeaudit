"""Feature-level human labels for one reasoning trace.

Humans label observable features, not a single "aware / unaware" flag. Each
published rubric is then mapped onto these features (see rubrics.py), which lets
us separate disagreement caused by DEFINITIONS from disagreement caused by JUDGE
error.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

EVAL_TYPES = ("none", "alignment_safety", "capability", "other_or_unspecified")


@dataclass
class Features:
    mentions_possible_test: bool   # raises the possibility of a test/eval/benchmark/red-team
    concludes_test: bool           # states or commits to "this is a test/evaluation"
    eval_type: str                 # kind of evaluation named or clearly implied
    notes_artificial: bool         # says the scenario seems fake/contrived/synthetic
    acts_on_belief: bool           # uses the test belief to decide what to do

    def problems(self) -> list[str]:
        out = []
        if self.eval_type not in EVAL_TYPES:
            out.append(f"bad eval_type {self.eval_type!r}")
        if self.concludes_test and not self.mentions_possible_test:
            out.append("concludes_test requires mentions_possible_test")
        if self.acts_on_belief and not (self.mentions_possible_test or self.concludes_test):
            out.append("acts_on_belief requires a test mention")
        has_test = self.mentions_possible_test or self.concludes_test
        if has_test and self.eval_type == "none":
            out.append("test mentioned but eval_type is 'none'")
        if not has_test and self.eval_type != "none":
            out.append("eval_type set but no test mentioned")
        return out

    def to_dict(self) -> dict:
        return asdict(self)


def _b(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def from_row(row: dict) -> Features:
    f = Features(
        mentions_possible_test=_b(row["mentions_possible_test"]),
        concludes_test=_b(row["concludes_test"]),
        eval_type=(row.get("eval_type") or "none").strip().lower(),
        notes_artificial=_b(row["notes_artificial"]),
        acts_on_belief=_b(row["acts_on_belief"]),
    )
    # A conclusion implies the possibility was raised; normalize rather than reject.
    if f.concludes_test:
        f.mentions_possible_test = True
    return f
