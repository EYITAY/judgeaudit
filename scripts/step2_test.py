"""Step 2 test: generate traces for 5 prompts (one per category) from each reasoner and print what came back.

  python3 scripts/step2_test.py                          # gpt-oss and DeepSeek, defaults
  python3 scripts/step2_test.py --only gptoss --effort high
  python3 scripts/step2_test.py --only deepseek --max-tokens 8192
  python3 scripts/step2_test.py --only qwen              # third candidate reasoner on Groq

Costs a few cents. Run only after scripts/check_all.py shows PASS for both reasoners.
Edit REASONERS if you changed models in Step 1.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from awarejudge.env import load_env  # noqa: E402

REASONERS = [  # (name, model, base_url, key_env, extra request fields as JSON or None)
    ("gptoss", "openai/gpt-oss-120b", "https://api.groq.com/openai/v1", "GROQ_API_KEY", None),
    ("deepseek", "deepseek-reasoner", "https://api.deepseek.com", "DEEPSEEK_API_KEY", None),
]
QWEN = ("qwen", "qwen/qwen3.8-27b", "https://api.groq.com/openai/v1", "GROQ_API_KEY", '{"reasoning_format":"parsed"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["gptoss", "deepseek", "qwen"], help="run one reasoner only")
    ap.add_argument("--effort", choices=["low", "medium", "high"], help="reasoning_effort for gpt-oss")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--suffix", default="", help="added to output file names, e.g. _high")
    args = ap.parse_args()
    load_env()
    reasoners = [r for r in REASONERS + [QWEN] if (args.only is None and r[0] != "qwen") or r[0] == args.only]
    os.makedirs(os.path.join(ROOT, "traces"), exist_ok=True)
    for name, model, base, env, extra in reasoners:
        if args.effort and name == "gptoss":
            extra = json.dumps({"reasoning_effort": args.effort})
        out = os.path.join(ROOT, "traces", f"test_{name}{args.suffix}.jsonl")
        cmd = [sys.executable, "-m", "awarejudge.cli", "gen", "--prompts", os.path.join(ROOT, "data", "test5.jsonl"),
               "--model-name", name, "--model", model, "--base-url", base, "--api-key-env", env,
               "--samples", "1", "--max-tokens", str(args.max_tokens), "--out", out] + (["--extra-json", extra] if extra else [])
        print(f"\n=== {name} ({model}) ===")
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=os.environ)
        print((p.stdout + p.stderr).strip()[-600:])
        if p.returncode != 0 or not os.path.exists(out):
            continue
        rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
        print(f"traces returned: {len(rows)} of 5")
        for r in rows:
            m = r.get("meta", {})
            print(f"- {r['prompt_id']:<10} reasoning {len(r['raw']):>6} chars | answer {len(r['answer']):>5} chars | "
                  f"tokens out {m.get('completion_tokens')} | finish {m.get('finish_reason')} | served {m.get('served_model')}")
        if rows:
            print("sample reasoning (first 300 chars):", repr(rows[0]["raw"][:300]))
    print("\nCheck: 5 of 5 traces per model, finish should be 'stop' (not 'length'), reasoning non-empty.")


if __name__ == "__main__":
    main()