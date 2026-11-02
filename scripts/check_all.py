"""Step 1: run the endpoint check for every planned model and print one summary.

  python scripts/check_all.py

Reads keys from environment variables (never prints them). Models whose key is missing are skipped.
Edit PLAN below if a model name or URL turns out to be wrong. Paste the SUMMARY block back to Claude.
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from awarejudge.env import load_env  # noqa: E402
PLAN = [  # (label, role, base_url, model, key_env)
    ("reasoner A", "reasoner", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "GROQ_API_KEY"),
    ("reasoner B", "reasoner", "https://api.deepseek.com", "deepseek-reasoner", "DEEPSEEK_API_KEY"),
    ("judge 1", "judge", "https://api.x.ai/v1", "grok-4-fast-non-reasoning", "XAI_API_KEY"),
    ("judge 2", "judge", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    ("summarizer", "judge", "https://api.openai.com/v1", "gpt-4o", "OPENAI_API_KEY"),
]


def apply_overrides():
    """Optional .env lines JUDGE2_MODEL, JUDGE2_BASE_URL, JUDGE2_KEY_ENV replace judge 2 without editing code."""
    m, b, k = (os.environ.get(x) for x in ("JUDGE2_MODEL", "JUDGE2_BASE_URL", "JUDGE2_KEY_ENV"))
    if not (m or b or k):
        return
    i = next(i for i, row in enumerate(PLAN) if row[0] == "judge 2")
    lab, role, base, model, env = PLAN[i]
    PLAN[i] = (lab, role, b or base, m or model, k or env)


def main():
    names = load_env()
    apply_overrides()
    if names:
        print(f"Loaded from .env: {', '.join(names)} (values not shown)")
    override = os.environ.get("CHECK_BASE_OVERRIDE")  # for offline testing only
    results = []
    for label, role, base, model, env in PLAN:
        print(f"\n=== {label}: {model} ({role}) ===")
        if not os.environ.get(env) and not override:
            print(f"SKIPPED: {env} is not set")
            results.append((label, model, "SKIPPED", f"{env} not set"))
            continue
        cmd = [sys.executable, os.path.join(HERE, "check_endpoint.py"), "--base-url", override or base,
               "--model", model, "--api-key-env", env if os.environ.get(env) else "PATH", "--role", role]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=os.environ)
        out = (p.stdout + p.stderr).strip()
        print(out)
        last = out.splitlines()[-1] if out else "no output"
        results.append((label, model, "PASS" if p.returncode == 0 else "FAIL", last[:160]))
    print("\n\n===== SUMMARY (safe to paste; contains no keys) =====")
    for label, model, status, note in results:
        print(f"{status:8} {label:11} {model:30} {note}")


if __name__ == "__main__":
    main()
