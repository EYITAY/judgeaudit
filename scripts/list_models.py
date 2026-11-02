"""List the model IDs a provider offers to your key, so you can pick valid names.

  python3 scripts/list_models.py --base-url https://api.groq.com/openai/v1 --api-key-env GROQ_API_KEY
  python3 scripts/list_models.py --base-url https://generativelanguage.googleapis.com/v1beta/openai --api-key-env GOOGLE_API_KEY

Optional: --grep llama   (only show IDs containing the text)
Prints IDs only; never prints your key.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from awarejudge.env import load_env  # noqa: E402
from awarejudge.traces import USER_AGENT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key-env", required=True)
    ap.add_argument("--grep", default="")
    a = ap.parse_args()
    load_env()
    key = os.environ.get(a.api_key_env, "")
    if not key:
        sys.exit(f"{a.api_key_env} is not set")
    req = urllib.request.Request(a.base_url.rstrip("/") + "/models",
                                 headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}")
    except Exception as e:  # noqa: BLE001
        sys.exit(f"{type(e).__name__}: {e}")
    items = data.get("data") or data.get("models") or []
    ids = sorted(str(m.get("id") or m.get("name")) for m in items)
    ids = [i for i in ids if a.grep.lower() in i.lower()]
    print(f"{len(ids)} model(s):")
    for i in ids:
        print(" ", i)


if __name__ == "__main__":
    main()
