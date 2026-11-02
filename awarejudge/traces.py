"""Traces, trace forms (raw vs summarized), and an OpenAI-compatible chat client.

A "model" here is anything with .chat(system, user, max_tokens, temperature) ->
(content, reasoning). The OpenAI-compatible client works with vLLM, Together,
OpenRouter, etc. Open-weight reasoning models usually return their reasoning in
`reasoning_content` or `reasoning`; we read either. Check your provider returns it.
"""
from __future__ import annotations

import datetime
import http.client
import json
import re
import time
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Protocol

USER_AGENT = "awarejudge/0.1 (research pilot; python-urllib)"  # some providers block the bare Python default (HTTP 403, code 1010)
FORMS = ("raw", "summary_long", "summary_short")
SUMMARY_WORDS = {"summary_long": 150, "summary_short": 60}

SUMMARIZER_SYSTEM = (
    "You write neutral summaries of another model's private reasoning. Describe what the "
    "reasoning considered and decided. Do not add information that is not in the reasoning."
)


@dataclass
class Trace:
    trace_id: str
    model: str
    prompt_id: str
    category: str
    prompt: str
    raw: str                                    # raw reasoning
    answer: str = ""
    forms: dict = field(default_factory=dict)   # form name -> text (raw is always included)
    meta: dict = field(default_factory=dict)    # provider, model ids, settings, date: who produced this trace

    def text(self, form: str) -> str:
        return self.raw if form == "raw" else self.forms[form]


def load_traces(path: str | Path) -> list[Trace]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(Trace(**json.loads(line)))
    return out


def save_traces(traces: list[Trace], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")


class ChatModel(Protocol):
    def chat(self, system: str, user: str, max_tokens: int = 2048, temperature: float = 1.0) -> tuple[str, str]: ...


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY", timeout: int = 300,
                 extra: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.key = os.environ.get(api_key_env, "")
        self.timeout = timeout
        self.provider = urllib.parse.urlparse(self.base_url).netloc
        self.extra = dict(extra or {})   # extra request fields, e.g. {"reasoning_effort": "high"}
        self.system_prefix = ""          # optional model-control line put before the system prompt, e.g. "Reasoning strength: low"
        self.last_meta: dict = {}

    def chat(self, system: str, user: str, max_tokens: int = 2048, temperature: float = 1.0,
             retries: int = 8) -> tuple[str, str]:
        if self.system_prefix:
            system = self.system_prefix + ("\n\n" + system if system else "")
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        body = json.dumps({"model": self.model, "messages": msgs, "max_tokens": max_tokens,
                           "temperature": temperature, **self.extra}).encode()
        last = None
        for attempt in range(retries):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}",
                         "User-Agent": USER_AGENT, "Accept": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())
                msg = data["choices"][0]["message"]
                usage = data.get("usage") or {}
                self.last_meta = {"provider": self.provider, "requested_model": self.model,
                                  "served_model": data.get("model", ""), "temperature": temperature,
                                  "max_tokens": max_tokens, "extra": self.extra, "system_prefix": self.system_prefix, "prompt_tokens": usage.get("prompt_tokens"),
                                  "completion_tokens": usage.get("completion_tokens"),
                                  "finish_reason": data["choices"][0].get("finish_reason"),
                                  "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                content = msg.get("content") or ""
                if not reasoning and "</think>" in content:  # some providers inline thinking in the reply text
                    head, _, tail = content.partition("</think>")
                    reasoning = head.replace("<think>", "", 1).strip()
                    content = tail.strip()
                return content, reasoning
            except urllib.error.HTTPError as e:
                try:
                    err_body = e.read()[:300].decode(errors="replace")   # NOT `body`: that is the request, reused on retry
                except Exception:  # noqa: BLE001
                    err_body = ""
                last = f"HTTP {e.code} {err_body}".strip()
                if e.code not in (408, 409, 429, 500, 502, 503, 504):
                    raise RuntimeError(f"{self.model}: HTTP {e.code}: {err_body[:300]}") from e
                if e.code == 429 and re.search(r"per day|\(TPD\)|\(RPD\)", err_body, re.I):
                    # a daily cap does not clear by waiting a minute; retrying only wastes time
                    raise RuntimeError(f"{self.model}: HTTP 429 (daily limit): {err_body[:300]}") from e
                ra = e.headers.get("Retry-After") if e.headers else None
                if ra and ra.replace(".", "", 1).isdigit():  # provider told us how long to wait
                    time.sleep(min(float(ra), 120) + 1)
                    continue
            except (OSError, http.client.HTTPException, json.JSONDecodeError, KeyError) as e:  # incl. ssl.SSLError, dropped connections, bad bodies
                last = e
            time.sleep(2 ** attempt * 2)
        raise RuntimeError(f"{self.model}: failed after {retries} tries: {last}")


def load_prompts(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def generate_traces(prompts: list[dict], model: ChatModel, model_name: str, samples: int,
                    max_tokens: int = 4096, temperature: float = 1.0, out_path: str | Path | None = None,
                    resume: bool = False, stats: dict | None = None) -> list[Trace]:
    """Generate traces. With out_path, each trace is appended to the file as soon as it arrives, so a crash
    loses at most one call. With resume, trace ids already in the file are skipped. Failed calls (after the
    client's own retries) and replies with no visible reasoning are counted in `stats`, not silently lost."""
    stats = stats if stats is not None else {}
    stats.update({"ok": 0, "skipped_existing": 0, "no_reasoning": 0, "failed": 0, "truncated": 0, "failed_ids": []})
    done = set()
    out: list[Trace] = []
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if resume and path.exists():
            out = load_traces(path)
            done = {t.trace_id for t in out}
        elif path.exists():
            path.unlink()
    for p in prompts:
        for s in range(samples):
            tid = f"{model_name}:{p['id']}:{s}"
            if tid in done:
                stats["skipped_existing"] += 1
                continue
            try:
                answer, reasoning = model.chat(p.get("system", ""), p["prompt"], max_tokens, temperature)
            except Exception as e:  # noqa: BLE001  (network/provider errors after retries)
                stats["failed"] += 1
                stats["failed_ids"].append(tid)
                print(f"  FAILED {tid}: {str(e)[:120]}", flush=True)
                continue
            meta = dict(getattr(model, "last_meta", {}) or {})
            if not reasoning.strip():
                stats["no_reasoning"] += 1  # no visible reasoning: not usable for this study
                continue
            if meta.get("finish_reason") == "length":
                stats["truncated"] += 1
            t = Trace(trace_id=tid, model=model_name, prompt_id=p["id"], category=p["category"],
                      prompt=p["prompt"], raw=reasoning, answer=answer, meta=meta)
            out.append(t)
            stats["ok"] += 1
            if out_path is not None:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
            if stats["ok"] % 10 == 0:
                print(f"  {stats['ok']} traces so far", flush=True)
    return out


def add_summaries(traces: list[Trace], summarizer: ChatModel, checkpoint=None) -> None:
    """Fill trace.forms with summaries at two compression levels. Mutates in place.
    `checkpoint(traces)` is called after each trace so progress can be saved; forms already present are skipped."""
    for i, t in enumerate(traces, 1):
        for form, words in SUMMARY_WORDS.items():
            if form in t.forms:
                continue
            user = f"Summarize this reasoning in about {words} words:\n\n{t.raw}"
            content, _ = summarizer.chat(SUMMARIZER_SYSTEM, user, max_tokens=600, temperature=0.0)
            sm = getattr(summarizer, "last_meta", None)
            if sm:
                t.meta.setdefault("summarizer", {})[form] = dict(sm)
            t.forms[form] = content.strip()
        if checkpoint is not None:
            checkpoint(traces)
        if i % 10 == 0:
            print(f"  summarized {i}/{len(traces)}", flush=True)