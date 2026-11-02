# Step 1: setup with six providers

Keys live in a file named `.env` in the kit's top folder (the one containing `awarejudge/` and `scripts/`). The kit reads it automatically. `.env` is listed in `.gitignore` so it is never committed.

1. Copy `.env.example` to a new file named exactly `.env`.
2. Open `.env` in VS Code and paste each key after its `=` sign. No spaces, no quotes needed.
3. Save. Do not share the file, paste it into chat, or screenshot it.

Alternative (temporary, lost when the terminal closes): `export GROQ_API_KEY="..."` on Mac or Linux, `$env:GROQ_API_KEY="..."` in Windows PowerShell. Variables already set in the terminal win over `.env`.

Base URLs (confirm against each provider's docs; check_endpoint.py tells you if one is wrong):
- OpenAI `https://api.openai.com/v1`
- DeepSeek `https://api.deepseek.com`
- Groq `https://api.groq.com/openai/v1`
- xAI `https://api.x.ai/v1`
- Gemini (OpenAI-compatible) `https://generativelanguage.googleapis.com/v1beta/openai`

## Proposed roles (verify each with the check script before committing)
| Role | Model | Why |
|---|---|---|
| Reasoner A | Groq `openai/gpt-oss-120b` | Open-weight; check that `reasoning` comes back |
| Reasoner B | DeepSeek reasoning model (try `deepseek-reasoner`) | Open-weight, different family. `deepseek-chat` is not a reasoning model |
| Judge 1 | xAI `grok-4-fast-non-reasoning` | Non-reasoning, different family from both reasoners |
| Judge 2 | Groq `llama-3.3-70b-versatile` (Meta Llama) | Non-reasoning, different family from all others. Check the name in Groq's model list; override with JUDGE2_MODEL in .env |
| Summarizer | OpenAI `GPT-4o` | Non-reasoning; summarizes reliably |
Gemini Flash stays in reserve (as a third judge if budget allows). Gemini and GPT-4o are not open-weight; they are not used to generate the traces.

## Checks
```
python scripts/check_endpoint.py --base-url https://api.groq.com/openai/v1 --model openai/gpt-oss-120b --api-key-env GROQ_API_KEY
python scripts/check_endpoint.py --base-url https://api.deepseek.com --model deepseek-reasoner --api-key-env DEEPSEEK_API_KEY
python scripts/check_endpoint.py --base-url https://api.x.ai/v1 --model grok-4-fast-non-reasoning --api-key-env XAI_API_KEY --role judge
python scripts/check_endpoint.py --base-url https://api.groq.com/openai/v1 --model llama-3.3-70b-versatile --api-key-env GROQ_API_KEY --role judge
python scripts/check_endpoint.py --base-url https://api.openai.com/v1 --model gpt-4o --api-key-env OPENAI_API_KEY --role judge
```
Reasoners must print PASS. Judges must print PASS with a parsed label.

## Step 2 commands (after the checks pass)
```
python -m awarejudge.cli gen --model-name gptoss --model openai/gpt-oss-120b --base-url https://api.groq.com/openai/v1 --api-key-env GROQ_API_KEY --samples 2 --out traces/a.jsonl
python -m awarejudge.cli gen --model-name deepseek --model deepseek-reasoner --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY --samples 2 --out traces/b.jsonl
cat traces/a.jsonl traces/b.jsonl > traces/all.jsonl
python -m awarejudge.cli forms --traces traces/all.jsonl --summ-model gpt-4o --summ-base-url https://api.openai.com/v1 --summ-api-key-env OPENAI_API_KEY --out traces/forms.jsonl
python -m awarejudge.cli judge --traces traces/forms.jsonl \
  --judge grok grok-4-fast-non-reasoning https://api.x.ai/v1 XAI_API_KEY \
  --judge llama llama-3.3-70b-versatile https://api.groq.com/openai/v1 GROQ_API_KEY \
  --out results/judgments.jsonl
```
Run each with a small `--samples 1` test on 5 prompts first and read the output before the full run.

---

# Appendix: Multi-provider or single provider? (discussion, 4 October 2026)

**Question:** What is the impact of using several providers versus one, and what are the merits and demerits?

| | Multi-provider (six separate keys) | Single provider (an aggregator) |
|---|---|---|
| Setup | Six keys, six billing accounts, six base URLs | One key, one URL, one bill |
| Code and errors | Each API has quirks: parameter names, reasoning field names, rate limits | One format and one set of errors to learn |
| Reasoning text | The model's own host usually returns the fullest reasoning | An aggregator may route to different backends, and some hide or trim reasoning |
| Judge independence | Judges from different companies are easy to choose | Also possible, if the aggregator hosts them |
| Reproducibility | Harder; others need your exact accounts and models | Easier; one key and one catalog |
| Cost | Uses credits and pricing you already have | One markup layer, simple to track |
| Reliability | One provider failing stops only its model | One outage or rate limit stops everything |
| Data handling | Data goes to six companies' terms | Goes to one intermediary plus whoever it routes to |

**How each could distort the results**
- *Multi-provider:* a difference between two judges now includes a difference in infrastructure, such as default settings or how the host handles the system prompt. For a judge audit this is a real risk. Mitigation: judges run at temperature 0; re-run a small sample to measure repeat-call noise; state it in the write-up, since this noise sits inside the measured judge-to-judge disagreement.
- *Single provider:* routing can silently change the backend or quantization between calls. Pin a provider if the aggregator allows it; otherwise record which backend served each call.

**Decision**
1. Keep the multi-provider setup: it is already paid for, it gives the cleanest reasoning text, and the judges are independent of each other and of the reasoners.
2. Log metadata on every call (done in the kit, see below).
3. Write the exact setup into the report so a reader knows what to rerun.
4. If a native provider hides reasoning, consider an aggregator for that one model only.

Caveat: none of these providers has been tested yet. Their actual reasoning output is the first thing `check_endpoint.py` shows.

**Metadata now logged (added after this discussion)**
- Each trace stores `meta`: provider host, requested model, model name the server reports, temperature, token limit, prompt and completion tokens, finish reason, UTC time. Each summary records the same under `meta.summarizer`.
- Each judgment row stores `judge_meta`: provider, requested and served model, temperature, finish reason, UTC time.
- The report ends with a "Judge setup" section listing each judge's provider, served model, number of calls, how many were cut off by the token limit, and the date range. Include that section in any write-up.
- Old trace files without `meta` still load.
