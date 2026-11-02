# After downloading: what to do, in order (Mac, VS Code)

Send Claude the output named under **Send** at each stop. Never send keys or the contents of `.env`.

## A. Set up the new copy (10 minutes)
1. Unzip the latest `awarejudge-complete.zip` somewhere new (for example the Desktop). You get a folder `awarejudge`.
2. Copy your `.env` into that folder's top level, next to `scripts/` and `awarejudge/`.
3. In VS Code: File, Open Folder, choose that `awarejudge` folder. Terminal, New Terminal.
4. Check you are in the right place and the files are there:
   ```
   ls -a
   ```
   You should see `.env`, `.env.example`, `.gitignore`, `scripts`, `data`, `docs`, `tests`.
5. Check the names inside `.env` (values hidden):
   ```
   sed 's/=.*/=<hidden>/' .env
   ```
   **Send:** this output. Expect five lines: GROQ_API_KEY, DEEPSEEK_API_KEY, XAI_API_KEY, OPENAI_API_KEY (ANTHROPIC_API_KEY and GOOGLE_API_KEY may also be present; they are not required).

## B. Step 1: check the models (2 minutes, a few cents)
```
python3 scripts/check_all.py
```
The first line should say `Loaded from .env:` and list your key names.
- Both reasoners must show `PASS: reasoning text is returned`.
- The three others must show `PASS: judge reply parses`.

**Send:** the SUMMARY block at the end. If a line says FAIL, also send that model's section.
**Stop here** until Claude confirms. Do not run later steps if a reasoner failed.

## C. Step 2 test: 5 traces per reasoning model (a few cents)
```
python3 scripts/step2_test.py
```
Look for: 5 of 5 traces, finish `stop` (not `length`), non-empty reasoning.
**Send:** the full output.

## D. Step 2 full run (about 100 traces per model)
Only after Claude says the test looks good.
```
python3 -m awarejudge.cli gen --model-name gptoss --model openai/gpt-oss-120b --base-url https://api.groq.com/openai/v1 --api-key-env GROQ_API_KEY --samples 2 --out traces/a.jsonl
python3 -m awarejudge.cli gen --model-name deepseek --model deepseek-reasoner --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY --samples 2 --out traces/b.jsonl
cat traces/a.jsonl traces/b.jsonl > traces/all.jsonl
wc -l traces/*.jsonl
```
Expect close to 100 lines in each of a and b. Fewer means some replies had no reasoning and were dropped.
If it stops with an error, keep the files, and send the last 10 lines of output.
**Send:** the `wc -l` output.

## E. Step 3: summaries
```
python3 -m awarejudge.cli forms --traces traces/all.jsonl --summ-model gpt-4o --summ-base-url https://api.openai.com/v1 --summ-api-key-env OPENAI_API_KEY --out traces/forms.jsonl
```
Open `traces/forms.jsonl` in VS Code and read two summaries against their raw reasoning. They must not add facts that are not in the reasoning.
**Send:** one sentence on whether the summaries look faithful.

## F. Step 4: label sheet (needs your second person)
```
python3 -m awarejudge.cli sheet --traces traces/forms.jsonl --n 100 --out labels/sheet.csv
```
1. Read `docs/labeling_guide.md` with the second labeler.
2. Both label the first 30 rows independently, in separate copies of the file, without discussing.
3. Compare. Reconcile differences. If agreement on "mentions a possible test" is poor, tell Claude before going on.
4. Label the remaining rows yourself from the raw reasoning only.
Do not look at any judge output until labeling is finished.
**Send:** your agreement result on the first 30.

## G. Step 5: run the judges
First a small test:
```
python3 -m awarejudge.cli judge --traces traces/forms.jsonl --forms raw --rubrics goodfire_style \
  --judge grok grok-4-fast-non-reasoning https://api.x.ai/v1 XAI_API_KEY \
  --judge llama llama-3.3-70b-versatile https://api.groq.com/openai/v1 GROQ_API_KEY \
  --out results/judge_test.jsonl
```
**Send:** the last line printed. Then the full run:
```
python3 -m awarejudge.cli judge --traces traces/forms.jsonl \
  --judge grok grok-4-fast-non-reasoning https://api.x.ai/v1 XAI_API_KEY \
  --judge llama llama-3.3-70b-versatile https://api.groq.com/openai/v1 GROQ_API_KEY \
  --out results/judgments.jsonl
```
It makes about 6,000 calls. Keep the terminal open until it finishes.

## H. Step 6: report and decision
```
python3 -m awarejudge.cli analyze --judgments results/judgments.jsonl --labels labels/sheet.csv --out results/report.md
```
**Send:** `results/report.md`. Claude reads it against the pre-set gate. Do not change thresholds after seeing it.

## Cost and safety reminders
- Stay under your $50 budget: check each provider's usage page after steps D and G.
- Do not commit or share `.env`, `traces/`, `labels/` or `results/` publicly until Claude and you have checked them. They are listed in `.gitignore`.
- Before publishing anything, replace the rubric texts in `awarejudge/rubrics.py` with the original wording and say which provider and model each judge used (the report's "Judge setup" section).

## If something goes wrong
| What you see | What to do |
|---|---|
| `command not found: python3` | Install Python 3.10+ from python.org, reopen the terminal |
| `No such file or directory: scripts/check_all.py` | You are in the wrong folder; run `ls` and open the right one |
| `SKIPPED: ... not set` for every key | `.env` is missing, misnamed, or the old kit version is open |
| HTTP 401 | The key is wrong or has no credit |
| HTTP 404 or "model not found" | The model name is wrong; check the provider's model list |
| HTTP 429 | Rate limit; wait a few minutes and rerun that command |
| Reasoner says FAIL: no usable reasoning | Provider hides reasoning; tell Claude and we swap it |
