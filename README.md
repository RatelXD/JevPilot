# JevPilot

JevPilot is a browser agent that accepts a user goal and operates a browser.
It uses an LLM for planning and text generation.
It can use Jev to select the next observed browser action.

The product has two run modes:

- `llm_only`: the LLM selects each action.
- `jev_hybrid`: Jev selects each observed action. The LLM plans, generates input text, and replans after a failure.

The goal is to complete browser work faster and with less model cost than
LLM-only use, without reducing task success. This is a goal. It is not a
measured result yet.

## Run a browser task

JevPilot accepts a URL, a natural-language goal, and at least one completion
condition. The code checks the completion condition. A model's `DONE` choice
does not prove success.

Example:

```bash
uv run --env-file .env jevpilot run \
  --mode jev_hybrid \
  --url http://127.0.0.1:8000/search \
  --goal "Search for Aurora and open its details page." \
  --verify-url-path /items/aurora \
  --verify-text "Aurora details" \
  --live
```

The CLI supports these completion checks:

- `--verify-url-path`: require an exact final URL path.
- `--verify-text`: require this text in the visible page body.

If both checks are present, both must pass.
JevPilot prints a run status and writes a JSON trace under `artifacts/runs/`.
The trace contains model calls, browser actions, verification results, and
available usage and cost data. It does not contain raw prompts or generated
text values.

The CLI starts a new browser context for each run.
It allows the initial URL origin by default.
Use `--allow-origin` for each additional exact origin required by the page.
JevPilot blocks requests to other origins.
Do not use a personal signed-in browser profile.
The current release does not support login flows or irreversible tasks.

### Configure an LLM provider and Jev

Select the LLM account path with `JEVPILOT_LLM_PROVIDER`:

- `gateway` uses the Chosun University API Gateway and its multi-model API key.
- `codex_subscription` uses the existing ChatGPT subscription through Codex CLI.

For a Gateway run, put its API key in the same local `.env` file as the Jev key.
The API key is not passed to the Codex subprocess:

```text
JEVPILOT_LLM_PROVIDER=gateway
JEVPILOT_LLM_API_KEY=<Chosun Gateway API key>
JEVPILOT_LLM_MODEL=gpt-6.1-sol
JEVPILOT_JEV_API_KEY=<Jev API key>
JEVPILOT_JEV_MODEL=jev-latest
JEVPILOT_JEV_OPERATION_THRESHOLD=0.5
JEVPILOT_JEV_TARGET_THRESHOLD=0.5
```

TypeSafe currently maps `jev-latest` to `jev-1.13.0`; the response model is
recorded in traces because the alias may move.
For the final demo, authenticate with the official CLI using `codex login`,
then set `JEVPILOT_LLM_PROVIDER=codex_subscription` and the chosen model ID.
Gateway model IDs must be enabled for the user's Chosun Gateway account; the
model list is account-specific.

For a local `.env` file, pass it to uv explicitly; uv does not load it
automatically. Keep `.env` ignored by git and set its permissions to `600`.
`.env.example` is a tracked template: merge its `KEY=value` entries into an
existing `.env` instead of replacing a file that may contain a Jev key.
The product CLI has no JevPilot-imposed spend cap: the selected provider bills
the configured account or subscription. Each Jev response prints and records
its input/output tokens and a USD estimate from TypeSafe's published input rate;
the exact billed amount is not returned by the API. Gateway USD billing remains
unknown because its account usage is denominated in credits.
There is no automatic fallback to a different provider.

## Compare the two modes

Use the task runner for repeatable local tasks:

```bash
uv run --env-file .env python benchmarks/tasks/run_suite.py \
  --mode compare \
  --task all \
  --repeat 5 \
  --smoke-gate \
  --live
```

The runner executes the same task in both modes. It alternates run order
across repeats. It stores every run, including failures and timeouts.
Currently defined tasks are:

- `T-00`: open an item from a catalog list.
- `T-01`: enter a search value, apply the search, and open the matching item.
- `T-02`: select the North filter and open the matching item.
- `T-03`: fill a two-step synthetic profile form, select a region, and submit.
- `T-04`: enter a search value, wait for an autocomplete suggestion, and open it.
- `T-05`: select two filters, apply them, and open the matching item.

`--task all --repeat 5` schedules 5 tasks × 2 modes × 5 repeats (50 comparison
runs). With `--smoke-gate`, the runner first completes T-01, T-02, and T-03 in
both modes (6 smoke runs), then starts the 50-run comparison only if all six
independently verify and the shared Gateway/Jev test budgets still have room.
Both phases share the same run and reservation budgets.
The benchmark runner applies test-only spending controls: Gateway account
balance must fit within the 10,000-credit test ceiling, and Jev calls use a
conservative reservation up to the $1 test budget. These controls do not apply
to the product CLI. The runner records Gateway balance before and after the
suite; the provider's credit balance is authoritative.
The benchmark's Jev reservation is $0.01 per transmission up to $1 total.
Reservations are conservative budget accounting, not invoice amounts.

Use only synthetic data in the local fixtures.
The user CLI and task runner use real model integrations.
Automated tests use local browser pages and scripted responses only to test
specific failure and edge cases. There is no product `mock` mode.

Online runs can take different actions. The resulting browser states can
differ. The comparison measures both complete systems. It does not claim that
every selector receives the same state after the first different action.

## Current M0 capability

The current implementation supports common, visible HTML/ARIA controls:

- `CLICK`
- `TYPE_TEXT`
- `SELECT` for an observed native select option
- `SCROLL_UP` and `SCROLL_DOWN` for the main page viewport
- bounded `WAIT` for a relevant page-state change
- `DONE`
- `BLOCKED`

The observer builds candidates from the current page. The executor binds each
candidate to an observed DOM node. Before a mutation, it checks the document,
target meaning, enabled state, visibility, and hit-test result. A decision is
consumed once. If a mutation may have occurred but its result is unknown,
JevPilot checks page state and does not retry the mutation automatically.

Scroll, condition-based wait, frames, shadow DOM, canvas, uploads, new tabs,
OS-level control, personal profiles, and payment, deletion, or external
message tasks are outside the current implementation scope.

## Development and tests

Use Python 3.12 or later, `uv`, Pydantic, Playwright, and pytest.

```bash
uv sync --extra dev
uv run python -m playwright install chromium
uv run pytest -q
```

The tests do not call model APIs. The live CLI requires `--live`, explicit
environment settings, and an approved Jev spend reservation. No performance
improvement is claimed until the agreed repeated comparison is complete.

See `JevPilot-implementation-design.md` and `JevPilot-plan.md` in the parent
workspace for the implementation plan and its current decisions.
