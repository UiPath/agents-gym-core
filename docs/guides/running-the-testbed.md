# Running the testbed

The testbed runs agents over their evaluation sets, scores every datapoint and aggregates the
scores. It is `TestbedRunnerCore` in `src/agents_gym_core/testbed_core.py`.

```bash
uv run gym eval --agents calculator \
    --models gpt-4.1-mini-2025-04-14 --experiment_name calc_smoke
```

`gym eval` loads agent configurations directly through `CoreAgentRegistry` and Hydra.
Use `gym uipath` for the [UiPath CLI workflow](#the-gym-command). Output goes to
`experiments/<experiment_name>/` ([run-output-formats.md](run-output-formats.md)).

## Before a run

- **Credentials**: see the README (Install, Configure models). `.env` at the repo root is
  loaded on import (`GymSettingsCore`); variables already exported in the shell win.
- **Backend**: `--model_backend` becomes the process default (`set_default_backend`), so LLM
  judges, models built inside tools and subprocesses use it too. A `"backend"` key in a JSON
  `--models` spec overrides it for that spec.
- **Init data**: aime_2025_agent, hotpotqa, hover, ifbench, livebench_math and papillon
  generate their eval sets in an `init:` script. Run `uv run gym init <agent>` once or pass
  `--reinit` to prepare it before evaluation. Initialization is checked before an agent's
  evaluation files are loaded.

## Flags

| Flag | Default | Effect |
|---|---|---|
| `--agents` | `calculator:base` | Agent selectors (below); duplicates dropped. |
| `--loops` | `basic` | Loop specs, e.g. `basic:any planning`; no version means `auto`. |
| `--models` | `gpt-4o-2024-11-20` + `text-embedding-3-large` | Model specs after one flag. A repeated `--models` replaces the earlier one. |
| `--model_backend` | `$AGENTS_GYM_LLM_BACKEND`, else `langchain` | `uipath` (UiPath LLM Gateway) or `langchain` (providers' API keys). |
| `--max_datapoints N` | `0` (all) | First N datapoints in eval-set order; with `--splits`, N per split. |
| `--datapoints A B` | all | Exact datapoint names; overrides `--max_datapoints` and `--splits`. |
| `--splits train val` | all | Datapoints whose eval-set file carries that split label. |
| `--num_runs N` | `1` | Repeat each datapoint; saved as `run_1/` … `run_N/`. |
| `--aggregation_mode` | `datapoint_run` | Or `evaluator_run`, `run_datapoint` ([formulas](run-output-formats.md#aggregation)). |
| `--max_concurrency N` | `20` | Concurrent run-and-evaluate tasks. |
| `--mode` | `fill` | See [Run modes](#run-modes). |
| `--output_dir` | `<repo>/experiments` | `<repo>/experiments` whatever the cwd; `./experiments` for a wheel install; `<project>/experiments` with `gym --project`. |
| `--experiment_name` | `experiment` | Subdirectory of `--output_dir`; use a distinct name for each configuration. |
| `--reinit` | off | Run each selected agent's init script with `force=True` first (as `gym init <agent> --force`). |
| `--run_report` | off | Execute `reporting/report_template.ipynb`; writes `report_executed.ipynb` and `.html` next to `combined_results.json`. |
| `--skip_saving_traces` | off | No `output.json`, `cost.json` or `generated_content/`; `results/` is still written. |
| `--handle_tool_errors` | off | Return a tool exception to the agent as `{"error": ...}`. Off: the stock `uipath_langchain` tool node lets it propagate and fail the run. |
| `--parallel_tool_calls` | off | Allow several tool calls per model turn. |
| `--print_trace`, `--verbose` | off | Stream the agent trace; verbose logging. |

| Selector | Resolves to |
|---|---|
| `claim_history` / `claim_history:test` | `base` / that version from `versions.yaml` |
| `claim_history:*` | `base` and every version |
| `*:base`, `*:*` | every agent's `base`; every agent and version |
| `@fast-benchmark` | agents with that `metadata.tags` entry; `@core` = every agent in `src/agents_gym_core/agents/` |

- **Model specs** are a name (`gpt-5-2025-08-07`) or JSON: `model`, `embeddings` (default
  `text-embedding-3-large`), role models for the loops that use them (`planning_llm`,
  `validation_llm`, `summarizing_llm`),
  `backend`, `from_factory`. Each spec gets its own loop key and directory, except that the
  loop key holds only the models the loop uses: a spec that differs from an earlier one only
  in another role, `backend` or `from_factory` runs once.
- **Agent loops**: an `agent.yaml` `loop:` (the six public benchmarks use
  `dspy-program:<program class or instance>`) replaces the default `basic` in `--loops`, and only that:
  `--agents hover --loops basic:any` runs hover on `basic:any`.
- **Datapoint names** match the eval set's `name`, not the directory (where `/` becomes `_`).
  A `--datapoints` or `--splits` filter that matches nothing logs a warning and exits 0.
- **Split labels** come from eval-set filenames: `_train`, `_val`, `_test`, `_dev` (or `-train`
  etc.), or `__<label>`. Unlabelled files are `all_data`, selected only when `all_data` is listed.

## Run modes

| `--mode` | Agent runs a datapoint when | Evaluator runs when |
|---|---|---|
| `fill` | no valid `output.json`, or the datapoint errored | the agent runs, or the result file is missing or errored |
| `overwrite` | always | always |
| `fill_rerun_evals` | as `fill` | always |
| `fill_skip_errors` | no valid `output.json` | the agent runs, or no result file |
| `fill_skip_run_errors` | no valid `output.json`, or the model endpoint failed | the agent runs, no result file, an unreadable one, or the evaluator raised; any other failed agent run keeps its results |

Valid `output.json`: exists, non-empty, parses as JSON (a corrupt one is re-run with the
warning `output.json is corrupt and will be re-run`). Errored: a file in `results/` has
`agent_output.type == "error"` or cannot be read. Datapoints not re-run are loaded from disk.

Details of result reuse:

- **Result reuse follows the experiment path**:
  `<experiment>/<agent>/<version>/<loop_name>/<loop_key>/<datapoint>/run_N`.
  The loop key records the loop, version and model names. Use a new experiment name or
  `--mode overwrite` when changing prompts, tools, evaluators, backend or temperature.
- **An evaluator that raised runs again.** An evaluator exception (a judge with bad
  credentials, say) is saved as `score: 0.0`, `details: "Exception thrown by evaluator: ..."`.
  `fill` and `fill_skip_run_errors` run that evaluator again on the saved output;
  `fill_skip_errors` keeps the zero.
- **An endpoint outage runs again in `fill_skip_run_errors`.** A run that failed because the
  model endpoint had an outage (`APIConnectionError`, `RateLimitError`,
  `AuthenticationError`, a timeout or a server error; `ENDPOINT_ERROR_TITLES` in `run_utils.py`)
  is an outage, not an agent failure, so the datapoint runs again.
- **Saved evaluator results determine error status**: a run is classified as errored when
  its `results/` files record an agent error or cannot be read.
- **Reloaded outputs are the saved ones.** A datapoint loaded from disk is scored against
  `agent_output.json`, what the agent returned (an error included). A run saved without that
  file is scored against the output parsed from `output.json`, where a single string field
  comes back as the bare string, not the dict the original run returned.
- **A datapoint run that raises is not scored.** The agent and the evaluators record their own
  errors, so this is anything else (criteria that fail validation, a disk error). The testbed
  lists such runs and exits with code 1; rerunning the command retries them.
- **Trace saving provides resumable outputs.** `--skip_saving_traces` suppresses new output
  writes; `fill` reuses valid saved outputs and reruns datapoints without them.
- **`overwrite` reruns the selected datapoints in place** and retains files from other selections.

## Recipes

```bash
# Smoke test: 3 datapoints, fresh
uv run gym eval --agents calculator --max_datapoints 3 \
    --mode overwrite --models gpt-4.1-mini-2025-04-14 --experiment_name calc_smoke
# Two versions x two models in one experiment
uv run gym eval --agents claim_history:baseline-test claim_history:test \
    --models gpt-4.1-mini-2025-04-14 gpt-5-2025-08-07 --experiment_name claim_cmp
# Public benchmark: build splits, then 5 runs with a report
uv run gym init hover
uv run gym eval --agents hover:test --num_runs 5 --run_report \
    --models gpt-4.1-mini-2025-04-14 --experiment_name hover_test
# Re-score saved outputs (e.g. after fixing judge credentials)
uv run gym eval --agents papillon:test --mode fill_rerun_evals \
    --models gpt-4.1-mini-2025-04-14 --experiment_name papillon_test
```

`gym reproduce <table>` runs the paper's multi-agent, multi-version evaluations, one table at a
time ([REPRODUCING](../REPRODUCING.md#tables-to-versions)).

## Reading the results

An `EXPERIMENT INVENTORY` is printed before and after the run: completed, errored, not run
and to be run, for datapoints and for evaluations. `To Be Run: 0` in both at the start means
every result comes from disk.

```
experiments/<experiment_name>/
├── final_score_per_agent.json        # headline: per loop key, agent, split; CI and cost
├── combined_results.json             # one row per datapoint × evaluator × run
├── agg_results_on_metrics.csv, agg_results_on_datapoints.csv
└── <agent>/<version>/<loop_name>/<loop_key>/<datapoint>/run_N/
    └── output.json, agent_output.json, cost.json, results/<evaluator_id>.json
```

Invocations that produce scores write top-level summaries for their selected agents, loops
and models. Per-datapoint results remain available in their experiment directories.
Confidence intervals are computed across repeated runs; `--num_runs 1` reports the single
run's score.

## Loops

| Spec | Behavior |
|---|---|
| `basic` (= `basic:auto`) | ReAct loop, `tool_choice="auto"`. |
| `basic:any` | `tool_choice="any"`: a tool call every turn. |
| `basic:auto-graceful`, `basic:any-graceful` | Force `end_execution` at the call limit or on repeated tool calls instead of failing. |
| `planning`, `planning:system-plan`, `planning:user-plan` | Plan first; the plan is added as a message, or merged into the system or user message. |
| `validation` | Validate the output before terminating. |
| `summarizing`, `summarizing:any` | Summarize long histories before each model call. |
| `deep-agents` | The `deepagents` harness: todos, filesystem tools, sub-agents. |
| `dspy-program:<dotted path>` | A DSPy program (class or instance); the version is its import path. |
| `<package.module.LoopClass>` | A loop class by import path. |

## Environment settings

| Variable | Effect | Default |
|---|---|---|
| `AGENTS_GYM_LLM_BACKEND` | Backend when `--model_backend` is absent | `langchain` |
| `UIPATH_LLM_SERVICE` | Gateway service for `uipath`; `llmgateway` uses the `LLMGW_*` variables in `.env.example` | `agenthub` |
| `AGENTS_GYM_MODEL_OVERRIDES` | Comma-separated `module:function` hooks run before models resolve | unset |
| `UIPATH_GYM_SKIP_INIT` | `true` skips init checks and scripts (`--reinit` becomes a no-op) | `false` |
| `UIPATH_AGENTS_GYM_TEMPERATURE` | Agent-model temperature; a model spec's own config can override it | `1.0` |
| `UIPATH_AGENTS_GYM_MAX_RETRIES` | Agent-model retries | `5` |
| `UIPATH_AGENTS_GYM_LOOP_RECURSION_LIMIT` | Max LLM calls per execution | `20` |
| `UIPATH_AGENTS_GYM_NO_TOOL_MESSAGES_LIMIT` | Max consecutive LLM messages without a tool call | `5` |

## The `gym` command

| Command | Role |
|---|---|
| `gym eval [flags]` | The testbed described above (`gym eval --help`). |
| `gym optimize <agent:version> [flags]` | Prompt optimization with ReAPO: `--engine react` (the default) or `--engine claude` (`gym optimize --help`). |
| `gym new <name> [--directory DIR]`, `gym --project DIR init\|eval\|optimize ...` | Create and use a standalone project for your own agent ([bring-your-agent.md](bring-your-agent.md)). |
| `gym init <agent> [--version V] [--force]`, `gym init --list` | Build an agent's local data (what `--reinit` calls); skipped when the `.gym_init_done` marker matches the init params. |
| `gym uipath generate [--agents a,b] [--with-evaluations] [--no-init]` | Write `langgraph.json` and `sdk_entrypoints.py` for the UiPath CLI, then run `uipath init`. The files are gitignored; an existing `CLAUDE.md`/`AGENTS.md` is kept. |
| `gym uipath run <agent> [index\|json]` | One datapoint through `uipath run`. |
| `gym uipath eval <agent> [--eval-set NAME] [--workers N] [--report]` | One eval set through `uipath eval`; results in `experiments/sdk_results/`. |

`gym uipath run` and `gym uipath eval` ensure the entry points include the selected agent and reflect its current `agent.yaml`/`versions.yaml` before execution. `uipath init` builds every listed agent, so model credentials must be configured.
