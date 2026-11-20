# Run output formats

What the testbed ([running-the-testbed.md](running-the-testbed.md)) writes, for tooling that
reads runs. Code: `runtime/` (`run_utils.py`, `runtime_helpers.py`, `cost_tracking.py`,
`runners/`) and `utils/trace_utils.py` under `src/agents_gym_core/`.

## Layout

```
<output_dir>/<experiment_name>/
├── combined_results.json, final_score_per_agent.json
├── agg_results_on_metrics.csv, agg_results_on_datapoints.csv
├── report_executed.ipynb, report_executed.html      # --run_report only
└── <agent>/<version>/<loop_name>/<loop_key>/         # "experiment_dir"
    ├── agent_details.json
    ├── evaluation_results.json
    └── <datapoint>/run_<N>/
        ├── output.json
        ├── agent_output.json
        ├── cost.json
        ├── generated_content/                       # only if the inputs have output_path
        └── results/<evaluator_id>.json
```

For example: `experiments/calc_smoke/calculator/base/basic/basic__auto__model:gpt-4.1-mini-2025-04-14,embeddings:text-embedding-3-large/TestSimpleAddition/run_1/`.

- **Loop key** (`LoopDetails.loop_key`): `<loop_name>__<loop_version>__<role>:<model>,...`,
  with `model` first, other roles sorted, `embeddings` last. A DSPy program has no embeddings:
  `dspy-program__gepa_artifact.benchmarks.hover.hover_program.HoverMultiHop__model:gpt-4.1-mini-2025-04-14`.
- **Datapoint directory**: the name with `/` replaced by `_`. `run_<N>/` is always present;
  the testbed numbers runs from 1. Build paths with `get_experiment_path`,
  `get_datapoint_dir` and `get_datapoint_output_file` (`run_utils.py`).
- The top-level files are rewritten by every invocation that produces scores and hold only
  its results; one with none (a `--datapoints` filter that matches nothing, say) leaves the
  previous files in place. The per-datapoint files accumulate across invocations.

| File | Written by | When |
|---|---|---|
| `agent_details.json` | `ExperimentOrchestrator._save_agent_details` | Before execution, once per loop-key directory |
| `agent_output.json` | `save_agent_output` | Just before `output.json`: `{"agent_output": ...}`, what the agent returned (an error dict included), unless it is not JSON |
| `output.json` | `save_execution_data` | After `graph.ainvoke` returns or raises, if a span survives filtering |
| `cost.json` | `save_cost_record`, from `ExecutionManager` | After `output.json`, if at least one LLM call was recorded |
| `generated_content/` | the agent | `ExecutionManager` points an `output_path` input here |
| `results/<evaluator_id>.json` | `RuntimeHelpers.run_and_save_evaluator` | Right after that datapoint's execution |
| `evaluation_results.json` | `save_evaluation_results`, from `TestbedRunnerCore._save_all_results` | After aggregation; one per loop that ran, an agent's own `loop:` included |
| top-level JSON and CSV | `ResultAggregator.save_results` | After all evaluations |

`--skip_saving_traces` suppresses `output.json`, `agent_output.json`, `cost.json` and the `output_path` redirect;
`results/` is always written.

## `output.json`

A JSON array with one element per kept span, each an OTLP JSON message (protobuf field names;
IDs base64-encoded):

```json
[
  {"resource_spans": [{"resource": {}, "scope_spans": [{"spans": [{
    "trace_id": "cUP3Vfryxa2asxNwzRDpvw==", "span_id": "...", "parent_span_id": "...",
    "name": "LangGraph", "kind": "SPAN_KIND_INTERNAL",
    "start_time_unix_nano": "...", "end_time_unix_nano": "...",
    "attributes": [
      {"key": "output.value", "value": {"string_value": "{\"answer\": 7}"}},
      {"key": "openinference.span.kind", "value": {"string_value": "CHAIN"}}
    ],
    "status": {"code": "STATUS_CODE_OK"}
  }]}]}]}
]
```

Kept spans (`_is_relevant_span`): tool spans (any `tool.name` attribute), `LangGraph` and
`terminate` spans of kind `CHAIN`, and spans of kind `LLM`. With none kept, no file is written.

| From | Removed |
|---|---|
| every span | `resource.attributes`, `scope`, `flags`; attributes `input.mime_type`, `output.mime_type`, `metadata` |
| `CHAIN` spans | `input.value` |
| `LLM` spans | `llm.tools.*`, `llm.invocation_parameters`, `llm.input_messages.*`, `llm.output_messages.*`, `llm.token_count.completion_details.*`, `llm.token_count.prompt_details.*`, `llm.system` |

LLM spans keep `input.value`, `output.value`, `llm.model_name`, `llm.provider`,
`llm.finish_reason` and the `llm.token_count.{prompt,completion,total}` counts.

### Agent output from the trace

A fresh run is scored against the graph's return value. A datapoint loaded from disk is scored
against `agent_output.json` when it exists, else against `get_agent_output_from_trace(spans)`:
scanning from the last span backwards, the first `LangGraph`/`terminate` `CHAIN` span whose
`output.value` is a JSON object gives that dict; an `output.value` that is not a JSON object
(valid JSON such as `42` or a list included) gives the stripped string. Nothing found gives
`{}`. The tracer stores a single-key dict with a string value (`Output(result="...")`) as the
bare string. Runs with an `agent_output.json` sidecar use that recorded graph return value;
trace fallback follows the serialized shapes described here.

### Errors

An exception from `graph.ainvoke` is not written to `output.json` (spans recorded before it
are). It becomes the run's `agent_output`, visible in `results/*.json`:

```json
{"type": "error", "code": "GENERAL_EXECUTION_ERROR", "title": "KeyError",
 "category": "SYSTEM", "details": "'missing_key'"}
```

- Any exception other than `AgentRuntimeError` gives this shape (`title` is the exception
  class, `details` is `str(exception)`). Every evaluator is then short-circuited to
  `score: 0.0` with `details` set to that string.
- `AgentRuntimeError` (e.g. `TERMINATION_MAX_ITERATIONS`, or `TERMINATION_LLM_RAISED_ERROR`
  from `raise_error`) gives `code`, `title`, `category`, `details` from its `error_info`.
  Evaluators run normally on the error dict.
- A run that returns `None` (conversational, no `end_execution`) gets
  `{"content": "<text of the last LLM output message>"}`.

## `cost.json`

`CostRecord.model_dump()` with sorted keys, accumulated by `CostTrackingCallback` over the
run's `on_llm_end` events:

```json
{
  "by_model": {"gpt-4.1-mini-2025-04-14": {"cached_tokens": 0, "cost_usd": 0.0003348,
                                           "input_tokens": 713, "num_calls": 2, "output_tokens": 31}},
  "cached_tokens": 0, "driver": {"...": "same fields as the top level"},
  "gateway_priced_calls": 0, "input_tokens": 713, "num_llm_calls": 2,
  "output_tokens": 31, "total_cost_usd": 0.0003348, "unpriced_calls": 0
}
```

| Field | Meaning |
|---|---|
| totals, `by_model` | Every LLM call inside `graph.ainvoke`, agent and tools; `by_model` keyed by the model name in the response metadata. LLM-judge calls are not included. |
| `unpriced_calls` | Calls with no entry in `MODEL_COST_PER_MILLION` and no gateway cost: tokens counted, $0. |
| `gateway_priced_calls` | Calls costed from the gateway's `associated_dollar_cost`, which wins over the price table. |
| `driver` | The same fields for the loop's own chat model only (tagged `gym:driver`, `DRIVER_TAG`). A tool that builds its own model counts in the totals, not here. |

`load_cost_record()` returns `None` for an absent, empty or unparsable file. A datapoint
loaded from disk in fill mode uses its saved `cost.json`.

## `results/<evaluator_id>.json`

```json
{
  "evaluator_id": "calculator_ExactMatchEvaluator",
  "datapoint_name": "TestSimpleAddition",
  "score": 1.0,
  "details": {"expected": "{'answer': 7.0}", "actual": "{'answer': 7.0}"},
  "evaluation_criteria": {"expectedOutput": {"answer": 7}},
  "agent_output": {"answer": 7},
  "split": "all_data"
}
```

| Field | Notes |
|---|---|
| `evaluator_id` | The file name stem. |
| `datapoint_name` | Unsanitized name, no run suffix. |
| `score` | 0.0 to 1.0. |
| `details` | `model_dump()` of a Pydantic justification, else as returned (string, dict or `null`). |
| `evaluation_criteria` | `datapoint.evaluation_criterias[evaluator_id]`, verbatim, except that an all-zero `toolCallsCount` becomes `{}` when no tool ran. |
| `agent_output` | The scored output; the error dict when the agent failed. |
| `split` | `agent.datapoint_splits.get(name, "all_data")`. |
| `judge_model`, `judge_prompt` | Only for `LLMJudgeMixin` evaluators: `evaluator_config.model` and `.prompt`. |

- An evaluator that raises is saved as `score: 0.0` with `details` starting
  `Exception thrown by evaluator:`. `fill` runs that evaluator again; `fill_skip_errors` keeps it.
- Only evaluator IDs present both in the datapoint's `evaluation_criterias` and among the
  agent's loaded evaluators run; other criteria are skipped without a message.

## `agent_details.json` and `evaluation_results.json`

`agent_details.json` records the agent as run: `name`, `version`, `loop`, `loop_details`,
`tags`, `system_prompt`, `user_prompt`, `signature_prompts`, `input_schema`, `output_schema`,
`tools` (name, description), `evaluator_weights`, `evaluation_set_name`,
`evaluation_set_id`, `num_datapoints`.

`evaluation_results.json` is the loop key's rows in the `combined_results.json` schema, except:
`experiment_unique_identifier` is `<agent>_<version>_<loop_key>_<evaluator_id>`,
`datapoint_name` keeps its `:run_<N>` suffix, `split` is always `all_data` (it is not passed
through), and `null` fields are omitted. Use `combined_results.json` for split-aware analysis.

## Aggregated files

`ResultAggregator.finalize()` returns `evaluation_results`, `aggregated_scores`, `metrics_df`,
`datapoints_df`, `costs_df` and `saved_files` (paths of the four files below, present only when there
were scores to save).

`GymRuntime.evaluate()` also returns `evaluation_complete`, `failed_datapoint_runs` and
`failed_evaluator_runs` for completion status and evaluation diagnostics.

### `combined_results.json`

A flat array, one object per datapoint, evaluator and run (`EvaluatorDatapointResult` fields):

```json
{
  "experiment_unique_identifier": "calculator__base_basic__auto__model:gpt-4.1-mini-2025-04-14,embeddings:text-embedding-3-large_TestSimpleAddition:run_1",
  "agent": "calculator", "agent_version": "base",
  "loop_name": "basic", "loop_version": "auto",
  "loop_key": "basic__auto__model:gpt-4.1-mini-2025-04-14,embeddings:text-embedding-3-large",
  "loop_models": {"model": "gpt-4.1-mini-2025-04-14", "embeddings": "text-embedding-3-large"},
  "datapoint_name": "TestSimpleAddition", "score": 1.0,
  "evaluator_id": "calculator_ExactMatchEvaluator",
  "agent_output": {"answer": 7}, "evaluation_criteria": {"expectedOutput": {"answer": 7}},
  "details": {"expected": "{'answer': 7.0}", "actual": "{'answer': 7.0}"},
  "evaluator_weights": {"calculator_ExactMatchEvaluator": 1.0, "calculator_LLMJudgeOutputEvaluator": 0.0},
  "run_number": 1, "split": "all_data"
}
```

- `experiment_unique_identifier` is `<agent>__<version>_<loop_key>_<datapoint>:run_<N>`.
- `score` is rounded to 4 decimals.
- LLM-judge rows include the recorded `judge_model` and `judge_prompt`. Cached results retain
  those values when loaded into combined reports.

### `final_score_per_agent.json`

`{loop_key: {"<agent>__<version>": {split: AgentScoreData}}}`:

```json
{"basic__auto__model:gpt-4.1-mini-2025-04-14,embeddings:text-embedding-3-large": {
  "calculator__base": {"all_data": {
    "loop_name": "basic",
    "runs_mean": 0.883, "runs_ci": [0.875, 0.891], "runs_whiskers": [0.875, 0.891],
    "runs_error_margin": 0.008, "runs": {"run_1": 0.8831, "run_2": 0.891, "run_3": 0.875},
    "cost": {"total_cost_usd": 0.0003, "cost_per_datapoint": 0.0003, "num_runs": 3, "...": "..."}}}}}
```

| Field | Meaning |
|---|---|
| `runs` | Score of each run (aggregated per [`--aggregation_mode`](#aggregation)). |
| `runs_mean` | Mean of `runs`. |
| `runs_ci` | 95% bootstrap percentile interval of that mean (10,000 resamples, seed 42); equals the score with one run. |
| `runs_whiskers` | Min and max of `runs`. |
| `runs_error_margin` | `max(ci_hi - mean, mean - ci_lo)`. |
| `cost` | `AgentCostSummary`, present when any datapoint-run has a `cost.json`. |

- **Splits**: `all_data` is always present, plus one key per named split in the results.
- **`cost`** is per agent and repeated on every split. `total_cost_usd`, the token counts,
  `num_llm_calls` and `by_model` are means per run; `cost_per_datapoint` averages over runs,
  then datapoints; `total_experiment_cost_usd` is the sum over all datapoint-runs;
  `num_datapoints`, `num_runs` and `datapoints_missing_cost` tell partial coverage from a
  free run; `unpriced_calls` and `gateway_priced_calls` are summed.
- **Rounding**: every float, costs included, is rounded to 4 decimals. Read `cost.json` for
  sub-cent costs.
- **`run_datapoint`** adds `datapoints_mean`, `datapoints_ci`, `datapoints_whiskers` (across
  datapoints) and `datapoints`: `{"<name>": mean, "<name>_ci": [lo, hi], "<name>_whiskers": [min, max]}`
  across runs.

### CSVs

Both have one row per loop key, agent, version, run and split, and start with `loop_name`,
`loop_key`, `agent`, `agent_version`, `run` (`run_1`), `split`. Then the first value column,
`total_tags` (non-null value columns in the row), and the remaining value columns.

| File | Value columns |
|---|---|
| `agg_results_on_metrics.csv` | One per evaluator: unweighted mean score in that run. |
| `agg_results_on_datapoints.csv` | One per datapoint: weighted score across its evaluators. |

### Aggregation

| `--aggregation_mode` | Score of one run |
|---|---|
| `evaluator_run` | Mean per evaluator, then the weighted mean across evaluators. |
| `datapoint_run` (default) | Weighted mean across evaluators per datapoint, then the mean across datapoints. |
| `run_datapoint` | As `datapoint_run`, plus the per-datapoint fields above. |

Weights are the agent's `evaluator_weights`; a missing entry weighs 1. Weight 0 keeps the
evaluator running (and spending judge calls) but out of the score. In `evaluator_run`, an
evaluator present on a few datapoints weighs as much as one present on all.

## Fill-mode predicates (`run_utils.py`)

| Predicate | True when |
|---|---|
| `is_datapoint_executed` | `output.json` exists, is non-empty and parses as JSON |
| `is_datapoint_errored` | `results/` exists and some `*.json` there has `agent_output.type == "error"` or cannot be read |
| `is_evaluator_executed` | `results/<evaluator_id>.json` exists and is not errored (with `include_errored=True`: exists) |

| State | `output.json` | `results/` | executed | errored |
|---|---|---|---|---|
| Never run | no | no | no | no |
| Ran, not yet evaluated | yes | no | yes | no |
| Ran and evaluated | yes | yes | yes | no |
| Agent failed, evaluated | if any span was kept | yes | if `output.json` | yes |
| Agent failed, not evaluated | if any span was kept | no | if `output.json` | no |

`should_run_datapoint` and `should_run_evaluator` combine these per mode (table in
[running-the-testbed.md](running-the-testbed.md#run-modes)).

When an evaluator is skipped, `load_cached_evaluator_result` reads its file and returns
`result` (`NumericEvaluationResult(score, details)`), `evaluation_criteria`, `agent_output`
and `split`, with recorded `judge_model` and `judge_prompt` when present; the caller adds
`agent_name`, `agent_version`, `loop_details` and `run_number`.
`details` is rebuilt by `reconstruct_details`: `null` or a string as is; a dict through the
evaluator's Pydantic justification type when it has one, else as a JSON string; anything else
via `str()`. If the file cannot be loaded at that point, the evaluator runs again.

## In-memory results

`run_and_evaluate_all` (`runners/streaming_runner.py`) returns what `ResultAggregator`
consumes: `"<agent>__<version>" → loop_key → evaluator_id → "<datapoint>:run_<N>" →
{result, evaluation_criteria, agent_output, agent_name, agent_version, loop_details,
run_number, split[, judge_model, judge_prompt]}`. The `:run_<N>` suffix is stripped in
`combined_results.json` and the CSVs, which carry the run as `run_number` / `run`.
