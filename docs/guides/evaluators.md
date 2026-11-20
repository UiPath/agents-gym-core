# Evaluators

How the testbed scores an agent: eval-set and evaluator file formats, built-in and custom
evaluators, weights, aggregation and error reporting.

Built-in evaluator classes come from the UiPath SDK (`uipath.eval.evaluators`). This repo owns
the JSON that configures them and any custom evaluator code, under each agent's
`src/agents_gym_core/agents/<agent>/evaluations/`. To build an agent around them, see
[new-agent.md](new-agent.md); for the on-disk result files, see
[run-output-formats.md](run-output-formats.md).

## 1. How a datapoint is scored

1. The testbed loads every file in `agent.evaluation_sets` and merges them into one set.
2. It builds every `evaluator-*.json` in the agent's `evaluations/evaluators/`
   (`EvaluationLoader.get_evaluators` in `hydra_agent/evaluation_loader.py`). The whole
   directory is always loaded. An `agent.evaluators:` list in `agent.yaml` is never read.
   Duplicate `id`s fail an assertion.
3. For each datapoint, an evaluator runs only if its `id` is a key of that datapoint's
   `evaluationCriterias`. An unrecognized id does not select an evaluator.
   `evaluatorRefs` is not consulted at run time; only the checker reads it.
4. A `null` criteria value means "use the evaluator's `defaultEvaluationCriteria`".
5. If the agent output is `{"type": "error", "code": "GENERAL_EXECUTION_ERROR", ...}` (an
   unexpected exception during the run), every evaluator is recorded as `0.0` without being
   called. Other agent errors, such as `OUTPUT_VALIDATION_ERROR`, `TERMINATION_MAX_ITERATIONS`
   and `TERMINATION_LLM_RAISED_ERROR` (the agent called `raise_error`), are passed to the
   evaluators as the output and scored normally.
6. Each result is written to
   `experiments/<experiment>/<agent>/<version>/<loop>/<loop_key>/<datapoint>/run_<n>/results/<evaluator_id>.json`
   with `score`, `details`, `evaluation_criteria`, `agent_output`, `split`, plus
   `judge_model` and `judge_prompt` for LLM judges.

## 2. How evaluator errors are recorded

| Problem | What gets recorded |
|---|---|
| Exception inside `evaluate()` | Recorded as `0.0` with `details` starting `Exception thrown by evaluator:`. The CLI reports the evaluator exception and exits with code 1; `fill` retries it on the saved agent output. |
| Criteria fail validation (wrong key, wrong type), or `null` criteria with no default | The affected datapoint run is listed as failed and excluded from aggregate scores; the CLI exits with code 1. |
| Unknown `evaluatorTypeId` with no `evaluatorSchema`, or an `evaluatorSchema` file that cannot be loaded | Evaluators are built after each agent run, so every affected datapoint run is listed as failed and not scored; the CLI exits with code 1. |
| Misspelled key in `evaluatorConfig` | Ignored; the default applies (`targetOutputKy` leaves `targetOutputKey` at `"*"`). |

`fill` mode (the default) runs an evaluator again when its result records an exception,
reusing `output.json`, so a `0.0` from a judge outage is retried on the next `fill` run;
`fill_skip_errors` keeps it. `--mode fill_rerun_evals` reruns every evaluator.

| `--mode` | Reruns the agent | Reruns evaluators |
|---|---|---|
| `overwrite` | always | always |
| `fill` (default) | if missing or errored | if missing or errored |
| `fill_rerun_evals` | if missing or errored | always |
| `fill_skip_errors` | if missing | if missing |
| `fill_skip_run_errors` | if missing, or the model endpoint failed | if missing, unreadable, or the evaluator raised; not for any other failed agent run |

A datapoint is missing when `output.json` is absent, empty or not valid JSON. It is errored
when any of its result files holds an error dict as `agent_output`, so an agent error with
no result files yet does not count as errored.

## 3. Weights and aggregation

`agent.evaluator_weights` maps evaluator id to weight. Unlisted evaluators weigh 1.

```yaml
agent:
  evaluator_weights:
    calculator_ExactMatchEvaluator: 1
    calculator_LLMJudgeOutputEvaluator: 0   # runs and is saved, but excluded from the score
```

A weight of 0 still runs the evaluator, and still pays for its judge calls. It is left out of
both the numerator and the denominator; if every weight is 0 the aggregate is `0.0`.
`evaluatorConfigs[].weight` inside an eval-set file is parsed by the SDK but ignored by the
testbed. Use `evaluator_weights`.

| `--aggregation_mode` | Score per run |
|---|---|
| `evaluator_run` | Mean of each evaluator over the datapoints it scored, then the weighted mean across evaluators |
| `datapoint_run` (default) | Weighted mean over each datapoint's evaluators, then the mean across datapoints |
| `run_datapoint` | As `datapoint_run`, plus per-datapoint scores across runs |

Runs are then combined (`runs_mean`, `runs_ci`) in
`experiments/<experiment>/final_score_per_agent.json`, keyed by loop key, then
`<agent>__<version>`, then split: `all_data` plus each named split.

## 4. Eval-set file

`evaluations/eval-sets/evaluation-set-<name>.json`, from calculator (trimmed):

```json
{
  "version": "1.0",
  "id": "32269a8e-d522-49c6-8c81-c3a774655183",
  "name": "Calculator Eval Set",
  "evaluatorRefs": ["calculator_ExactMatchEvaluator", "calculator_ToolCallCountEvaluator"],
  "evaluations": [
    {
      "id": "f0096037-3d73-436c-baaa-271931163295",
      "name": "TestSimpleAddition",
      "inputs": {"expression": "how much is 2 + 5"},
      "evaluationCriterias": {
        "calculator_ExactMatchEvaluator": {"expectedOutput": {"answer": 7}},
        "calculator_ToolCallCountEvaluator": {"toolCallsCount": {"add": ["=", 1], "multiply": ["=", 0]}}
      }
    }
  ]
}
```

| Field | Rule |
|---|---|
| `version` | `"1.0"`. Without it the SDK parses the file as the legacy format. |
| `id`, `name` | Required. |
| `evaluatorRefs` | Unused at run time. The checker requires it to be non-empty, use semantic ids rather than UUIDs, list every criteria key, have every ref used by at least one evaluation and, with `--check-evaluators`, have an evaluator file for each ref. |
| `evaluations[].id`, `.name` | Required. `name` is the datapoint key used for result directories and `--datapoints`; `/` becomes `_` in directory names, and names must stay unique after that, ignoring letter case. |
| `evaluations[].inputs` | Keys must equal the agent's `input_schema` field names. |
| `evaluations[].evaluationCriterias` | Required. Keys are evaluator ids; values are criteria objects or `null`. |
| `evaluations[].inputMockingStrategy.prompt` | Optional. Passed to simulation tools and to trajectory judges as `{{SimulationInstructions}}`. |

Older files (calculator, for one) also carry `expectedOutput`, `simulationInstructions`,
`simulateTools`, `toolsToSimulate`, `evalSetId`, `createdAt` and `source` on each item. No
evaluator reads them; expectations live only in `evaluationCriterias`.

The filename sets the split label: a `_train`, `_val`, `_test` or `_dev` suffix (or `-train`
etc.), or `__<label>` for a custom split (`evaluation-set-claim_history_train.json` is `train`).
Anything else is `all_data`. Splits drive `--splits` and the per-split scores.

## 5. Evaluator file

`evaluations/evaluators/evaluator-<id>.json`, from calculator:

```json
{
  "version": "1.0",
  "id": "calculator_ExactMatchEvaluator",
  "description": "ExactMatchEvaluator for calculator agent",
  "evaluatorTypeId": "uipath-exact-match",
  "evaluatorConfig": {
    "name": "ExactMatchEvaluator",
    "defaultEvaluationCriteria": null,
    "targetOutputKey": "*",
    "caseSensitive": false,
    "negated": false
  }
}
```

- `version` must be `"1.0"`. Anything else is parsed as a legacy evaluator and fails to load.
- The filename must be `evaluator-<id>.json` (checker). Other filenames are not loaded.
- `id` is the key used in `evaluationCriterias`. Convention: `<agent>_<Name>`.
- No `createdAt` or `updatedAt` (checker).
- `evaluatorConfig` keys may be camelCase or snake_case (`targetOutputKey` or
  `target_output_key`). Unknown keys are ignored.
- `defaultEvaluationCriteria` applies when a datapoint's criteria is `null`; sql's tool-call
  count and order evaluators work this way.

## 6. Built-in types

Criteria keys are the ones `check_eval_sets.py` requires (`type_to_required_field`). Write
them in camelCase: the SDK also accepts snake_case, but the checker does not.

| `evaluatorTypeId` | Criteria key | Score | Config (defaults) | Used by (calculator, enterprise agents) |
|---|---|---|---|---|
| `uipath-exact-match` | `expectedOutput` | 0 or 1 | `targetOutputKey` `"*"`, `caseSensitive` false, `negated` false | calculator, customer_support_triage_agent |
| `uipath-contains` | `searchText` | 0 or 1 | `targetOutputKey`, `caseSensitive` false, `negated` false | product_questions_agent |
| `uipath-json-similarity` | `expectedOutput` | 0 to 1 | `targetOutputKey` | none |
| `uipath-llm-judge-output-semantic-similarity` | `expectedOutput` | 0 to 1 | `prompt`, `model` `""`, `temperature` 0.0, `maxTokens` null, `targetOutputKey` | calculator and the five enterprise agents |
| `uipath-llm-judge-output-strict-json-similarity` | `expectedOutput` | 0 to 1 | as above | none |
| `uipath-llm-judge-trajectory-similarity` | `expectedAgentBehavior` | 0 to 1 | `prompt`, `model`, `temperature`, `maxTokens` | none |
| `uipath-llm-judge-trajectory-simulation` | `expectedAgentBehavior` | 0 to 1 | as above | none |
| `uipath-tool-call-count` | `toolCallsCount` | matched / expected | `strict` false | calculator, claim_history, customer_support_triage_agent, meeting_management, sql |
| `uipath-tool-call-order` | `toolCallsOrder` | LCS / expected | `strict` false | calculator, sql |
| `uipath-tool-call-args` | `toolCalls` | matched / expected | `strict` false, `subset` false | calculator, claim_history |
| `uipath-tool-call-output` | `toolOutputs` | matched / expected | `strict` false | sql |

For judge configurations, choose a concrete variant from the table. The identifiers
`uipath-llm-judge-output` and `uipath-llm-judge-trajectory` name SDK base classes.

The SDK also provides `uipath-binary-classification` and `uipath-multiclass-classification`.
Core averages their per-datapoint 0 or 1 scores. Call the SDK's `reduce_scores` separately
for their classification-specific aggregation.

### Output extraction (`targetOutputKey`)

Applies to exact-match, contains, JSON similarity and the output judges.

- `"*"` takes the whole output dict. An output produced by `end_execution` contains every
  `output_schema` field, optional ones as `null`, so an `"*"` `expectedOutput` must list them all.
- A field name, a dot/bracket path (`"a.b[0].c"`) or a list of keys selects part of it. The same
  path is read from `expectedOutput`, so it must exist on both sides. Missing on either side
  records `0.0` with an exception detail.
- Ints and floats are converted to float, recursively, on both sides. Booleans are left alone.

### `uipath-exact-match`

- If either side is a string, it compares the `str()` forms, lowercased unless `caseSensitive`.
  `7` against `"7"` therefore fails (`"7.0"` against `"7"`); keep the JSON types equal.
- Otherwise it uses Python `==`: dict key order does not matter, list order does.
- `expectedOutput` must be an object or a string. A bare number fails criteria validation;
  wrap it as `{"answer": 7}`.

### `uipath-contains`

Scores `searchText in str(extracted output)`, case-insensitive by default. With `"*"` it
searches the Python repr of the whole dict, key names included, with numbers as floats
(`3` becomes `3.0`). Point `targetOutputKey` at the text field; product_questions_agent uses
`"target_output_key": "FinalResponse"`.

### `uipath-json-similarity`

Walks the expected value. Numbers score `1 - |e - a| / |e|`, clamped to 0..1 (an expected 0
needs an actual 0). Strings score `1 - levenshtein / max_len`, case-sensitive. Booleans score
1 or 0. A type mismatch scores 0 for that subtree. Only expected dict keys count. Lists compare
by position: missing elements count as misses, extra actual elements are ignored. The score
is matched leaves over total leaves; an empty expected value scores 1.0.

### LLM judges

- Output judges replace `{{ExpectedOutput}}` and `{{ActualOutput}}` with the `str()` of the
  extracted values. Trajectory judges replace `{{ExpectedAgentBehavior}}` (from the criteria),
  `{{AgentRunHistory}}` (the trace rendered as text), `{{UserOrSyntheticInput}}`
  (`str(inputs)`) and `{{SimulationInstructions}}`. Any other `{{...}}`, `{{AgentInput}}`
  included, stays literal.
- A missing primary placeholder is appended to the end of the prompt. The checker fails it
  anyway: output judges need `{{ExpectedOutput}}` and `{{ActualOutput}}`, trajectory judges
  need `{{ExpectedAgentBehavior}}` and `{{AgentRunHistory}}`.
- With no `prompt`, the SDK default applies; calculator's prompt is the semantic-similarity
  default verbatim. The strict-JSON judge's system prompt applies a key-by-key penalty rubric.
- The judge must answer through a forced `submit_evaluation` tool call carrying
  `justification` and a 0 to 100 `score`. The result is `clamp(round(score / 100, 2))`.
- A trajectory judge reads only `expectedAgentBehavior` inside the criteria. An item-level
  `expectedAgentBehavior` field is ignored.
- The checker warns when a prompt uses `{{ExpectedAgentBehavior}}` and a datapoint's criteria
  lack it.

### Judge model routing

The testbed calls `configure_model_resolution()` at startup (`TestbedRunnerCore.__init__`),
which installs `route_llm_judges_through_registry()` from `models/judge_routing.py`. Judge
calls then go through `get_chat_model` on the run's `--model_backend`, with the same
credentials as the agent.

- `model` is passed straight to `get_chat_model`. The SDK default `""` does not mean "tenant
  default" here, so always name a registered model. The evaluator JSONs in this repo mostly
  use `gpt-4o-2024-11-20`.
- `temperature` comes from the JSON. `maxTokens: null` becomes 4096, or 8000 when the JSON model
  name contains `claude-haiku-4-5` or `claude-sonnet-4-5`.
- `gym uipath eval` shells out to `uipath eval`, which does not install this routing; its judges use
  the SDK's own UiPath LLM service.

### Tool-call evaluators

- Calls are read from the trace's tool spans. `end_execution` and `raise_error` are not
  counted: a calculator run that called `add` once records `{'add': 1}`.
- Names are case-sensitive. They are compared after the runtime's sanitisation (whitespace
  becomes `_`, characters outside `[A-Za-z0-9_-]` are dropped). Use the tool's registered name.
- If the agent made no tool calls, any non-empty criteria scores 0.0, except count criteria
  whose every entry expects zero calls (`["=", 0]` or `["==", 0]`, e.g. `{"multiply": ["=", 0]}`),
  which score 1.0. Empty criteria with no calls scores 1.0.

| Type | Criteria | Score | `strict: true` |
|---|---|---|---|
| count | `{"<tool>": ["<op>", n]}`, op one of `= == != > < >= <=` | Fraction of entries satisfied. Unlisted tools are ignored; an uncalled tool counts 0. | 0.0 at the first failed entry |
| order | Ordered list of names, repeats allowed | Longest common subsequence / expected length; extra calls are tolerated | 1.0 only for the identical sequence |
| args | `[{"name": ..., "args": {...}}]` | Each expected call takes the first unused actual call of that name whose args contain every expected key with an `==` value. Extra actual args and call order are ignored. | 1.0 only if every call matched |
| output | `[{"name": ..., "output": "..."}]` | Exact, case-sensitive string match on the tool's returned content (copy it from `output.json`) | 0.0 at the first same-name mismatch or missing call |

`subset: true` and `subset: false` behave identically for args: both check only the expected
keys. Equality is Python's, so `7 == 7.0` but `"7" != 7`.

## 7. Custom evaluators

A custom evaluator has three parts: a class in `evaluations/evaluators/custom/<file>.py`, an
evaluator JSON whose `evaluatorSchema` points at it, and criteria entries in the eval sets.
Real ones: claim_history (two deterministic checks), hover (retrieval accuracy), papillon
(an LLM judge of quality and leakage). The example below belongs to the word_counter agent from
[new-agent.md](new-agent.md).

`evaluations/evaluators/custom/word_counter_evaluators.py`:

```python
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.base_evaluator import (
    BaseEvaluationCriteria,
    BaseEvaluatorConfig,
    BaseEvaluatorJustification,
)
from uipath.eval.models import EvaluationResult, NumericEvaluationResult, WorkloadExecution


class CountToleranceCriteria(BaseEvaluationCriteria):
    expected_count: int  # no default, so required; JSON key "expectedCount" or "expected_count"


class CountToleranceConfig(BaseEvaluatorConfig[CountToleranceCriteria]):
    name: str = "CountToleranceEvaluator"
    tolerance: int = 0  # set per evaluator in evaluatorConfig


class CountToleranceEvaluator(
    BaseEvaluator[CountToleranceCriteria, CountToleranceConfig, BaseEvaluatorJustification]
):
    @classmethod
    def get_evaluator_id(cls) -> str:
        return "CountToleranceEvaluator"

    async def evaluate(
        self, workload_execution: WorkloadExecution, evaluation_criteria: CountToleranceCriteria
    ) -> EvaluationResult:
        output = workload_execution.workload_output
        actual = output.get("word_count") if isinstance(output, dict) else None
        expected = evaluation_criteria.expected_count
        ok = isinstance(actual, int) and abs(actual - expected) <= self.evaluator_config.tolerance
        return NumericEvaluationResult(
            score=float(ok),
            details=BaseEvaluatorJustification(expected=str(expected), actual=str(actual)),
        )
```

`evaluations/evaluators/evaluator-word_counter_CountToleranceEvaluator.json`:

```json
{
  "version": "1.0",
  "id": "word_counter_CountToleranceEvaluator",
  "description": "word_count within tolerance of the expected count",
  "evaluatorTypeId": "CountToleranceEvaluator",
  "evaluatorSchema": "file://word_counter_evaluators.py:CountToleranceEvaluator",
  "evaluatorConfig": {
    "name": "CountToleranceEvaluator",
    "defaultEvaluationCriteria": null,
    "tolerance": 1
  }
}
```

Criteria in the eval set: `"word_counter_CountToleranceEvaluator": {"expectedCount": 4}`.
Add the id to `evaluatorRefs` too.

- Subclass `BaseEvaluator[Criteria, Config, Justification]`, where the justification type is
  `str` or a `BaseEvaluatorJustification` subclass. Implement `get_evaluator_id()` and
  `async evaluate()`, and return `NumericEvaluationResult(score=<0..1>, details=...)`.
- The `file://<file>.py:<Class>` path is tried as given (absolute, or relative to the working
  directory), then under `evaluations/evaluators/`, then under `evaluations/evaluators/custom/`.
  Keep the file in `custom/`: the checker looks only there for the criteria class, and if it
  cannot find it, it requires `expectedOutput` instead.
- Set `evaluatorTypeId` to the `get_evaluator_id()` value. This is a convention only: the
  factory loads the class from `evaluatorSchema` and does not compare the two.
- The file is executed as a standalone module, so relative imports fail. Import from the
  package instead; claim_history imports `ClaimHistoryLookupOutput` from
  `agents_gym_core.agents.claim_history.tools`.
- Criteria fields use camelCase aliases, and both spellings are accepted. The checker treats a
  field as required when it has no default: a bare annotation (including `x: str | None`),
  `Field()` or `Field(...)`. A field with `= <value>`, `Field(default=<value>)` or
  `Field(default_factory=list)` is optional.
- Read config fields through `self.evaluator_config`.
- An exception inside `evaluate()` is recorded as `0.0` (section 2). Handle non-dict outputs
  and agent error dicts (`{"type": "error", ...}`) explicitly.
- `workload_execution` carries `agent_input`, `workload_output`, `workload_trace` (OTel spans)
  and `simulation_instructions`. For trajectory checks, use `extract_tool_calls`,
  `extract_tool_calls_names` or `extract_tool_calls_outputs` from
  `uipath.eval._helpers.evaluators_helpers`.

## 8. Validate before running

```bash
uv run python -m agents_gym_core.prompt_optimization.check_eval_sets --agent calculator --check-evaluators
```

- `--check-evaluators` includes evaluator-file rules and verifies referenced evaluators on disk.
- The checker validates eval-set structure and evaluator file fields. Build the evaluators to
  verify their type identifiers and `evaluatorSchema` files, and check config keys, output field
  names and tool names against the agent definition:

```bash
uv run python -c "
from agents_gym_core.hydra_agent import CoreAgentRegistry, EvaluationLoader
print(list(EvaluationLoader.get_evaluators('calculator', CoreAgentRegistry().get_directory_provider())))"
```

## 9. Debugging

| Symptom | Likely cause |
|---|---|
| Every evaluator is 0.0 on a datapoint | `agent_output` is a `GENERAL_EXECUTION_ERROR`; read its `details` |
| An evaluator has no result file | Its id is not a key in that datapoint's `evaluationCriterias`, or its criteria failed validation (look for `gather_limited: task <N> failed`) |
| Judge is always 0.0 | `details` says `Exception thrown by evaluator:`. Fix the model name or credentials, then run again; `fill` reruns the evaluators that raised. |
| Exact match is 0.0 on a correct answer | `7` against `"7"`, an `"*"` key with extra or `null` fields, or list order |
| Tool evaluators are 0.0 | Tool name differs from the registered name, or the agent made no tool calls |
| Old scores persist after fixing an evaluator | `fill` reuses result files; use `--mode fill_rerun_evals` |
| `ValueError: Evaluator file not found` | `evaluatorSchema` path; see section 7 |
| `FileNotFoundError: Evaluation set not found` | The path after `core::` is relative to the agent's own directory |
