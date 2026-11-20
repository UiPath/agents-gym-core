# Creating a new agent

From an empty directory to an agent the testbed (`gym eval`)
loads, runs and scores. Related guides: [evaluators.md](evaluators.md) (evaluator types and
scoring), [running-the-testbed.md](running-the-testbed.md) (every flag),
[prompts.md](prompts.md) (prompt rendering) and [loops.md](loops.md).

## 1. Layout

An agent is a directory `src/agents_gym_core/agents/<name>/`:

| Path | Needed | Purpose |
|---|---|---|
| `agent.yaml` | always | Schemas, prompt paths, tools, eval sets, weights |
| `prompts/system_prompt.md` | tool-loop agents | Instructions; never templated with inputs |
| `prompts/user_prompt.md` | always | Template rendered with each datapoint's inputs |
| `tools.py` | custom tools | `@tool` functions referenced from `agent.yaml` |
| `evaluations/eval-sets/evaluation-set-<name>.json` | always | Datapoints and their criteria |
| `evaluations/evaluators/evaluator-<id>.json` | always | One file per evaluator |
| `evaluations/evaluators/custom/<file>.py` | custom evaluators | `BaseEvaluator` subclasses |
| `versions.yaml` | optional | Named overrides, selected as `<name>:<version>` |
| `init.py` | optional | Data preparation, run by `gym init` (section 4.7) |

- The directory name is the agent name: the registry uses it, and it is what `--agents`
  takes. `name:` in `agent.yaml` is not read.
- Use a valid Python identifier (lowercase, underscores). `_target_` paths import
  `agents_gym_core.agents.<name>.tools`. No `__init__.py` is needed; calculator has none.
- A new agent's files, `evaluations/` included, are tracked by git. Only the six
  public-benchmark agents (aime_2025_agent, hotpotqa, hover, ifbench, livebench_math, papillon)
  generate and gitignore their `evaluations/`.

## 2. Worked example: `word_counter`

One tool, one output field, one deterministic evaluator. Create these files under
`src/agents_gym_core/agents/word_counter/`.

`agent.yaml`:

```yaml
version: base

metadata:
  name: "Word Counter"
  tags:
    - tutorial

agent:
  system_prompt: core::prompts/system_prompt.md
  user_prompt: core::prompts/user_prompt.md

  input_schema:
    fields:
      text:
        type: str
        description: "The text whose words should be counted"

  output_schema:
    fields:
      word_count:
        type: int
        description: "Number of words in the text"

  tools:
    - _target_: agents_gym_core.agents.word_counter.tools.count_words

  evaluation_sets:
    - core::evaluations/eval-sets/evaluation-set-word_counter.json
```

`tools.py`:

```python
from typing import Annotated

from langchain_core.tools import tool


@tool
async def count_words(text: Annotated[str, "Text to count words in"]) -> int:
    """Count the whitespace-separated words in a text."""
    return len(text.split())
```

`prompts/system_prompt.md` (it names the tool and the output field exactly):

```markdown
You count words. Call the `count_words` tool exactly once with the full input text,
then call `end_execution` with `word_count` set to the number the tool returned.
Do not count the words yourself.
```

`prompts/user_prompt.md` (`{text}` is the input field):

```markdown
Text: {text}
```

`evaluations/evaluators/evaluator-word_counter_ExactMatchEvaluator.json`:

```json
{
  "version": "1.0",
  "id": "word_counter_ExactMatchEvaluator",
  "description": "word_count must equal the expected count",
  "evaluatorTypeId": "uipath-exact-match",
  "evaluatorConfig": {
    "name": "ExactMatchEvaluator",
    "defaultEvaluationCriteria": null,
    "targetOutputKey": "word_count",
    "caseSensitive": false,
    "negated": false
  }
}
```

`evaluations/eval-sets/evaluation-set-word_counter.json` (generate fresh ids with
`python -c "import uuid; print(uuid.uuid4())"`):

```json
{
  "version": "1.0",
  "id": "5f0c6a2e-8d1b-4c3a-9e7f-2b6d4a1c8e90",
  "name": "Word Counter Eval Set",
  "evaluatorRefs": ["word_counter_ExactMatchEvaluator"],
  "evaluations": [
    {
      "id": "0a9e3c1d-6b4f-4e2a-8c7d-1f5b9e3a7c20",
      "name": "FourWords",
      "inputs": {"text": "the quick brown fox"},
      "evaluationCriterias": {
        "word_counter_ExactMatchEvaluator": {"expectedOutput": {"word_count": 4}}
      }
    },
    {
      "id": "7d2b8f4a-1c6e-4a9b-b3d5-8e0f2c4a6b13",
      "name": "OneWord",
      "inputs": {"text": "hello"},
      "evaluationCriterias": {
        "word_counter_ExactMatchEvaluator": {"expectedOutput": {"word_count": 1}}
      }
    }
  ]
}
```

What has to agree:

| This | Must equal |
|---|---|
| `count_words` (function name) | The tool name in the system prompt |
| `word_count` (output field) | The field the system prompt asks for, and the key in `expectedOutput` and `targetOutputKey` |
| `text` (input field) | `{text}` in `user_prompt.md` and the key in every datapoint's `inputs` |
| `word_counter_ExactMatchEvaluator` | The evaluator filename, its `id`, `evaluatorRefs`, and each `evaluationCriterias` key |

A custom evaluator for this agent (a tolerance check) is worked through in
[evaluators.md section 7](evaluators.md#7-custom-evaluators).

## 3. Validate and run

**1. Check the eval-set and evaluator files.**

```bash
uv run python -m agents_gym_core.prompt_optimization.check_eval_sets --agent word_counter --check-evaluators
```

Expect `Errors: 0`.

**2. Load the agent without calling a model.** This builds the tools, schemas, prompts and
evaluators exactly as a run does, and shows what the model will see:

```bash
uv run python -c "
from agents_gym_core.hydra_agent import CoreAgentRegistry, EvaluationLoader
from agents_gym_core.loops.nodes import create_messages
reg = CoreAgentRegistry()
a = reg.get_agent('word_counter')
print('tools:', [t.name for t in a.tools])
dp = a.evaluation_set.evaluations[0]
print('user prompt:', create_messages(a, a.input_schema(**dp.inputs))[1].content)
print('evaluators:', list(EvaluationLoader.get_evaluators('word_counter', reg.get_directory_provider())))"
```

Expected: `tools: ['count_words']`, `user prompt: Text: the quick brown fox`,
`evaluators: ['word_counter_ExactMatchEvaluator']`. If this prints the warning
`User prompt template failed (missing key …)`, a `{token}` did not match an input field
(section 4.2).

**3. Run one datapoint.** This calls the model; set up credentials as in the README's
[Configure models](../../README.md#configure-models).

```bash
uv run gym eval \
  --agents word_counter \
  --models gpt-4.1-mini-2025-04-14 \
  --max_datapoints 1 \
  --experiment_name word_counter_smoke
```

- Always pass `--experiment_name`. The default, `experiment`, is shared by every run that omits it.
- The default `--mode fill` reuses completed results and retries failed agent/evaluator runs
  ([run modes](running-the-testbed.md#run-modes)). After editing a prompt or tool, use a new
  experiment name or `--mode overwrite` to evaluate the new configuration.
- `--datapoints FourWords` runs datapoints by exact name. If no name matches, the run logs a
  warning and exits 0 having run nothing.
- Pass `--models` explicitly. The default is `gpt-4o-2024-11-20`.

Where to look, under `experiments/word_counter_smoke/`:

| File | Contents |
|---|---|
| `final_score_per_agent.json` | Headline score per loop and agent (`runs_mean`); 1.0 here when the run succeeds |
| `word_counter/base/basic/<loop_key>/agent_details.json` | The prompts, schemas and tool names the run actually used |
| `.../<loop_key>/FourWords/run_1/output.json` | The trace: every tool call and the final output |
| `.../<loop_key>/FourWords/run_1/results/word_counter_ExactMatchEvaluator.json` | Score, `details`, criteria and the agent output |

## 4. Reference and rules

### 4.1 `agent.yaml` keys

| Key | Notes |
|---|---|
| `version` | `base` in `agent.yaml`. Defaults to `base`. |
| `metadata.tags` | Selectable with `--agents @<tag>`. Every agent here also carries the implicit tag `core`. The tag `conversational` switches the agent to conversational mode (section 4.3). |
| `loop` (top level, not under `agent:`) | Default loop. It replaces `basic` in `--loops`, which is the default there. |
| `init` | Data preparation (section 4.7). |
| `agent.system_prompt` | Optional `core::<path>` or inline instructions; defaults to empty text. DSPy programs can configure instructions per predictor. |
| `agent.user_prompt` | `core::<path>` or inline text. Required. |
| `agent.input_schema.fields` / `agent.output_schema.fields` | `<field>: {type, description, default}` (section 4.4). |
| `agent.tools` | List of `_target_` entries (section 4.3). |
| `agent.evaluation_sets` | Testbed evaluation uses a non-empty list of `core::` paths, merged into one set. |
| `agent.evaluator_weights` | Evaluator id to weight; unlisted evaluators weigh 1, and 0 runs without scoring. |

`core::<path>` resolves relative to the agent's own directory. Use this form for eval-set
files and file-backed prompts. The `system_prompt` and `user_prompt` fields also accept
inline instructions.

### 4.2 Prompts

- **`system_prompt.md` is used verbatim.** Put input substitutions in the user prompt;
  system-prompt JSON examples can use single braces.
- **`user_prompt.md` is rendered with `str.format_map`** over the input fields.
  - A `{token}` that names no input field, or a single-brace JSON example such as
    `{"word_count": 3}`, raises `KeyError`. The whole user prompt is then replaced by
    a generic request carrying the run state as JSON.
  - Double braces escape formatting: `{{text}}` renders as the literal `{text}`.
  - A lone `}`, or `{}` or `{0}`, raises `ValueError` and the datapoint fails.
  - An input that is `null` renders as `None`.
- Name every tool and every output field in the system prompt exactly as registered
  (case-sensitive).

### 4.3 Tools

The name the model sees, and the name that tool-call evaluators match, is the tool object's
`.name`:

| `_target_` points at | Name |
|---|---|
| `@tool` function | The function name: `count_words`, calculator's `add`, claim_history's `claim_history_lookup` |
| `@tool("explicit")` function | `"explicit"` |
| A tool class, called with the entry's other keys as kwargs | Its `name` field; calculator passes `name: "escalation"` to `EscalationTool`, whose default is `escalation_tool` |
| A factory function that returns a tool | The returned tool's name; sql's `create_sql_db_query_tool` returns `sql_db_query` |

Class and factory targets are constructed once with Hydra using the entry's arguments.
The result must be a `BaseTool` or a callable with a docstring and supported signature;
returned callables are wrapped as `StructuredTool` without being called during loading.
Use `@tool` for a function intended to run only when the agent invokes it. An undecorated
`_target_` function is a construction-time factory, not an automatically deferred tool.
Construction completes during loading, and constructor errors are reported at that stage.

Hydra's target restrictions apply to both the factory and recursive nested targets. The
resolved top-level `_target_` import path is captured once; escaped or unresolved
interpolation syntax is rejected. Loading does not modify the original config, including
read-only configs and references to parent fields. Arguments default to native Python
dicts/lists (`_convert_: all`); an explicit `_convert_` or `_recursive_` policy, including
one on a nested factory, is honored. Factory `_partial_: true` is rejected: return the
runtime tool or callable from the factory instead.

Rules:

- Use the registered tool name in the system prompt, including its exact case.
- Register each function tool under `agent.tools` to make it available to the agent.
- A `_target_` resolving to an existing tool (including `@tool`) is not called during
  loading. Fields such as `name:` and `description:` apply to a copy for that agent
  version; the shared tool stays unchanged. Unknown fields raise `TypeError` before
  their values are resolved. Nested `_target_` entries in these field overrides remain
  configuration data, not additional factories.
- The docstring is the tool description the model reads; `Annotated[type, "..."]` describes
  each argument.
- By default an exception raised by a tool fails the datapoint. With `--handle_tool_errors` it
  is returned to the model as an `{"error": ...}` result and the run continues. Either way,
  prefer returning a specific error string the model can act on.
- `end_execution` and `raise_error` are added automatically, except in conversational mode.
  Do not declare them. Refer to them by these names in the prompt.
- `end_execution`'s arguments are the `output_schema` fields; it is how the agent produces its
  output. A missing required field fails the datapoint with `OUTPUT_VALIDATION_ERROR`. Fields
  the schema lacks are dropped, so a prompt that asks for `count` while the schema says
  `word_count` loses the value. The final output holds every schema field, optional ones as
  `null`.
- `raise_error` ends the run with `TERMINATION_LLM_RAISED_ERROR`; evaluators still score that
  error output.
- Use `async def` when the body awaits or is trivially cheap (calculator, word_counter). Put
  blocking calls (sync HTTP, database) in a plain `def` tool, which runs in a worker thread;
  inside an `async def` tool they stall every concurrent run.
- In conversational mode (`metadata.tags: [conversational]`) no flow-control tools are added,
  a non-empty `input` field is sent as the user message in place of `user_prompt.md`, and the
  run ends when the model replies without a tool call.

To check names, print `[t.name for t in a.tools]` (section 3, step 2), or read `tools` in a
run's `agent_details.json`.

### 4.4 Schema field types

| `type` | Python type |
|---|---|
| `str`, `int`, `float`, `bool`, `list`, `dict` | as named |
| `A \| B`, `A \| None` | union of the above |
| `Literal['a', 'b']` | `Literal` |
| `List[str]` / `list[int]` / `list[dict]` (inner `str`, `int`, `float`, `bool` or `dict`) | `list[...]` |
| `List[Literal['a', 'b']]` | `list[Literal[...]]` |

Unrecognized type names resolve to `str`, including inside unions and lists. Use the supported
forms above when declaring field types.

A field with no `default` is required. `type: str | None` alone makes the value nullable but
still required; add `default: null` to make it optional.

### 4.5 `versions.yaml`

```yaml
train:
  version: train            # required: without it, loading raises ValueError
  agent:
    evaluation_sets:
      - core::evaluations/eval-sets/evaluation-set-claim_history_train.json
```

Select with `--agents claim_history:train`, or `claim_history:*` for all versions. Each
entry is merged over `agent.yaml` with OmegaConf: mappings merge key by key, and lists
(`tools`, `evaluation_sets`) are replaced whole. Calculator's `missing_add_tool` version
therefore re-lists every tool it keeps.

### 4.6 Eval sets and splits

- Name eval-set files `evaluation-set-*.json`; the checker only finds that pattern.
- A `_train`, `_val`, `_test` or `_dev` filename suffix (or `-train` etc.), or `__<label>`, sets
  the split label used by `--splits` and in the per-split scores. claim_history's `train` and
  `test` versions point at `evaluation-set-claim_history_train.json` and `_test.json`.
- `inputs` keys must match `input_schema` field names, case included.
- An evaluator runs on a datapoint only if its id is a key of that datapoint's
  `evaluationCriterias`. Criteria formats per type are in
  [evaluators.md section 6](evaluators.md#6-built-in-types).

### 4.7 `init:` (generated data)

```yaml
init:
  script: core::init.py       # must define init(agent_dir: Path, params: dict) -> None
  data_dirs:                  # directories the script materialises; injected as params["data_dirs"]
    - evaluations/eval-sets
  params:                     # forwarded to init()
    seed: 0
```

- An agent with an `init:` block fails to load with `AgentNotInitialisedError` until its init
  has run. `UIPATH_GYM_SKIP_INIT=true` skips the check.
- Run it with `gym init <agent>`, `gym init <agent> --force`, or the testbed's `--reinit`.
  `gym init --list` shows each agent's status.
- Success writes a gitignored `.gym_init_done` marker holding a hash of `params`. Changing
  `params` invalidates it.
- Omit `init:` when every file is committed, as for word_counter, calculator and sql.

### 4.8 Loops

`--loops <name>:<version>`; a name with no version means `auto`.

| Loop | Behaviour |
|---|---|
| `basic` (default) | Tool-calling loop with tool choice `auto`: the model may take tool-less turns |
| `basic:any` | Tool choice `any`: a tool call on every turn |
| `basic:auto-graceful`, `basic:any-graceful` | Force `end_execution` at the call limit rather than failing with `TERMINATION_MAX_ITERATIONS` |
| `summarizing`, `summarizing:any` | Summarize message history at configured token thresholds |

Planning, validation, deep-agents and DSPy programs: [loops.md](loops.md).

The call budget is `UIPATH_AGENTS_GYM_LOOP_RECURSION_LIMIT` model calls (default 20).
Graceful variants and `summarizing` request `end_execution` at that limit; the other
`BasicLoop`-based loops raise `TERMINATION_MAX_ITERATIONS`. Tool loops raise
`THINKING_LIMIT_EXCEEDED` when the model keeps answering without tool calls past
`UIPATH_AGENTS_GYM_NO_TOOL_MESSAGES_LIMIT` (default 5).

## 5. Debugging

| Symptom | Check |
|---|---|
| `Agent '<name>' not found in CoreDirectoryProvider` | `src/agents_gym_core/agents/<name>/agent.yaml` exists; the name has no hyphens |
| `InstantiationException: Error locating target '...'` | The import path; that `tools.py` imports cleanly (`python -c "from agents_gym_core.agents.<name>.tools import <fn>"`) |
| Model never calls a tool | Prompt tool names against `[t.name for t in a.tools]`; the tool is listed under `agent.tools` |
| Model behaves as if it saw no input | Step 2 of section 3 prints `User prompt template failed (missing key …)` and the trace's user prompt is a generic request carrying the run state as JSON (bad `{token}` or single-brace JSON), or `{{field}}` was written for `{field}` |
| Instructions ignored entirely | `system_prompt` lacks `core::` and was sent as literal text |
| `OUTPUT_VALIDATION_ERROR` | A required output field was not passed to `end_execution`; align prompt and schema names |
| `TERMINATION_MAX_ITERATIONS` | The model is looping, often retrying a failing tool; make the tool return a clear error string and give the prompt a path to `end_execution` |
| Output field has the wrong type | Its `type` fell back to `str` (section 4.4) |
| An evaluator scores nothing, or 0.0 | See [evaluators.md section 9](evaluators.md#9-debugging) |
| Edits have no effect | `fill` mode reused the old run; use `--mode overwrite` or a new `--experiment_name` |

Before committing: the checker passes with `--check-evaluators`; step 2 prints the
expected tool names and a rendered user prompt; every `id` in the eval set is a fresh UUID;
and a one-datapoint testbed run scores as expected.
