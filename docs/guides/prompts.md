# Prompts

How an agent's prompts are loaded and rendered, how its tools and output schema reach the
model, and how its output is extracted. Which loop does what is in [loops.md](loops.md).

## Loading

`CoreAgentRegistry._build_agent` (`hydra_agent/core_agent_registry.py`) loads
`agent.system_prompt`, `agent.user_prompt` and every `agent.signature_prompts` value through
`PromptLoader.load_prompt_from_file` (`utils/prompt_loader.py`):

| Value | Result |
|---|---|
| `"core::prompts/system_prompt.md"` | The file `src/agents_gym_core/agents/<agent>/prompts/system_prompt.md`, stripped of leading/trailing whitespace |
| Anything else | Used verbatim as the prompt text, e.g. calculator's `worst_sys_prompt: "You are a calculator."` |

- File references use the exact `core::` prefix; other strings are inline prompt text.
- A `core::` file that does not exist raises `FileNotFoundError` when the agent loads.
- `user_prompt` is required. `system_prompt` may be omitted (it becomes `""`), as in hover,
  whose DSPy program carries its own instructions.
- `signature_demos` values must be `core::` paths to JSON files (`{"demos": [...]}` or a bare list).

### Optimizer MCP file access

The optimizer's MCP file helpers resolve relative paths within the selected agent. Prompt
reads, previews and updates share this containment check, as do configuration, evaluator,
eval-set and tool-data lookups and fallback file listings. Absolute paths, parent traversal,
Windows-style paths, NUL bytes and symlinks escaping the relevant root are rejected.
Agent-directory aliases must stay inside their agent registry; eval-set file aliases must
stay inside the resolved eval-set directory. Contained symlink aliases remain supported.

Core's 20 tools support inspection, evaluation, prompt editing and optimizer logging;
configuration, evaluators and eval sets are read-only. Optimization runs enforce pinned-version
split restrictions and started-experiment protections. Path containment assumes a trusted
workspace with a stable filesystem; process isolation and protection against concurrent
filesystem replacement require host-level controls.

## Rendering

What each loop does with the two prompts:

| Loop | `system_prompt` | `user_prompt` |
|---|---|---|
| `basic`, `planning`, `validation`, `summarizing` | Verbatim | `format_map(inputs)` |
| `deep-agents` | Verbatim, plus an `end_execution` instruction | `format_map(inputs)`; on a missing key, `json.dumps(inputs)` |
| `dspy-program` | Signature instructions (see [DSPy programs](#dspy-programs)) | Inputs are passed directly to the DSPy program |

### ReAct loops

`create_messages` (`loops/nodes.py`) builds the first two messages:

```python
[SystemMessage(content=scenario.system_prompt),                 # never templated
 HumanMessage(content=scenario.user_prompt.format_map(inputs))]  # templated
```

**`system_prompt.md` is used verbatim.** A `{expression}` placed in calculator's system prompt
would reach the model as the literal text `{expression}`; `{{expression}}` arrives with both
braces. Put every input reference in `user_prompt.md`.

**`user_prompt.md` goes through `str.format_map`** with the input as a dict. Calculator's
`user_prompt.md` is `Calculate the result of: {expression}.`

| In `user_prompt.md` | The model sees |
|---|---|
| `{expression}` | The input value |
| `{{expression}}` | The literal `{expression}`, with no substitution |
| `{query}` (no such input field) | The whole template is discarded, replaced by a generic request carrying the run state as JSON (`{"expression":"2+3","messages":[],"inner_state":{...}}`) |
| An unescaped JSON example, `{"answer": 1}` | The same fallback: `"answer"` is parsed as a field name |
| `{{"answer": 1}}` | `{"answer": 1}` |
| A lone `{` or `}`, or `{0}` | `ValueError`: only `KeyError` is caught, so the run fails at `init` |
| `{expression[0]}`, `{expression.upper}` | Evaluated: indexing and attribute access work |

Calculator's `wrong_param_input_schema` (`{query}`) and `extra_param_schema` (`{rules}`) versions
demonstrate this fallback: loading one without a model as in step 2 of
[Validate and run](new-agent.md#3-validate-and-run)
(`get_agent("calculator", "wrong_param_input_schema")`) prints the warning
`User prompt template failed (missing key …)`, and a run's trace shows the run state as JSON in
place of the template.

### DSPy programs

`dspy-program` passes input fields directly to the program. Its prompt configuration uses
predictor instructions: `signature_prompts[predictor]` if set; else
`system_prompt` if non-empty; else the program's own instructions.

- **`signature_prompts` beats `system_prompt`.** Once a predictor has a signature prompt,
  `system_prompt` is not prepended to it. A non-empty `system_prompt` replaces the instructions
  of every predictor that lacks one.
- Each predictor uses its matching `signature_prompts` entry. Predictors without a matching
  entry use `system_prompt` or their program instructions. Keys are predictor names from
  `program.named_predictors()`, with or without a trailing `.predict`. HoVer's are
  `summarize1.predict`, `summarize2.predict`, `create_query_hop2.predict` and
  `create_query_hop3.predict`, matched by `summarize1` and so on.
- HoVer and PAPILLON configure each predictor through `signature_prompts`. AIME uses
  `system_prompt.md` as the instruction of its single predictor, `predict`.

### Conversational agents

Add `conversational` to `metadata.tags` to select conversational execution
(`Agent.is_conversational`).

- If the input has a non-empty `input` field, it becomes the user message verbatim and
  `user_prompt.md` is ignored; otherwise `user_prompt.md` is rendered as usual.
- `end_execution` and `raise_error` are not added. The run ends at the first AI message without
  tool calls, with no iteration limit.
- The output is the run's new messages under `uipath__agent_response_messages`. No other output
  field is ever filled, so every other field must be optional or `terminate` raises
  `OUTPUT_VALIDATION_ERROR`. If the schema does not declare `uipath__agent_response_messages`,
  the output is empty.
- Use the basic loop for conversational agents; validation loops use structured termination tools.

## Tools as the model sees them

In ReAct loops the LLM node binds, before every call, the agent's tools plus `end_execution` and
`raise_error` (unless the agent is conversational), and `manage_memory` and `search_memory` when
built with `memory_tools=True`.

| Tool kind | Name the model sees | Description |
|---|---|---|
| `@tool` function (calculator `add`, `multiply`; claim_history `claim_history_lookup`) | The function name | The docstring; parameter descriptions from `Annotated[type, "..."]` |
| Class tool, e.g. `agents_gym_core.common_tools.EscalationTool` | Its `name:` from YAML (calculator: `escalation`; default `escalation_tool`) | `EscalationTool` prepends a fixed "This is an escalation…" paragraph to the YAML `description` |

- **Names are matched exactly, case included.** Refer to each tool in the system prompt by its
  bound name.
- Decorate function tools with `@tool`. Hydra calls an undecorated `_target_` function while
  loading, so one with required parameters fails with Hydra's
  `InstantiationException: Error in call to target '...'`.
- **YAML overrides on a `@tool` target apply to a copy of the tool.** `ToolInstantiator` copies
  the decorated object with the overrides set, so `description:` under
  `_target_: agents_gym_core.agents.calculator.tools.add` changes what the model sees in that
  version only. A key that is not a field of the tool raises `TypeError`. Calculator's
  `bad_description_add_tool`, `bad_description_mul_tool` and `swap_description_add_and_mul_tool`
  versions use this to change the `add` and `multiply` descriptions.

## Output: `end_execution`

ReAct loops do not parse the model's text. The run's output is the arguments of the model's
`end_execution` call. `create_end_execution_tool` uses the agent's `output_schema` model as the
tool's parameter schema, so the model sees field names, types and descriptions only there; they
are not added to the system prompt. Calculator's `end_execution` takes
`{"answer": {"type": "integer", "description": "The calculated result"}}`, with `answer` required.

`terminate` (uipath_langchain `create_terminate_node`) validates the arguments with
`output_schema.model_validate(args).model_dump()`:

| The model… | Result |
|---|---|
| Passes a field that is not in `output_schema` | Schemas built from `agent.yaml` ignore extra fields |
| Uses a different field name (`appeal_tips` for claim_history's `AppealTips`) | Treated as an extra field; required fields are validated by their exact names |
| Omits a required field | `AgentRuntimeError` `OUTPUT_VALIDATION_ERROR`; the datapoint fails |
| Omits an optional field (`default: null`) | The default |
| Calls `raise_error(message, details)` | `AgentRuntimeError` `TERMINATION_LLM_RAISED_ERROR` |
| Calls `end_execution` together with a regular tool | `end_execution` is dropped and the run continues |

- **Field names must match `output_schema` exactly**, case included: claim_history uses
  `AppealTips` and `ErrorMessage`, aime_2025_agent `reasoning` and `answer`. Name each field in the
  system prompt and say when to call `end_execution`.
- **Required in the schema means required in the call.** A field is optional only with a
  `default` (`default: null` also makes its type `T | None`). claim_history declares both
  `AppealTips` and `ErrorMessage` without defaults, so both are required in `end_execution`.
  When nothing matches, pass `AppealTips: []` together with the error message.

Other loops: `dspy-program` reads prediction attributes, and `deep-agents` reads the
`end_execution` tool message.

## Versions (`versions.yaml`)

Each top-level key is a version, selected with `--agents <agent>:<version>`. Its value is merged
onto `agent.yaml` with `OmegaConf.merge` and may add keys the base lacks (e.g. `signature_demos`).

```yaml
worst_sys_prompt:                  # calculator
  version: worst_sys_prompt
  agent:
    system_prompt: "You are a calculator."
```

- **Lists are replaced, not merged.** A version's `tools:` is the complete tool list, which is
  why calculator's `missing_add_tool` repeats `multiply` and `escalation`. Mappings merge key by
  key, so a version can override one `signature_prompts` entry and keep the rest.
- **Every non-base entry must set `version:`.** If the merged value is still `base`, loading
  raises `ValueError`. The value is not checked against the key, and it names the output
  directory, so set it to the key; two entries with one value write to the same place.
- `base` always means `agent.yaml` alone.

The paper agents use the released GPT-5 ReAPO prompts by default. Select `baseline*` versions
for the original benchmark prompts. The calculator example uses its original prompts and
diagnostic variants.

Patterns in this repo:

| Pattern | Examples |
|---|---|
| Eval-split selectors (override only `evaluation_sets`) | `claim_history:train`, `claim_history:test`, `hover:train` / `val` / `test` |
| Prompt variants | `claim_history:baseline`, `claim_history:reapo__gpt-5`, `aime_2025_agent:reapo__gpt-5-test` |
| Ablations | calculator `worst_sys_prompt`, `misleading_sys_prompt`, `missing_*_tool`, `wrong_param_input_schema` |

## Checklist

`user_prompt.md`
- Every `{field}` is an `input_schema` field, spelled and cased exactly.
- Literal braces (JSON examples) are doubled: `{{` and `}}`.
- Building the messages prints no `User prompt template failed` warning, and no trace shows the
  run state as JSON in place of the user prompt.

`system_prompt.md`
- No `{field}` placeholders.
- Tool names match the bound names exactly (function name, or `name:` for class tools).
- Output field names match `output_schema` exactly, and the prompt says to call `end_execution`.
- It does not tell the model to omit a field the schema requires.

`agent.yaml` and `versions.yaml`
- Prompt paths start with `core::`.
- Optional output fields have `default: null`.
- Each version sets `version:` to its key, and repeats the whole `tools:` list when it changes it.
- DSPy agents: every `signature_prompts` key matches a predictor name.
