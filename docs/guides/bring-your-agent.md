# Bring your own agent

Install the [framework](../../README.md#install), activate its environment, and create a project:

```bash
gym new my_agent --directory ../my-agent-project
cd ../my-agent-project
cp .env.example .env
```

## Connect your agent

Edit `agents/my_agent/adapter.py`. The YAML entrypoint `core::adapter.py:run` loads it directly:

```python
async def run(*, inputs: dict, system_prompt: str, user_prompt: str, model) -> dict:
    result = await your_agent.ainvoke({
        **inputs,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt.format(**inputs),
    })
    return {"answer": result["answer"]}
```

Import or construct `your_agent` through your application's integration. A synchronous
`def run` also works. Return a dictionary matching the YAML output schema; input and output
field names must be distinct. Update the template's `Answer` model, and the `{question}`
placeholder in `prompts/user_prompt.md` and `prompts/baseline/user_prompt.md`, when changing
its schema.

Forward the current prompts on **every call** so optimization reaches your agent. The injected
`model` is a LangChain chat model with Gym tracing and cost accounting. Agents with their own
clients need their own provider configuration and cost integration. Install imported agent
packages in the active environment. The working directory is on Python's import path; the agent
directory is not.

## Add data and choose a score

Write one case per line in `agents/my_agent/data/train.jsonl` and `test.jsonl`:

```json
{"id":"train-001","inputs":{"question":"What is 2 + 2?"},"expected_output":{"answer":"4"}}
```

`inputs` and `expected_output` keys are the schema's field names; with the template's exact
match, a case whose keys differ from the output schema scores 0. Use IDs that are unique
ignoring case (with `/` read as `_`) and distinct input objects across both splits. ReAPO sees train; test measures
the final result. Use representative train/test cases when assessing optimization quality.

```bash
gym --project . init my_agent
```

After data edits, add `--force` to initialization. Configure credentials using the framework's
[model guide](../../README.md#configure-models).

The template uses case-insensitive exact match over the whole output. Customize
`evaluations/evaluators/evaluator-my_agent_exact.json` inside the agent; initialization preserves
it. See [Evaluators](evaluators.md) for judges, tool-call checks and custom metrics.

## Baseline, optimize, test

Follow the generated project README or [Your own agent](../../README.md#your-own-agent).
`baseline-test` uses frozen baseline prompts. Optimization selects `:train` automatically and
restores its best evaluated train round; `:test` evaluates the resulting current prompts.
Commit the baseline first (`git init` the project if needed). `-r` limits rounds, not dollars; increase repeats for stochastic
agents. Use `-c` to resume and `--reset -o` to start again from baseline prompts.

Scores and traces are in `experiments/`; optimization snapshots and accounting are in
`tmp_*/`. Reuse `agents/my_agent/prompts/*.md` in your application. Use a fresh experiment
name when changing prompts or data because completed cases are cached.

## Project selection

`gym --project DIR` selects a project for `init`, `eval` and `optimize`; the optimizer's
tool server and evaluation runs use the same project. Without `--project`, commands use the
bundled agents whatever the current directory.

The selected project's `.env` takes precedence over the framework checkout's `.env`, which
supplies the keys the project's file leaves out; shell variables still win. From Python, pass
`agents_gym_core.project.ProjectDirectoryProvider(Path(...))` to
`agents_gym_core.sdk.project_registry.ProjectAgentRegistry(directory_provider=...)`.
`gym reproduce` always uses the bundled paper agents.

For native configuration-based agents, see [Building an agent](new-agent.md).
