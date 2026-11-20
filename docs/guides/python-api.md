# Python API

`GymRuntime` runs and evaluates agents from Python with an existing LangChain chat model (`llm`).

## Run an agent

In a notebook:

```python
from agents_gym_core.runtime.api import GymRuntime

gym = GymRuntime(model=llm)
result = await gym.run("calculator", {"expression": "17 * 23"})
print(result)  # {"answer": 391}
```

In a script, use `asyncio.run(gym.run(...))` instead of top-level `await`.

The agent runs with its own `loop:` if it has one (the public benchmarks' DSPy programs), as in
the testbed, otherwise `basic:auto`. Single-agent execution does not load evaluation sets.

| Argument | Does |
|---|---|
| `version=` | selects an agent version |
| `loop_version=` (runtime) | selects a basic-loop variant |
| `config=` | passes LangGraph run settings |
| `embeddings=` (runtime) | supplies the embedding model to tools, with the testbed's injection rules |
| `tools=` (`run()`, `graph()`) | replaces the configured tools before their factories run |
| `input_schema=` (`run()`, `graph()`) | a graph-local input model; it must define every field the user prompt uses, or `graph()` raises `ValueError` |

Tools with their own model configuration keep it, and the overrides apply only to that call.
For streaming or repeated calls, `gym.graph("calculator")` returns the native compiled graph.

```python
from agents_gym_core.agents.sql.tools import SQLTools

sql_tools = SQLTools(db_path="orders.sqlite", llm=llm)()
result = await gym.run("sql", {"query": "How many orders are there?"}, tools=sql_tools)
```

Tracing is opt-in through `setup_tracer()`.

## Evaluate in-memory cases

`evaluate()` uses the testbed's runner, tracing and reports:

```python
from uipath.eval.evaluators.evaluator_factory import EvaluatorFactory
from uipath.eval.models.evaluation_set import EvaluationItem

exact = EvaluatorFactory.create_evaluator({
    "version": "1.0", "id": "exact", "description": "exact answer",
    "evaluatorTypeId": "uipath-exact-match",
    "evaluatorConfig": {"targetOutputKey": "*", "caseSensitive": True},
})
case = EvaluationItem(id="math", name="math", inputs={"expression": "17 * 23"},
                      evaluationCriterias={"exact": {"expectedOutput": {"answer": 391}}})
report = await gym.evaluate("calculator", [case], [exact],
                            repeats=2, concurrency=2, output_dir="experiments/demo")
display(report["metrics_df"])
```

- `cases` are SDK `EvaluationItem` objects; pass evaluator instances explicitly.
- `graph=` takes a graph built with tool or input-schema overrides; `graphs=[basic_graph,
  planning_graph]` compares loops in one experiment with shared concurrency.
- `GymRuntime(model, loop_name="planning")` selects the planning loop; `loop_name="deep-agents"`
  runs the same typed inputs and normalized outputs through the Deep Agents harness.
- `judge_model=judge_llm` scores LLM evaluators with your own chat model, without modifying the
  evaluator instances or setting up routing.
- `evaluator_weights` controls the overall score and keeps every evaluator's result.
- `tracing=` passes a `(span_exporter, tracer_provider, log_exporter)` session. Without it,
  `evaluate()` uses a session set up with `setup_tracer()`, or one shared by concurrent calls
  and ended after the last.

Runs and reports in the output directory are overwritten. Check `report["evaluation_complete"]`
before using aggregate scores: `failed_datapoint_runs` lists unscored cases and
`failed_evaluator_runs` SDK evaluator exceptions of agent runs that did not fail, and
`evaluate()` warns when either is non-empty. `costs_df` has the native model-call and token counts.
