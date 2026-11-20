# Agents Gym Core

Evaluate an LLM agent, improve its prompts with ReAPO, and compare the result on held-out
data. This repository also contains the code, agents and released prompts for
*Simplicity is Enough: ReAct Agents for Prompt Optimization* (NeurIPS 2026).

| Your goal | Start here |
|---|---|
| Reproduce the paper's evaluations | [Reproduce the paper](#reproduce-the-paper) |
| Optimize your own Python agent | [Your own agent](#your-own-agent) |
| Try a bundled agent | [Quickstart](#quickstart) |
| Extend the framework | [Development](#development) and the [agent guide](docs/guides/new-agent.md) |

ReAPO casts prompt optimization as a ReAct agent with evaluation and introspection tools. Each
round, the optimizer agent evaluates the target system on its train split, reads per-datapoint
results and traces of the failures, attributes the errors and rewrites the prompts; it keeps
the best-on-train prompt set, a single greedy lineage with no evolutionary search. Across six
public benchmarks, the paper finds that it matches or surpasses GEPA and MIPROv2 with 2 to 7
times fewer evaluation rollouts and lower optimization cost, and that it outperforms GEPA on
three of five enterprise tasks. Its prompts transfer across model families, and the approach
remains effective when driven by a 9B open-weight model.

- **Prompt optimization**: ReAPO (a ReAct agent that runs, inspects and edits the target agent
  through 20 MCP tools), GEPA-DSPy (the paper's enterprise baseline), a native GEPA variant
  (`python -m agents_gym_core.prompt_optimization.gepa`) and a `claude` engine that runs ReAPO
  through the Claude Code CLI.
- **Agents as configuration**: an `agent.yaml` (schemas, tools, loop, prompts, eval sets) plus
  prompt files; `versions.yaml` holds named variants, including every prompt set the paper reports.
- **Loops**: ReAct-style `basic`, validation, planning, summarizing, DSPy programs and more.
- **Evaluation**: exact-match, JSON-similarity, tool-call, LLM-judge and custom Python
  evaluators; repeated runs with confidence intervals; token and dollar cost per run.
- **Models**: the UiPath LLM Gateway or the providers' APIs (OpenAI, Azure OpenAI, Anthropic,
  Gemini, Fireworks, any OpenAI-compatible server such as vLLM), for every model role.

## Install

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/UiPath/agents-gym-core
cd agents-gym-core
python3 scripts/check_dependency_policy.py   # offline check of dependency origins and hashes
uv sync --locked                             # add --extra dspy for the public benchmarks
cp .env.example .env                         # then configure one model backend
```

- Python 3.11.5 to 3.13 and uv 0.13.4 or newer, from a source checkout (not on PyPI).
- The `gepa-artifact` submodule holds GEPA's released programs and data at the paper's commit;
  `GIT_LFS_SKIP_SMUDGE=1` skips its 1.9 GB of archived results. Without the submodule,
  `gym init` clones it into `~/.cache/agents-gym-core` (`GEPA_ARTIFACT_DIR` overrides it).
- `gym init` builds the public benchmarks' splits from pinned dataset revisions. HotpotQA and
  HoVer also build a shared 1 GB BM25 index over a 1.8 GB corpus, so initialize only the agents
  you need.
- Supply-chain checks and CI's guarded install: [dependency policy](docs/guides/dependency-security.md).
  Dataset sources and licences: [REPRODUCING](docs/REPRODUCING.md#datasets).

From a wheel (benchmark data is still built by `gym init`):

```bash
uv build --wheel
uv export --locked --no-emit-project --extra dspy -o requirements.txt
python3 -m venv .venv-wheel && . .venv-wheel/bin/activate  # Python 3.11.5-3.13
pip install "pip==26.3"
pip install --require-hashes -r requirements.txt
pip install --no-deps dist/agents_gym_core-*.whl
```

## Configure models

Set `AGENTS_GYM_LLM_BACKEND` in `.env`:

| Backend | Models resolve through |
|---|---|
| `langchain` (default) | the providers' APIs: Azure OpenAI if `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` are set, else OpenAI (`OPENAI_API_KEY`; `OPENAI_BASE_URL` for vLLM and other compatible servers); Anthropic, Gemini and Fireworks by their own keys |
| `uipath` | the UiPath LLM Gateway (`LLMGW_*` / `UIPATH_*`) |

The backend applies to every role: agent, optimizer, LLM judges and models inside tools
(`--model_backend` overrides it for one run). Each LLM judge uses the model in its evaluator
JSON. `.env.example` also has endpoint settings for `Qwen/Qwen3.5-9B` (`QWEN_9B_VLLM_*`) and
`deepseek-ai/DeepSeek-V4.1-Flash-1017` (`DEEPSEEK_V41_FLASH_VLLM_*`) on your own vLLM server or
a hosted endpoint such as Fireworks. On the `langchain` backend, `AGENTS_GYM_MODEL_BASE_URL`
sends model names no registry knows (a self-hosted system model, say) to an OpenAI-compatible
server.

## The `gym` command

| Command | Does |
|---|---|
| `gym new <name>` | creates a standalone project with a Python adapter, prompts and example JSONL data |
| `gym init <agent>` | prepares an agent's data: splits, downloads, indexes (`--list` shows which agents need it) |
| `gym eval --agents <agent[:version]> ...` | the testbed: evaluates agents on their eval sets over repeated runs |
| `gym optimize <agent:version>` | optimizes an agent's prompts; ReAPO by default |
| `gym reproduce <table>` | evaluates the paper's prompts for one table of the paper (`--list`) |
| `gym uipath generate \| run \| eval` | packages agents for the UiPath CLI and runs them with `uipath run` / `uipath eval` |

Each command takes `--help`.

## Your own agent

```bash
source .venv/bin/activate
gym new my_agent --directory ../my-agent-project
cd ../my-agent-project && cp .env.example .env   # add your credentials
gym --project . init my_agent                    # converts the example JSONL; no model calls
```

Replace `agents/my_agent/adapter.py` with your agent and add representative train/test data
(the project README walks through it). The adapter, sync or async, receives typed inputs, the
current prompts and the configured LangChain model; use those prompts so ReAPO can improve them.
Commit the baseline, then:

```bash
gym --project . eval --agents my_agent:baseline-test --models gpt-4.1-mini-2025-04-14 --num_runs 1 --experiment_name baseline
gym --project . optimize my_agent -m gpt-4.1-mini-2025-04-14 --optimizer-model gpt-5-2025-08-07 -r 2 -o
gym --project . eval --agents my_agent:test --models gpt-4.1-mini-2025-04-14 --num_runs 1 --experiment_name optimized
```

Compare the two `experiments/*/final_score_per_agent.json`. Optimization uses `:train` and edits
the project's prompts in place; `:test` stays held out. Adapter contract, scoring and data:
[Bring your own agent](docs/guides/bring-your-agent.md).

## Quickstart

```bash
uv run gym eval --agents calculator \
    --models gpt-4.1-mini-2025-04-14 --num_runs 2 --experiment_name quickstart
```

Scores land in `experiments/quickstart/final_score_per_agent.json`. Select agents by name,
`name:version`, `*` wildcards or `@tag`; the default `fill` mode reuses an experiment's
completed results and retries failed ones.

Optimize `claim_history` with ReAPO from its unoptimized baseline:

```bash
uv run gym optimize claim_history:train -m gpt-4.1-mini-2025-04-14 \
    --optimizer-model gpt-5-2025-08-07 -r 2 --reset -o
```

ReAPO edits the prompt files in place (`--reset` first copies `prompts/baseline/` over them;
restore them with git afterwards, or let the scripts below do it) and keeps each round's prompts and scores under
`tmp_<name>_<agent>_.../round_*`, with time, tokens and cost in `run_metadata.json`. `-c`
continues an interrupted run.

## Python API

```python
from agents_gym_core.runtime.api import GymRuntime

gym = GymRuntime(model=llm)  # any LangChain chat model
result = await gym.run("calculator", {"expression": "17 * 23"})  # {"answer": 391}
report = await gym.evaluate("calculator", cases, evaluators, repeats=2)  # SDK EvaluationItems
```

Tool and schema overrides, in-memory evaluation, loop comparison and custom judges:
[Python API](docs/guides/python-api.md).

## Reproduce the paper

The paper's optimized prompts are committed as agent versions, so its tables can be evaluated
without re-optimizing:

```bash
uv sync --locked --extra dspy
uv run gym reproduce --list
uv run gym reproduce main-mini --agents aime_2025_agent --versions baseline-test --dry-run
uv run gym init aime_2025_agent
uv run gym eval --agents aime_2025_agent:baseline-test --models gpt-4.1-mini-2025-04-14 --num_runs 1 --max_datapoints 1 --experiment_name paper_smoke
```

The one-case run checks setup: its final summary should count 0 errored datapoints and
evaluations. To evaluate a full table with the paper's data and repeat counts, initialize its
agents and run it (paid model calls; `--agents` and `--versions` narrow it, `--dry-run` previews
it):

```bash
for a in hotpotqa hover aime_2025_agent livebench_math ifbench papillon; do uv run gym init "$a"; done
uv run gym reproduce main-mini
```

A rerun resumes where it stopped: it retries model-endpoint outages and evaluator exceptions and
keeps failed agent runs as scored. Use a new `--experiment-name` for a fresh evaluation.
Apart from the Qwen tables, the system model is `gpt-4.1-mini-2025-04-14` for the public
benchmarks and the enterprise tool ablation and `gpt-5-2025-08-07` for the other enterprise
tables; the enterprise judges use `gpt-4o-2024-11-20`. The Qwen tables (`transfer`, `self-opt-9b`, `self-opt-122b`,
`qwen-agents`) need a Qwen3.5-9B (vLLM) or Qwen3.5-122B-A10B endpoint.

To run the optimizers yourself:

```bash
bash scripts/run_react_benchmarks.sh   # ReAPO, public benchmarks
bash scripts/run_react.sh              # ReAPO, enterprise agents
bash scripts/run_gepa.sh               # GEPA-DSPy baseline, enterprise agents
```

[docs/REPRODUCING.md](docs/REPRODUCING.md) maps each table to its prompt versions and settings.
Prompt optimization and final evaluation use separate train and test partitions; the public
benchmarks keep GEPA's published partitions ([split provenance](docs/REPRODUCING.md#data)).

## Repository layout

```
src/agents_gym_core/
  agents/                 agent definitions (the 11 paper agents, calculator and nine more)
  loops/                  loop templates and the loop factory
  hydra_agent/            agent registry, schema builder, evaluation loading, init
  models/                 model registries, backends, judge routing
  runtime/                experiment orchestration, runners, aggregation, cost tracking
  prompt_optimization/    ReAPO (react_optimizer, mcp_server, mcp_tools, skills), GEPA
  reproduce.py            the paper's tables, for `gym reproduce`
  sdk/cli.py              the `gym` command
  testbed_core.py         the testbed
scripts/                  optimizer scripts
gepa-artifact/            submodule: GEPA's released benchmark programs and data
docs/                     reproduction guide, framework guides
tests/                    offline tests (no model calls)
```

## Documentation

- [Reproducing ReAPO](docs/REPRODUCING.md): every table's versions and settings
- [Running the testbed](docs/guides/running-the-testbed.md): flags, selectors, run modes
- [Building an agent](docs/guides/new-agent.md): from an empty folder to a validated run
- [Bring your own agent](docs/guides/bring-your-agent.md): standalone projects and Python adapters
- [Python API](docs/guides/python-api.md): `GymRuntime` runs and in-memory evaluation
- [Loops](docs/guides/loops.md) and [prompts](docs/guides/prompts.md): what each loop does and how prompts are rendered
- [Evaluators](docs/guides/evaluators.md): built-in types, criteria, custom evaluators
- [Run output formats](docs/guides/run-output-formats.md): what the testbed writes
- [NOTICE](NOTICE): third-party code and data in this repository

## Development

```bash
uv sync --locked --extra dspy --extra dev
uv run pytest -q tests                # offline, a few minutes; benchmark tests skip until `gym init`
uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
uv run pyright src tests scripts
```

## Citation

```bibtex
@inproceedings{rusu2026simplicity,
  title     = {Simplicity is Enough: {ReAct} Agents for Prompt Optimization},
  author    = {Dumitrescu, Andrei and Rusu, Andrei C. and Badea, Adrian Catalin and
               Li, Bai and Maria, Cosmin and Anghelina, Ion and
               Religa, Tomasz Lukasz and Bobolea, Dragos H},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

See [CITATION.cff](CITATION.cff). License: [MIT](LICENSE).
