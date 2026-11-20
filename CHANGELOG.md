# Changelog

## [1.0.0]

Initial public release, accompanying the paper *Simplicity is Enough: ReAct Agents for Prompt
Optimization*.

- The 11 agents evaluated in the paper: six public benchmarks (HotpotQA, HoVer, AIME 2025,
  LiveBench-Math, IFBench, PUPA) on GEPA's DSPy programs, and five enterprise agents, with the
  paper's prompts as agent versions.
- Ten additional agents with their eval sets and evaluators: the
  `calculator` example, a mentorship chatbot (`athena_chatbot`),
  `hts_classifier_agent` (an excerpt of the US Harmonized Tariff Schedule; its CROSS rulings
  lookup uses an offline fixture), `loan`, `product_recommendation`, `salesforce_agent`,
  `search_agent`, `semantic_analysis`, `servicenow_work_order_accelerator` and the `for_loop`
  example. `search_agent` is the only one that calls a live service: it searches the web with
  Tavily or Google when `TAVILY_API_KEY` or `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` are set, and
  with DuckDuckGo otherwise or when the key is rejected.
- Prompt optimizers: ReAPO (the `react` engine) and GEPA-DSPy, the paper's GEPA baseline on
  the enterprise agents. Also included: a native GEPA variant and the `claude` engine, which runs
  ReAPO through the Claude Code CLI.
- The `gym` command: `new` (a standalone project for your own Python agent, selected with
  `gym --project DIR`), `init`, `eval` (the testbed: repeated runs, confidence intervals, token
  and dollar cost), `optimize`, `reproduce` (the paper's tables) and `uipath`
  (`generate` / `run` / `eval` through the UiPath CLI).
- A Python API, `GymRuntime`, for typed agent execution, native graph access, batch evaluation,
  loop comparisons, tracing and reports with existing LangChain chat models.
- Optimizer scripts, `docs/REPRODUCING.md` and framework guides.
