# Claude engine

`--engine claude` runs ReAPO's optimization protocol (the skill prompts in `../skills/`)
through the Claude Code CLI. The paper's ReAPO experiments use the `react` engine's
LangGraph agent.

```bash
gym optimize calculator --engine claude --optimizer-model opus -r 4 -b 20
```

The CLI runs non-interactively (`--print`) through ReAPO's 20 MCP tools (server `coach-light`)
(`--strict-mcp-config`, built-in tools disabled). Agent inspection, evaluation, prompt edits
and logging use those tools: `update_prompt` edits prompts and `save_optimizer_log` writes
logs. `-b` caps the Claude CLI session's model spend in USD. Evaluation-model calls have
their own cost accounting in the optimizer workspace. Requires the `claude` CLI, installed
and signed in.

The workspace is the same as the react engine's: `tmp_<name>_<agent>_..._skill_<skill>_temp_<t>/`
with one `round_NNN/` directory per round, plus `claude_session_*.log` (readable
transcript) and `raw_stream_*.jsonl` (the CLI's stream, parsed by `stream_parser.py`).
`-c` continues from the last completed round; `-o` starts over.
