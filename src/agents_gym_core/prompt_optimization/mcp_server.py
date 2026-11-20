"""The 20 MCP tools ReAPO's optimizer works through (listed in the paper's appendix).

The tools read the target agent, edit its prompts, run evaluations and read their results;
they cannot change eval sets, evaluators or agent configuration. The server is named
``coach-light``, the name the claude engine's tool allowlist uses. In an optimization run
(``COACH_TARGET_AGENT`` set), the tools evaluate only the target version and do not serve the
content of its other splits: see ``mcp_tools/helpers.py``.
"""

import argparse
import os
import signal
from contextlib import suppress
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from agents_gym_core.project import ProjectDirectoryProvider
from agents_gym_core.prompt_optimization.best_round import save_last_prompts
from agents_gym_core.prompt_optimization.mcp_tools import (
    tools_agents,
    tools_datapoints,
    tools_experiments,
    tools_runner,
    usage_tracker,
)
from agents_gym_core.prompt_optimization.mcp_tools.helpers import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_MAX_DATAPOINTS,
    DEFAULT_MODEL,
    _workspace,
    optimization_context,
    optimization_target,
)
from agents_gym_core.utils.directory_provider import CoreDirectoryProvider

mcp = MCPServer("coach-light")


# ---------------------------------------------------------------------------
# Agent listing & inspection
# ---------------------------------------------------------------------------


@mcp.tool()
def get_agent_files(agent_name: str, source: str = "all") -> str:
    """List all files in an agent's directory tree with relative paths and sizes.

    Args:
        agent_name: Name of the agent
        source: "all" (default) to search both gym and core, or "gym"/"core" for a specific source
    """
    return tools_agents.get_agent_files(agent_name, source)


@mcp.tool()
def read_agent_file(agent_name: str, relative_path: str, source: str = "all") -> str:
    """Read any file from an agent's directory (tools.py, versions.yaml, data files, evaluator JSON, etc.).

    Args:
        agent_name: Name of the agent
        relative_path: Path relative to the agent directory (e.g. "tools.py", "versions.yaml", "data/input.csv", "evaluations/evaluators/evaluator-judge.json")
        source: "all" (default) to search both gym and core, or "gym"/"core" for a specific source
    """
    return tools_agents.read_agent_file(agent_name, relative_path, source)


@mcp.tool()
def summarize_agent(agent_name: str, source: str = "all") -> str:
    """Get a concise summary of an agent: what it does, its tools, inputs/outputs, and eval score overview.

    Args:
        agent_name: Name of the agent
        source: "all" (default) to search both gym and core, or "gym"/"core" for a specific source
    """
    return tools_agents.summarize_agent(agent_name, source)


@mcp.tool()
def inspect_agent(agent_name: str, source: str = "all") -> str:
    """Deep-inspect an agent's configuration. Shows tools, eval count, schemas, cloud tags, evaluators, folders.

    Args:
        agent_name: Name of the agent to inspect
        source: "all" (default) to search both gym and core, or "gym"/"core" for a specific source
    """
    return tools_agents.inspect_agent(agent_name, source)


@mcp.tool()
def inspect_tools(agent_name: str, source: str = "all") -> str:
    """Inspect an agent's tools in detail: _target_ paths, arguments, tracedump data stats.

    Args:
        agent_name: Name of the agent
        source: "all" (default) to search both gym and core, or "gym"/"core" for a specific source
    """
    return tools_agents.inspect_tools(agent_name, source)


@mcp.tool()
def get_prompt(agent_name: str, prompt_type: str = "system", source: str = "all") -> str:
    """Read one configured prompt of an agent.

    Args:
        agent_name: Name of the agent
        prompt_type: "system", "user", or an exact key from agent.signature_prompts
        source: "all" (default) to search both gym and core, or "gym"/"core" for a specific source
    """
    return tools_agents.get_prompt(agent_name, prompt_type, source)


# update_prompt and run_agent_async are async so that they run on the server's event loop, one at a
# time and in the order the calls arrive (sync tools run in worker threads, in parallel). A run
# started in the same turn as an edit then reads, and saves as its round's prompts, the edited prompt.
@mcp.tool()
async def update_prompt(
    agent_name: str,
    new_content: str,
    prompt_type: str = "system",
    source: str = "all",
    round_name: str = "",
) -> str:
    """Replace one configured prompt of an agent. After updating, re-run the agent to verify the change.

    Args:
        agent_name: Name of the agent
        new_content: The new prompt content (full replacement)
        prompt_type: "system", "user", or an exact key from agent.signature_prompts
        source: "all" (default) to search both gym and core, or "gym"/"core" for a specific source
        round_name: Round identifier (e.g. "round_001") to label this prompt version in usage tracking. Empty = auto-increment.
    """
    reply = tools_agents.update_prompt(agent_name, new_content, prompt_type, source, round_name)
    target, workspace = optimization_target(), _workspace()
    if target and workspace:  # what -c restores, even if the session is killed before it ends
        with suppress(OSError, UnicodeError):
            save_last_prompts(workspace, target[0])
    return reply


# ---------------------------------------------------------------------------
# Eval-set checking & datapoint viewing (read-only)
# ---------------------------------------------------------------------------


@mcp.tool()
def check_evals(agent_name: str = "", source: str = "all") -> str:
    """Run the eval-set and evaluator checker on an agent or all agents. Reports errors and warnings.

    Args:
        agent_name: Specific agent to check (empty = all agents)
        source: "all" (default) to search both gym and core, or "gym"/"core" for a specific source
    """
    return tools_datapoints.check_evals(agent_name, source)


@mcp.tool()
def get_datapoint(
    agent_name: str,
    datapoint_name: str,
    source: str = "all",
) -> str:
    """Get a specific datapoint from an eval-set by name. Shows inputs, expectedOutput, and evaluationCriterias.
    Searches all eval sets automatically.

    Args:
        agent_name: Name of the agent
        datapoint_name: Name of the datapoint (evaluation name)
        source: "all" (default) to search both gym and core, or "gym"/"core" for a specific source
    """
    return tools_datapoints.get_datapoint(agent_name, datapoint_name, source)


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------


@mcp.tool()
async def run_agent_async(
    agents: str,
    max_datapoints: int = DEFAULT_MAX_DATAPOINTS,
    model: str = DEFAULT_MODEL,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    loops: str = "",
    datapoints: str = "",
) -> str:
    """Start an agent run asynchronously. Returns a run_id to track progress. Each run creates a new experiment folder (e.g. mcp_run_001, mcp_run_002) so previous results are preserved.

    Args:
        agents: Space-separated agent specs (e.g. "calculator" or "search_agent")
        max_datapoints: Max datapoints per agent. 0 = full eval-set (default). Use a positive number (e.g. 10) for a faster partial run during debugging. Ignored when datapoints is provided.
        model: Model to use (default: gpt-5-2025-08-07)
        experiment_name: Experiment name for output (default: mcp_run)
        loops: Space-separated loop specs (e.g. "deep-agents" or "basic:any planning"). Empty = default (basic).
        datapoints: Pipe-separated datapoint names to run (exact match filter), e.g. "Name one|Name two". Use | as separator since names may contain spaces. Empty = all datapoints. Preferred over max_datapoints for targeted debugging.
    """
    return tools_runner.run_agent_async(
        agents, max_datapoints, model, experiment_name, loops, datapoints
    )


@mcp.tool()
def get_run_status(run_id: str) -> str:
    """Check the status and progress of an async agent run.

    Args:
        run_id: The run ID returned by run_agent_async
    """
    return tools_runner.get_run_status(run_id)


@mcp.tool()
def wait_for_run(run_id: str, poll_interval: int = 30) -> str:
    """Block until an async agent run completes and return the final status.
    Use this instead of polling get_run_status in a loop.

    Args:
        run_id: The run ID returned by run_agent_async
        poll_interval: Seconds between internal status checks (default: 30)
    """
    return tools_runner.wait_for_run(run_id, poll_interval)


@mcp.tool()
def save_optimizer_log(output_dir: str, filename: str, content: str) -> str:
    """Save a log file to the optimizer output directory.
    Use this to save prompt snapshots, score summaries, and final reports.

    Args:
        output_dir: The output directory name (e.g. "tmp_claude_optimizer_agent_model")
        filename: The filename to write, supports subdirs (e.g. "round_000/scores.md")
        content: The file content
    """
    return tools_runner.save_optimizer_log(output_dir, filename, content)


@mcp.tool()
def read_optimizer_log(output_dir: str, filename: str) -> str:
    """Read a file from the optimizer output directory.
    Use this to read back saved prompts for reverting changes.

    Args:
        output_dir: The output directory name (e.g. "tmp_claude_optimizer_agent_model")
        filename: The filename to read (e.g. "round_000/system_prompt.md")
    """
    return tools_runner.read_optimizer_log(output_dir, filename)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


@mcp.tool(meta={"anthropic/maxResultSizeChars": tools_experiments.MAX_FULL_RESULTS_MCP_CHARS})
def get_full_results(experiment_name: str, agent_name: str = "") -> str:
    """Get the full results from an experiment: scores and per-datapoint details with justifications.

    Args:
        experiment_name: Name of the experiment (default from run_agent is "mcp_run")
        agent_name: Filter to a specific agent (empty = all agents)
    """
    return tools_experiments.get_full_results(experiment_name, agent_name)


@mcp.tool()
def list_experiments(experiment_name: str = "") -> str:
    """List available experiment results. Shows agents, models, and scores.

    Args:
        experiment_name: Specific experiment to inspect (empty = list all experiments)
    """
    return tools_experiments.list_experiments(experiment_name)


@mcp.tool()
def get_experiment_results(
    experiment_name: str,
    agent_name: str,
    datapoint_name: str = "",
) -> str:
    """Get detailed experiment results for an agent. Shows scores, outputs, and errors per datapoint.

    Args:
        experiment_name: Name of the experiment
        agent_name: Name of the agent
        datapoint_name: Specific datapoint to inspect (empty = summary of all)
    """
    return tools_experiments.get_experiment_results(experiment_name, agent_name, datapoint_name)


@mcp.tool()
def get_agent_messages(
    experiment_name: str,
    agent_name: str,
    datapoint_name: str,
    run_number: int = 1,
) -> str:
    """Get the full message history (conversation) from an agent run. Useful for debugging agent behavior.

    Args:
        experiment_name: Name of the experiment
        agent_name: Name of the agent
        datapoint_name: Name of the datapoint/test case
        run_number: Run number (default: 1)
    """
    return tools_experiments.get_agent_messages(
        experiment_name, agent_name, datapoint_name, run_number
    )


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------


@mcp.tool()
def get_usage_stats(agent_name: str = "") -> str:
    """Get a full usage report: prompt version history and rounds per agent.

    Shows:
    - System/user prompt versions (round_000, round_001 ...) with timestamps
    - Per-round record: which prompt versions were active, model, datapoints, score, experiment

    Args:
        agent_name: Filter to a specific agent (empty = all agents)
    """
    return usage_tracker.get_stats(agent_name)


@mcp.tool()
def reset_usage_stats(
    agent_name: str = "",
    prompt_type: str = "",
    scope: str = "",
) -> str:
    """Reset usage statistics with fine-grained control.

    Granularity levels:
      reset_usage_stats()
          → wipe everything (all agents, all versions, all runs)

      reset_usage_stats(agent_name="my_agent")
          → wipe one agent completely

      reset_usage_stats(agent_name="my_agent", prompt_type="system")
          → reset system prompt version history only (back to no versions), keep runs

      reset_usage_stats(agent_name="my_agent", prompt_type="user")
          → reset user prompt version history only, keep runs

      reset_usage_stats(agent_name="my_agent", scope="rounds")
          → clear rounds ledger only, keep prompt version history

    Args:
        agent_name: Target agent. If empty, resets ALL agents.
        prompt_type: "system" or "user" — resets only that prompt's version history.
                     Requires agent_name.
        scope: "rounds" — resets only the rounds ledger for the agent.
               Requires agent_name.
    """
    return usage_tracker.reset_stats(agent_name, prompt_type, scope)


_STOP_SIGNALS = [s for s in (signal.SIGTERM, getattr(signal, "SIGHUP", None)) if s is not None]


def _stop_runs_and_die(signum: int, _frame: object) -> None:
    """On SIGTERM/SIGHUP, stop the testbed runs still in flight, then die of the signal."""
    for sig in _STOP_SIGNALS:
        signal.signal(sig, signal.SIG_IGN)
    tools_runner.terminate_active_runs()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _serve() -> None:
    # No testbed run outlives the server: whether its session closes or it is signalled.
    for sig in _STOP_SIGNALS:
        if signal.getsignal(sig) != signal.SIG_IGN:  # e.g. SIGHUP under nohup: keep ignoring it
            signal.signal(sig, _stop_runs_and_die)
    try:
        mcp.run()
    finally:
        tools_runner.terminate_active_runs()


def main(
    argv: list[str] | None = None, *, directory_provider: ProjectDirectoryProvider | None = None
) -> None:
    if directory_provider is not None and type(directory_provider) is not ProjectDirectoryProvider:
        raise TypeError("directory_provider must be an exact ProjectDirectoryProvider or None")
    parser = argparse.ArgumentParser(description="ReAPO tool server")
    parser.add_argument("--project", type=Path)
    args = parser.parse_args(argv)
    provider = directory_provider or (
        ProjectDirectoryProvider(args.project) if args.project else CoreDirectoryProvider()
    )
    with optimization_context(provider):
        _serve()


if __name__ == "__main__":
    main()
