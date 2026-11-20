"""ReAPO: a LangGraph ReAct agent that optimizes another agent's prompts through MCP tools.

The optimizer follows a skill prompt (``skills/full.md``): it evaluates the agent, reads the
worst traces, edits the prompts and keeps or reverts each change, all through tool calls.
"""

import asyncio
import json
import os
import posixpath
import signal
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import create_react_agent

from agents_gym_core.models.chat.clients import (
    HuggingFaceChatModels,
    VllmChatModels,
    get_chat_model,
)
from agents_gym_core.prompt_optimization.mcp_tools.helpers import (
    EVAL_LOOPS_ENV,
    EVAL_MODEL_ENV,
    TARGET_ENV,
    _resolve_agent_path,
    compute_output_dir,
    init_started_experiments,
    repo_root,
    require_agent_dir,
    tee_to_log,
)
from agents_gym_core.prompt_optimization.mcp_tools.mcp_client import coach_light_session
from agents_gym_core.prompt_optimization.react_optimizer.prompts import (
    DEFAULT_SKILL,
    SYSTEM_PROMPT,
    build_optimization_prompt,
)
from agents_gym_core.prompt_optimization.run_accounting import (
    SESSIONS_DIRNAME,
    print_run_summary,
    run_totals,
)
from agents_gym_core.runtime.cost_tracking import CostTrackingCallback

# ---------------------------------------------------------------------------
# Tools ablation (the paper's Full, Score-Only and Macro configurations)
# ---------------------------------------------------------------------------

# The Results Analysis tools each configuration removes. Score-Only leaves the optimizer
# only the aggregate score of each run; Macro keeps get_full_results, whose evaluator feedback
# on failures includes expected values. Both hide the eval-set files from read_agent_file.
ABLATION_REMOVED_TOOLS: dict[str, set[str]] = {
    "full": set(),
    "score-only": {
        "get_agent_messages",
        "get_datapoint",
        "get_experiment_results",
        "get_full_results",
        "check_evals",
    },
    "macro": {"get_agent_messages", "get_datapoint", "get_experiment_results", "check_evals"},
}
_EVAL_SETS_DIR = "evaluations/eval-sets"


def _is_eval_sets_path(path: str) -> bool:
    """True when ``path`` points into the eval sets however it is spelled (``./``, ``..``, ``//``,
    backslashes, another case or a compatibility character such as ``ſ``, which macOS file
    systems resolve to the same directory)."""
    normalized = posixpath.normpath(path.replace("\\", "/"))
    normalized = unicodedata.normalize("NFKC", normalized).casefold()
    return f"/{_EVAL_SETS_DIR}/" in f"/{normalized}/"


def _resolves_into_eval_sets(agent_name: str, source: str, path: str) -> bool:
    """True when ``path`` reaches the agent's eval sets, including file aliases."""
    agent_dir, err = require_agent_dir(agent_name, source)
    if err:
        return False  # the server refuses the read itself
    target = _resolve_agent_path(agent_dir, path)
    if target is None:
        return False
    try:
        eval_sets = (agent_dir / _EVAL_SETS_DIR).resolve()
        if target == eval_sets or target.is_relative_to(eval_sets):
            return True
        return target.is_file() and any(
            target.samefile(candidate) for candidate in eval_sets.rglob("*") if candidate.is_file()
        )
    except (OSError, RuntimeError, ValueError):
        return True


def _block_eval_sets(tool: BaseTool) -> BaseTool:
    async def read(**kwargs: Any) -> Any:
        path = kwargs.get("relative_path", "")
        agent_name, source = kwargs.get("agent_name", ""), kwargs.get("source", "all")
        if _is_eval_sets_path(path) or _resolves_into_eval_sets(agent_name, source, path):
            return f"Access denied: '{path}' is not available in this ablation mode."
        return await tool.ainvoke(kwargs)

    return StructuredTool.from_function(
        coroutine=read, name=tool.name, description=tool.description, args_schema=tool.args_schema
    )


def apply_ablation(tools: list[BaseTool], ablation: str) -> list[BaseTool]:
    """The optimizer's tools under a tools-ablation configuration."""
    kept = [tool for tool in tools if tool.name not in ABLATION_REMOVED_TOOLS[ablation]]
    if ablation == "full":
        return kept
    return [_block_eval_sets(tool) if tool.name == "read_agent_file" else tool for tool in kept]


def return_tool_errors(tools: list[BaseTool]) -> list[BaseTool]:
    """Return MCP tool errors as tool results so the optimizer can inspect and correct a call."""
    for tool in tools:
        tool.handle_tool_error = True
    return tools


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------

MAX_CONTEXT_CHARS = 600_000
# Keep enough response headroom for local models with 131K-token contexts.
LOCAL_MAX_CONTEXT_CHARS = 400_000
MAX_TOOL_MESSAGE_CHARS = 100_000
CHECKPOINT_FILENAME = "react_checkpoints.sqlite3"
THREAD_STATE_FILENAME = "react_thread.json"


def _utc_now() -> str:
    """Return a stable UTC timestamp for optimizer metadata."""
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
    temporary_path.replace(path)


def _resolve_thread_id(
    output_path: Path, start_round: int, resume: bool | None = None
) -> tuple[str, str]:
    """Return the persistent LangGraph thread and the continuation mode.

    A workspace created before checkpointing has round artifacts but no thread
    state. Its first continuation bootstraps from those artifacts; every later
    continuation resumes the exact persisted LangGraph message state.

    ``resume`` is the caller's intent (``--continue``), kept separate from the
    completed-round index: a run interrupted before round 0 saved its stats has
    ``start_round == 0`` but must still resume its checkpointed thread.
    Defaults to ``start_round > 0``.
    """
    if resume is None:
        resume = start_round > 0
    thread_state_path = output_path / THREAD_STATE_FILENAME
    checkpoint_path = output_path / CHECKPOINT_FILENAME

    if resume and thread_state_path.is_file() and checkpoint_path.is_file():
        try:
            payload = json.loads(thread_state_path.read_text())
            thread_id = payload["thread_id"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid React thread metadata: {thread_state_path}") from exc
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError(f"Invalid React thread ID in {thread_state_path}")
        return thread_id, "checkpoint"

    thread_id = f"react-{uuid4().hex}"
    mode = "artifact_bootstrap" if resume and start_round > 0 else "new"
    _write_json(
        thread_state_path,
        {
            "thread_id": thread_id,
            "created_at": _utc_now(),
            "mode": mode,
        },
    )
    return thread_id, mode


def _msg_chars(msg: BaseMessage) -> int:
    """Estimate the character count of a message."""
    content = msg.content if hasattr(msg, "content") else ""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(str(block)) for block in content)
    return len(str(content))


def _normalize_tool_content(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Flatten MCP text-content blocks into plain strings for model invocation."""
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, list):
            texts = [
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in msg.content
            ]
            msg.content = "\n".join(texts)
    return messages


def _truncate_tool_content(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Apply the per-tool-result character budget before model invocation."""
    bounded: list[BaseMessage] = []
    for msg in messages:
        content = msg.content if isinstance(msg, ToolMessage) else None
        if isinstance(content, str) and len(content) > MAX_TOOL_MESSAGE_CHARS:
            half = MAX_TOOL_MESSAGE_CHARS // 2
            marker = "\n\n[... oversized tool result truncated ...]\n\n"
            content = content[:half] + marker + content[-half:]
            bounded.append(msg.model_copy(update={"content": content}))
        else:
            bounded.append(msg)
    return bounded


def _build_pre_model_hook(max_chars: int = MAX_CONTEXT_CHARS):
    """Build a pre_model_hook that trims old messages to stay within context budget.

    Strategy:
    - Always keep the first message, the protocol (langgraph adds the system prompt itself)
    - Keep as many recent messages as fit within the budget
    - When trimming, summarize what was lost (the first experiment or score line of each run result)
    - Uses `llm_input_messages` so the full history stays in state for tool reference
    """

    def pre_model_hook(state: dict[str, Any]) -> dict[str, Any]:
        messages: list[BaseMessage] = state.get("messages", [])
        if not messages:
            return {"llm_input_messages": messages}

        messages = _truncate_tool_content(_normalize_tool_content(messages))

        total = sum(_msg_chars(m) for m in messages)
        if total <= max_chars:
            return {"llm_input_messages": messages}

        # Always preserve the protocol (the system prompt is not in the state)
        n_preserved = min(1, len(messages))
        preserved = messages[:n_preserved]
        preserved_chars = sum(_msg_chars(m) for m in preserved)
        remaining = messages[n_preserved:]

        # Keep messages from the end until we exhaust the budget
        budget = max_chars - preserved_chars - 2000  # reserve space for summary
        kept: list[BaseMessage] = []
        for msg in reversed(remaining):
            size = _msg_chars(msg)
            if budget - size < 0 and kept:
                break
            kept.insert(0, msg)
            budget -= size
        while kept and isinstance(kept[0], ToolMessage):  # its tool call was trimmed
            kept.pop(0)

        n_trimmed = len(remaining) - len(kept)
        if n_trimmed > 0:
            # Build a summary of trimmed content so the optimizer doesn't lose key info
            trimmed_msgs = remaining[: len(remaining) - len(kept)]
            summary_parts = [
                f"[Context management: {n_trimmed} earlier messages were trimmed to fit context window.",
                "Key information from trimmed messages:",
            ]

            # Extract scores and round info from trimmed tool messages
            for msg in trimmed_msgs:
                content = msg.content if hasattr(msg, "content") else ""
                if not isinstance(content, str):
                    continue
                # Capture score summaries
                if "AGGREGATED SCORES" in content or "Run completed" in content:
                    # Keep the first experiment-name or score line
                    for line in content.split("\n"):
                        if "Experiment:" in line or "runs_mean" in line or "%" in line:
                            summary_parts.append(f"  {line.strip()}")
                            break

            summary_parts.append(
                "Use read_optimizer_log to retrieve saved prompts and scores from previous rounds.]"
            )
            trim_notice = HumanMessage(content="\n".join(summary_parts))
            return {"llm_input_messages": preserved + [trim_notice] + kept}

        return {"llm_input_messages": preserved + kept}

    return pre_model_hook


def _build_post_model_hook(output_dir: str = ""):
    """Build a post_model_hook that answers a text-only response with a reminder to call a tool.

    The reminder asks for the next tool action or the final ``save_optimizer_log`` call. After
    three consecutive text-only responses the hook adds nothing.
    """
    consecutive_text_only = 0
    max_text_only = 3  # bound consecutive text-only turns

    def post_model_hook(state: dict[str, Any]) -> dict[str, Any]:
        nonlocal consecutive_text_only
        messages: list[BaseMessage] = state.get("messages", [])
        if not messages:
            return {}

        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage):
            return {}

        has_tool_calls = bool(getattr(last_msg, "tool_calls", None))
        if has_tool_calls:
            consecutive_text_only = 0
            return {}

        # Count consecutive text-only turns before requesting another tool action.
        consecutive_text_only += 1
        if consecutive_text_only >= max_text_only:
            # Reset the counter and allow termination at the text-only limit.
            consecutive_text_only = 0
            return {}

        # Nudge the model to continue with tool calls
        dir_hint = f'output_dir="{output_dir}", ' if output_dir else ""
        nudge = HumanMessage(
            content=(
                "You produced a text-only response. In this optimization loop, you MUST "
                "call a tool in every response — no exceptions. "
                "If all rounds are complete, you MUST call "
                f'save_optimizer_log({dir_hint}filename="final_summary.md", content=<your summary here>) '
                "RIGHT NOW to save the final summary before terminating. "
                "Do not produce another text response — call the tool."
            )
        )
        return {"messages": [nudge]}

    return post_model_hook


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


def _print_stream_event(node: str, update: dict, step_count: int, elapsed: float) -> None:
    """Print a human-readable progress line for a stream event."""
    messages = update.get("messages", [])
    for msg in messages:
        msg_type = type(msg).__name__

        if msg_type == "AIMessage":
            # Print step header for each agent turn
            print(f"\n--- Step {step_count} ({elapsed:.0f}s) ---")

            # Show any text content
            content = msg.content if hasattr(msg, "content") else ""
            if isinstance(content, str) and content.strip():
                print(f"\n{content}\n")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text.strip():
                            print(f"\n{text}\n")

            # Show tool calls
            tool_calls = getattr(msg, "tool_calls", [])
            for tc in tool_calls:
                name = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
                args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})

                args_preview = str(args)
                if len(args_preview) > 120:
                    args_preview = args_preview[:120] + "..."
                print(f"  -> {name}({args_preview})")

        elif msg_type == "ToolMessage":
            content = msg.content if hasattr(msg, "content") else ""
            if isinstance(content, str):
                first_line = content.split("\n")[0][:120]
                print(f"  <- {first_line}")


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------


def _cancel_on_stop_signals(received: list[signal.Signals]):
    """Cancel the running task on SIGTERM/SIGHUP, recording the signal in ``received``.

    Returns a function that puts the previous handlers back. Does nothing where the event loop
    cannot handle signals (another thread, Windows).
    """
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    previous: dict[signal.Signals, Any] = {}

    def stop(sig: signal.Signals) -> None:
        if received or task is None:  # a second signal stops at once, as it would by default
            loop.remove_signal_handler(sig)
            os.kill(os.getpid(), sig)
            return
        received.append(sig)
        task.cancel()

    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is None:
            continue
        handler = signal.getsignal(sig)
        if handler == signal.SIG_IGN:  # e.g. SIGHUP under nohup: keep ignoring it
            continue
        try:
            loop.add_signal_handler(sig, stop, sig)
        except (NotImplementedError, RuntimeError, ValueError):
            continue
        previous[sig] = handler

    def restore() -> None:
        for sig, handler in previous.items():
            loop.remove_signal_handler(sig)
            if handler is not None:  # None: not set from Python; removal restored the default
                signal.signal(sig, handler)

    return restore


async def run_react_optimizer(
    agent_name: str,
    model: str = "gpt-5-2025-08-07",
    optimizer_model: str = "gpt-5-2025-08-07",
    num_rounds: int = 5,
    output_dir: str | None = None,
    name: str = "react_optimizer",
    loops: str = "",
    temperature: float = 0.0,
    recursion_limit: int = 500,
    max_chars: int = 20_000,
    skill_version: str = DEFAULT_SKILL,
    ablation: str = "full",
    start_round: int = 0,
    resume: bool = False,
) -> dict:
    """Run the ReAct prompt optimizer on an agent.

    Args:
        agent_name: Agent to optimize (must exist in gym or core agents).
        model: Model used for agent eval runs.
        optimizer_model: Model used for the optimizer agent itself.
        num_rounds: Number of optimization rounds (default: 5, as the CLI).
        name: Output directory label (default: react_optimizer).
        skill_version: Skill prompt: full (default), trimmed or tool-free.
        max_chars: Max chars for get_agent_messages output (default: 20000).
        output_dir: Output directory name. Defaults to the CLI layout from
            ``compute_output_dir(agent_name, optimizer_model, model, name, skill=skill_version)``,
            with an ``_ablation_{ablation}`` suffix for score-only and macro.
        loops: Loop spec for eval runs (e.g. "basic:any"). Empty = testbed default.
        temperature: Temperature for the optimizer LLM (default: 0.0).
        ablation: Tools ablation: full (default), score-only or macro (``ABLATION_REMOVED_TOOLS``).
        recursion_limit: Max LangGraph recursion steps (default: 500, as the CLI).
        start_round: First round to run; earlier rounds are immutable.
        resume: Continue the workspace's checkpointed thread (``--continue``),
            even when no round has completed yet.

    Returns:
        Dict with keys: output_dir, duration_s, total_steps.
    """
    if not output_dir:  # named as the CLI names it, so skills and ablations never share one
        output_dir = compute_output_dir(
            agent_name, optimizer_model, model, name, skill=skill_version
        )
        if ablation != "full":
            output_dir += f"_ablation_{ablation}"
    output_path = repo_root() / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    # Read by the MCP server subprocess: the workspace, the trace size limit, and the one agent
    # version (and eval model and loops) this run may evaluate and read results of.
    os.environ[TARGET_ENV] = agent_name
    os.environ[EVAL_MODEL_ENV] = model
    os.environ[EVAL_LOOPS_ENV] = loops
    os.environ["COACH_OUTPUT_DIR"] = output_dir
    init_started_experiments(output_path)
    os.environ["COACH_MAX_CHARS"] = str(max_chars)
    os.environ["COACH_IMMUTABLE_ROUNDS_BEFORE"] = str(start_round)

    thread_id, resume_mode = _resolve_thread_id(
        output_path, start_round, resume=resume or start_round > 0
    )
    session_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    session_metadata_path = output_path / SESSIONS_DIRNAME / f"{session_id}.json"
    metadata: dict[str, Any] = {
        "session_id": session_id,
        "status": "starting",
        "started_at": _utc_now(),
        "agent_name": agent_name,
        "model": model,
        "optimizer_model": optimizer_model,
        "num_rounds": num_rounds,
        "loops": loops,
        "optimizer_temperature": temperature,
        "eval_temperature": os.environ.get("UIPATH_AGENTS_GYM_TEMPERATURE", "1.0"),
        "skill_version": skill_version,
        "ablation": ablation,
        "start_round": start_round,
        "thread_id": thread_id,
        "resume_mode": resume_mode,
    }
    _write_json(session_metadata_path, metadata)
    _write_json(output_path / "run_metadata.json", metadata)

    cost_tracker = CostTrackingCallback()

    def save_metadata(update: dict[str, Any]) -> None:
        metadata.update(update, optimizer_cost=cost_tracker.record.model_dump(exclude={"driver"}))
        _write_json(session_metadata_path, metadata)
        totals = run_totals(output_path)
        _write_json(output_path / "run_metadata.json", {**metadata, **totals})
        if "accounting_error" not in totals:
            print_run_summary(totals)

    # SIGTERM/SIGHUP (kill, a closed terminal) end the session as Ctrl-C does: the run is
    # cancelled, so its duration and optimizer cost are recorded before it exits.
    stop_signals: list[signal.Signals] = []
    restore_signals = _cancel_on_stop_signals(stop_signals)

    start = time.time()
    step_count = 0
    try:
        with tee_to_log(output_dir, "react") as log_file_path:
            metadata["status"] = "running"
            metadata["log_file"] = str(log_file_path.relative_to(repo_root()))
            _write_json(session_metadata_path, metadata)
            _write_json(output_path / "run_metadata.json", metadata)

            print(f"\n{'=' * 60}")
            print("  ReAPO")
            print(f"  Agent:           {agent_name}")
            print(f"  Eval model:      {model}")
            print(f"  Optimizer model: {optimizer_model}")
            print(f"  Rounds:          {num_rounds}")
            print(f"  Ablation:        {ablation}")
            print(f"  Start round:     {start_round}")
            print(f"  Resume mode:     {resume_mode}")
            print(f"  Output:          {output_dir}/")
            print(f"  Log:             {log_file_path.relative_to(repo_root())}")
            print(f"{'=' * 60}\n")

            # Create the optimizer LLM
            llm = get_chat_model(optimizer_model, temperature=temperature)

            # Build the optimization prompt
            user_prompt = build_optimization_prompt(
                agent_name,
                model,
                num_rounds,
                output_dir,
                skill_version,
                loops=loops,
                optimizer_model=optimizer_model,
                start_round=start_round,
            )

            checkpoint_path = output_path / CHECKPOINT_FILENAME
            async with (
                AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer,
                coach_light_session() as tools,
            ):
                # A workspace's checkpoints may come from elsewhere: restore only the types
                # langgraph registers as safe, never arbitrary constructors.
                checkpointer.serde = JsonPlusSerializer(allowed_msgpack_modules=None)
                tools = return_tool_errors(apply_ablation(tools, ablation))
                # Persist the complete ReAct state so --continue retains the same
                # optimizer conversation instead of reconstructing it from summaries.
                agent = create_react_agent(
                    model=llm,
                    tools=tools,
                    prompt=SYSTEM_PROMPT,
                    pre_model_hook=_build_pre_model_hook(
                        LOCAL_MAX_CONTEXT_CHARS
                        if optimizer_model in VllmChatModels
                        or optimizer_model in HuggingFaceChatModels
                        else MAX_CONTEXT_CHARS
                    ),
                    post_model_hook=_build_post_model_hook(output_dir=output_dir),
                    checkpointer=checkpointer,
                )

                config: RunnableConfig = {
                    "recursion_limit": recursion_limit,
                    "configurable": {"thread_id": thread_id},
                    "callbacks": [cost_tracker],
                }
                stream_input: dict[str, list[HumanMessage]] | None = {
                    "messages": [HumanMessage(content=user_prompt)]
                }
                if resume_mode == "checkpoint":
                    snapshot = await agent.aget_state(config)
                    if snapshot.next:
                        # The prior process stopped between graph nodes. Resume the
                        # pending node without injecting a second protocol message.
                        stream_input = None

                async for event in agent.astream(
                    stream_input,
                    config=config,
                    stream_mode="updates",
                ):
                    for node_name, update in event.items():
                        if not isinstance(update, dict):
                            continue
                        elapsed = time.time() - start
                        # Count only agent turns (not tool results or hooks)
                        if node_name == "agent":
                            step_count += 1
                        _print_stream_event(node_name, update, step_count, elapsed)

            duration = time.time() - start
            print(f"\n{'=' * 60}")
            print("  Optimization complete")
            print(f"  Duration: {duration:.0f}s")
            print(f"  Steps:    {step_count}")
            print(f"  Output:   {output_dir}/")
            print(f"  Log:      {log_file_path.relative_to(repo_root())}")
            print(f"{'=' * 60}\n")
    except BaseException as exc:
        duration = time.time() - start
        # The MCP session's task group wraps the error that ended the run.
        while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
            exc = exc.exceptions[0]
        save_metadata(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "duration_s": round(duration, 1),
                "total_steps": step_count,
                "error_type": stop_signals[0].name if stop_signals else type(exc).__name__,
                "error": f"stopped by {stop_signals[0].name}" if stop_signals else str(exc),
            }
        )
        if stop_signals:
            raise SystemExit(128 + stop_signals[0]) from None
        raise
    finally:
        restore_signals()

    save_metadata(
        {
            "status": "completed",
            "finished_at": _utc_now(),
            "duration_s": round(duration, 1),
            "total_steps": step_count,
        }
    )
    return {
        "output_dir": output_dir,
        "duration_s": round(duration, 1),
        "total_steps": step_count,
    }
