"""MCP tools for browsing experiment results and agent messages."""

import glob
import json
import os
import re
from pathlib import Path
from typing import TypedDict

from agents_gym_core.prompt_optimization.mcp_tools.helpers import (
    experiments_dir,
    format_experiment_scores,
    load_json_safe,
    optimization_target,
    started_experiments,
)

# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


class _DpInfo(TypedDict):
    scores: list[float]
    details: list[dict[str, object] | str | None]
    has_error: bool
    error_detail: str


def _resolve_experiment(experiment_name: str) -> tuple[str, Path]:
    """Resolve a logical experiment name to its latest numbered run."""
    exp_dir = experiments_dir() / experiment_name
    if exp_dir.is_dir() or not experiments_dir().is_dir():
        return experiment_name, exp_dir

    pattern = re.compile(rf"^{re.escape(experiment_name)}_(\d+)$")
    matches = [
        (int(match.group(1)), candidate.name, candidate)
        for candidate in experiments_dir().iterdir()
        if candidate.is_dir() and (match := pattern.fullmatch(candidate.name))
    ]
    if not matches:
        return experiment_name, exp_dir
    _, resolved_name, resolved_dir = max(matches)
    return resolved_name, resolved_dir


def _invalid(*names: str) -> str | None:
    for name in names:
        if "\\" in name or ".." in name or name.startswith("/"):
            return f"Invalid name: {name}"
    return None


def _invalid_agent(agent_name: str) -> str | None:
    return f"Invalid agent name: {agent_name}" if "/" in agent_name else _invalid(agent_name)


def _refusal(experiment_name: str, exp_dir: Path) -> str | None:
    """None if this experiment may be read: inside experiments/ and, in an optimization run,
    one the run started itself (so never another version's results, e.g. the test split)."""
    if not exp_dir.resolve().is_relative_to(experiments_dir().resolve()):
        return f"Invalid name: {experiment_name}"
    target = optimization_target()
    if target is None:
        return None
    if experiment_name not in started_experiments():
        return f"Access denied: '{experiment_name}' was not started by this optimization run."
    agent_dir = exp_dir / target[0]
    if agent_dir.is_dir() and any(d.is_dir() and d.name != target[1] for d in agent_dir.iterdir()):
        return f"Access denied: '{experiment_name}' has results for other {target[0]} versions."
    return None


def _readable_experiments() -> list[str]:
    return [
        d.name
        for d in sorted(experiments_dir().iterdir())
        if d.is_dir() and _refusal(d.name, d) is None
    ]


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def list_experiments(experiment_name: str = "") -> str:
    """List available experiment results. Shows agents, models, and scores."""
    if not experiments_dir().is_dir():
        return "No experiments directory found."
    if not experiment_name:
        return _list_all_experiments()
    if err := _invalid(experiment_name):
        return err
    return _inspect_experiment(experiment_name)


def _list_all_experiments() -> str:
    """List all experiments with a brief score summary."""
    experiments = _readable_experiments()
    if not experiments:
        return "No experiments found."

    lines = ["Available experiments:"]
    for exp in experiments:
        exp_dir = experiments_dir() / exp
        agent_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
        lines.append(f"  {exp} ({len(agent_dirs)} agents)")

        scores_file = exp_dir / "final_score_per_agent.json"
        if scores_file.exists():
            scores, err = load_json_safe(scores_file)
            if not err and isinstance(scores, dict):
                lines.extend(format_experiment_scores(scores))

    return "\n".join(lines)


def _inspect_experiment(experiment_name: str) -> str:
    """Detailed view of a single experiment."""
    experiment_name, exp_dir = _resolve_experiment(experiment_name)
    if not exp_dir.is_dir():
        return f"Experiment '{experiment_name}' not found."
    if refused := _refusal(experiment_name, exp_dir):
        return refused

    lines = [f"=== Experiment: {experiment_name} ==="]

    scores_file = exp_dir / "final_score_per_agent.json"
    if scores_file.exists():
        scores, err = load_json_safe(scores_file)
        if err:
            lines.append(f"\n{err}")
        elif isinstance(scores, dict):
            lines.append("\nFinal scores:")
            lines.extend(format_experiment_scores(scores))

    combined_file = exp_dir / "combined_results.json"
    if combined_file.exists():
        combined, err = load_json_safe(combined_file)
        if err:
            lines.append(f"\n{err}")
        elif isinstance(combined, list):
            lines.extend(_format_combined_results(combined))

    return "\n".join(lines)


def _format_combined_results(combined: list[dict]) -> list[str]:
    """Format combined_results.json into display lines."""
    lines = [f"\nCombined results: {len(combined)} entries"]
    agent_counts: dict[str, int] = {}
    agent_errors: dict[str, int] = {}
    for entry in combined:
        agent = entry.get("agent", "unknown")
        agent_counts[agent] = agent_counts.get(agent, 0) + 1
        output = entry.get("agent_output", {})
        if isinstance(output, dict) and output.get("type") == "error":
            agent_errors[agent] = agent_errors.get(agent, 0) + 1
    for agent, count in sorted(agent_counts.items()):
        errors = agent_errors.get(agent, 0)
        err_str = f" ({errors} errors)" if errors else ""
        lines.append(f"  {agent}: {count} results{err_str}")
    return lines


# ---------------------------------------------------------------------------
# Full experiment results
# ---------------------------------------------------------------------------


MAX_FULL_RESULTS_CHARS = 50_000
MAX_FULL_RESULTS_OVERRUN_CHARS = 5_000
MAX_DETAIL_VALUE_CHARS = 500
_STANDARD_DETAIL_FIELDS = {"justification", "expected", "actual"}
_FULL_RESULTS_TRUNCATION_MARKER = (
    "\n\n... [truncated — use get_agent_messages for specific datapoint details]"
)
MAX_FULL_RESULTS_MCP_CHARS = (
    MAX_FULL_RESULTS_CHARS
    + MAX_FULL_RESULTS_OVERRUN_CHARS
    + len(_FULL_RESULTS_TRUNCATION_MARKER)
    + 1_000
)


def _truncate_full_results(
    result: str,
    failure_details_prefix_length: int | None,
    failure_detail_blocks: list[str],
) -> str:
    """Truncate at the size limit, finishing a failure-detail block if needed.

    The soft limit may be exceeded by at most the remainder of one datapoint's
    existing detail block. This keeps evaluator metrics from leaving its
    Expected/Actual lines cut off at the boundary.
    """
    if len(result) <= MAX_FULL_RESULTS_CHARS:
        return result

    cutoff = MAX_FULL_RESULTS_CHARS
    if failure_details_prefix_length is not None and cutoff >= failure_details_prefix_length:
        previous_block_end = failure_details_prefix_length
        for block in failure_detail_blocks:
            block_start = previous_block_end + 2
            block_end = block_start + len(block)
            if cutoff < block_start:
                cutoff = previous_block_end
                break
            if cutoff < block_end:
                cutoff = (
                    block_end
                    if block_end - MAX_FULL_RESULTS_CHARS <= MAX_FULL_RESULTS_OVERRUN_CHARS
                    else previous_block_end
                )
                break
            previous_block_end = block_end
        else:
            cutoff = len(result)

    if cutoff == len(result):
        return result
    return result[:cutoff] + _FULL_RESULTS_TRUNCATION_MARKER


def get_full_results(experiment_name: str, agent_name: str = "") -> str:
    """Get the full results from an experiment: scores and per-datapoint details, worst-first.

    If agent_name is provided, filters results to that agent only.
    """
    if err := _invalid(experiment_name):
        return err
    experiment_name, exp_dir = _resolve_experiment(experiment_name)
    if not exp_dir.is_dir():
        available = _readable_experiments() if experiments_dir().is_dir() else []
        return (
            f"Experiment '{experiment_name}' not found. Available: {', '.join(available) or 'none'}"
        )
    if refused := _refusal(experiment_name, exp_dir):
        return refused

    lines = [f"=== Full Results: {experiment_name} ==="]

    # Aggregated scores with agent:version and CI
    scores_file = exp_dir / "final_score_per_agent.json"
    if scores_file.exists():
        scores, err = load_json_safe(scores_file)
        if err:
            lines.append(f"\n{err}")
        elif isinstance(scores, dict):
            lines.append("\n## Scores")
            lines.extend(format_experiment_scores(scores))

    # Per-datapoint results aggregated and sorted worst-first
    combined_file = exp_dir / "combined_results.json"
    if not combined_file.exists():
        lines.append("\nNo combined_results.json found.")
        return "\n".join(lines)

    combined, err = load_json_safe(combined_file)
    if err:
        lines.append(f"\n{err}")
        return "\n".join(lines)

    if not isinstance(combined, list):
        return "\n".join(lines)

    if agent_name:
        # Try exact match first, then fall back to base name (before ':') for versioned agents
        combined = [e for e in combined if e.get("agent") == agent_name]
        if not combined and ":" in agent_name:
            base_name = agent_name.split(":")[0]
            combined_all, _ = load_json_safe(combined_file)
            if isinstance(combined_all, list):
                combined = [e for e in combined_all if e.get("agent") == base_name]

    # Aggregate by datapoint
    by_dp: dict[str, _DpInfo] = {}
    for entry in combined:
        dp = entry.get("datapoint_name", "?")
        if dp not in by_dp:
            by_dp[dp] = {"scores": [], "details": [], "has_error": False, "error_detail": ""}
        score = entry.get("score")
        if isinstance(score, (int, float)):
            by_dp[dp]["scores"].append(score)
        out = entry.get("agent_output", {})
        if isinstance(out, dict) and out.get("type") == "error":
            by_dp[dp]["has_error"] = True
            by_dp[dp]["error_detail"] = (
                f"{out.get('code', '')} - {out.get('title', '')}: {out.get('details', '')}"
            )
        by_dp[dp]["details"].append(entry.get("details", {}))

    # Sort worst-first
    sorted_dps = sorted(
        by_dp.items(),
        key=lambda x: (sum(x[1]["scores"]) / len(x[1]["scores"])) if x[1]["scores"] else 0,
    )

    lines.append(f"\n## Per-Datapoint Scores ({len(sorted_dps)} datapoints, worst first)")
    for dp, info in sorted_dps:
        avg = sum(info["scores"]) / len(info["scores"]) if info["scores"] else 0
        err_tag = " [ERROR]" if info["has_error"] else ""
        status = "PASS" if avg >= 0.95 else "FAIL"
        lines.append(f"  {dp}: {avg * 100:.1f}% {status}{err_tag}")

    # Failure details
    failure_details_prefix_length = None
    failure_detail_blocks: list[str] = []
    failing = [
        (dp, info)
        for dp, info in sorted_dps
        if not info["scores"] or sum(info["scores"]) / len(info["scores"]) < 0.95
    ]
    if failing:
        lines.append(f"\n## Failure Details ({len(failing)} failing datapoints)")
        failure_details_prefix_length = len("\n".join(lines))
        for dp, info in failing:
            block_lines = [f"  ### {dp}"]
            if info.get("error_detail"):
                block_lines.append(f"    Error: {info['error_detail'][:500]}")
            for det in info["details"]:
                block_lines.extend(_format_evaluator_details(det, indent="    "))
            failure_detail_blocks.append("\n".join(block_lines))

    result = "\n".join(lines)
    if failure_detail_blocks:
        result += "".join(f"\n\n{block}" for block in failure_detail_blocks)
    return _truncate_full_results(result, failure_details_prefix_length, failure_detail_blocks)


# ---------------------------------------------------------------------------
# Per-agent experiment results
# ---------------------------------------------------------------------------


def get_experiment_results(experiment_name: str, agent_name: str, datapoint_name: str = "") -> str:
    """Get detailed experiment results for an agent."""
    if err := _invalid(experiment_name) or _invalid_agent(agent_name):
        return err
    experiment_name, exp_dir = _resolve_experiment(experiment_name)
    if refused := _refusal(experiment_name, exp_dir):
        return refused
    agent_dir = exp_dir / agent_name
    if not agent_dir.is_dir() and ":" in agent_name:
        # Versioned agents store results under the base name (before ':')
        agent_dir = exp_dir / agent_name.split(":")[0]
    if not agent_dir.is_dir():
        return f"Agent '{agent_name}' not found in experiment '{experiment_name}'."

    datapoints = _collect_datapoints(agent_dir)
    if not datapoints:
        return f"No results found for agent '{agent_name}'."

    if datapoint_name:
        return _format_datapoint_detail(agent_name, datapoint_name, datapoints)
    return _format_datapoint_summary(agent_name, experiment_name, datapoints)


def _collect_datapoints(agent_dir: Path) -> dict[str, list[dict]]:
    """Collect all result files grouped by datapoint name."""
    datapoints: dict[str, list[dict]] = {}
    for rf in agent_dir.rglob("results/*.json"):
        try:
            data = json.loads(rf.read_text())
            dp_name = data.get("datapoint_name", rf.parent.parent.name)
            datapoints.setdefault(dp_name, []).append(data)
        except Exception:
            pass
    return datapoints


def _extract_details_dict(details: dict[str, object] | str | None) -> dict[str, object]:
    """Coerce a details value to a dict regardless of whether it's a str, dict, or None.

    Evaluators are expected to return a BaseEvaluatorJustification (serialised to dict
    by EvaluationResultDto), but older or third-party evaluators may return a plain
    string. In that case we wrap it under 'justification' so callers always get a dict.
    """
    if isinstance(details, dict):
        return details.copy()
    if isinstance(details, str) and details:
        return {"justification": details}
    return {}


def _render_detail_value(value: object, max_chars: int | None = MAX_DETAIL_VALUE_CHARS) -> str:
    """Render one evaluator detail compactly while preserving structured values."""
    if isinstance(value, str):
        rendered = value
    else:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    if max_chars is not None and len(rendered) > max_chars:
        return rendered[:max_chars] + "... [truncated]"
    return rendered


def _escape_noncanonical_detail_text(value: str) -> str:
    """Keep free-form evaluator text from looking like canonical Expected/Actual lines."""
    return (
        value.replace("\r", r"\r")
        .replace("\n", r"\n")
        .replace("Expected:", r"Expected\u003a")
        .replace("Actual:", r"Actual\u003a")
    )


def _format_evaluator_details(
    details: dict[str, object] | str | None,
    indent: str = "",
    *,
    preserve_justification: bool = False,
) -> list[str]:
    """Format every field emitted by an evaluator's BaseEvaluatorJustification."""
    parsed = _extract_details_dict(details)
    if not parsed:
        return []

    lines: list[str] = []
    custom_fields = [
        (key, value) for key, value in parsed.items() if key not in _STANDARD_DETAIL_FIELDS
    ]
    # The PUPA side experiment: the score without its quality and leakage components.
    if os.environ.get("COACH_HIDE_EVALUATOR_FIELDS") == "1":
        custom_fields = []

    # Keep structured metrics first so the failure components are easy to scan.
    if custom_fields:
        lines.append(f"{indent}Evaluator details:")
        for key, value in custom_fields:
            safe_key = _escape_noncanonical_detail_text(_render_detail_value(key))
            safe_value = _escape_noncanonical_detail_text(_render_detail_value(value))
            lines.append(f"{indent}  {safe_key}: {safe_value}")

    for key, label in (
        ("justification", "Justification"),
        ("expected", "Expected"),
        ("actual", "Actual"),
    ):
        value = parsed.get(key)
        if value not in (None, ""):
            max_chars = (
                None
                if key == "justification" and preserve_justification
                else MAX_DETAIL_VALUE_CHARS
            )
            rendered = _render_detail_value(value, max_chars=max_chars)
            if key == "justification":
                rendered = _escape_noncanonical_detail_text(rendered)
            lines.append(f"{indent}{label}: {rendered}")

    return lines


def _format_datapoint_detail(
    agent_name: str, datapoint_name: str, datapoints: dict[str, list[dict]]
) -> str:
    """Format detailed results for a single datapoint."""
    if datapoint_name not in datapoints:
        available = ", ".join(sorted(datapoints.keys()))
        return f"Datapoint '{datapoint_name}' not found. Available: {available}"

    results = datapoints[datapoint_name]
    lines = [f"=== {agent_name} / {datapoint_name} ==="]
    for r in results:
        lines.append(f"\nEvaluator: {r.get('evaluator_id', '?')}")
        lines.append(f"Score: {r.get('score', 'N/A')}")
        lines.extend(_format_evaluator_details(r.get("details"), preserve_justification=True))
        output = r.get("agent_output", {})
        if isinstance(output, dict) and output.get("type") == "error":
            lines.append(f"ERROR: {output.get('code', '')} - {output.get('title', '')}")
            lines.append(f"  {output.get('details', '')}")
    return "\n".join(lines)


def _format_datapoint_summary(
    agent_name: str, experiment_name: str, datapoints: dict[str, list[dict]]
) -> str:
    """Format a summary of all datapoints with average scores."""
    lines = [f"=== {agent_name} in {experiment_name} ({len(datapoints)} datapoints) ==="]
    total_scores: list[float] = []

    for dp_name in sorted(datapoints.keys()):
        results = datapoints[dp_name]
        scores = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]
        avg_score = sum(scores) / len(scores) if scores else 0
        has_error = any(
            isinstance(r.get("agent_output"), dict) and r["agent_output"].get("type") == "error"
            for r in results
        )
        err_str = " [ERROR]" if has_error else ""
        lines.append(f"  {dp_name}: {avg_score * 100:.1f}%{err_str} ({len(results)} evaluations)")
        total_scores.extend(scores)

    if total_scores:
        overall = sum(total_scores) / len(total_scores)
        lines.append(f"\nOverall average: {overall * 100:.1f}% ({len(total_scores)} evaluations)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent messages (reads output.json — OTLP trace format)
# ---------------------------------------------------------------------------


# COACH_MAX_CHARS, set by both optimizers from --max-chars.
_DEFAULT_MAX_MESSAGES_CHARS = 20_000
_MAX_MESSAGES_CHARS = int(os.environ.get("COACH_MAX_CHARS", _DEFAULT_MAX_MESSAGES_CHARS))

# Per-span truncation limits
_MSG_TRUNCATE = 1000  # per message content
_TOOL_TRUNCATE = 1500  # per tool output
_FINAL_TRUNCATE = 4000  # final agent output (larger — often structured JSON)


def get_agent_messages(
    experiment_name: str, agent_name: str, datapoint_name: str, run_number: int = 1
) -> str:
    """Get the full message history (conversation) from an agent run."""
    if err := _invalid(experiment_name):
        return err
    if "\\" in datapoint_name or datapoint_name in {"", ".", ".."}:
        return f"Invalid datapoint_name: {datapoint_name}"
    if err := _invalid_agent(agent_name):
        return err

    experiment_name, exp_dir = _resolve_experiment(experiment_name)
    if refused := _refusal(experiment_name, exp_dir):
        return refused
    agent_dir = exp_dir / agent_name
    if not agent_dir.is_dir() and ":" in agent_name:
        # Versioned agents store results under the base name (before ':')
        agent_dir = exp_dir / agent_name.split(":")[0]
    if not agent_dir.is_dir():
        return f"Agent '{agent_name}' not found in experiment '{experiment_name}'."

    output_file = _find_output_file(agent_dir, datapoint_name, run_number)
    if not output_file:
        return f"No output.json found for {agent_name}/{datapoint_name}/run_{run_number}"

    try:
        content = output_file.read_text()
        if not content.strip():
            return "Empty output.json"
        spans = _parse_output_json(content)
    except Exception as e:
        return f"Error reading output.json: {e}"

    if not spans:
        return f"No spans found in {output_file}"

    llm_count = sum(1 for s in spans if s.get("span_kind") == "LLM")
    tool_count = sum(1 for s in spans if s.get("span_kind") == "TOOL")

    lines = [
        f"=== Messages: {agent_name} / {datapoint_name} (run {run_number}) ===",
        f"Spans: {len(spans)} total ({llm_count} LLM, {tool_count} tool)",
        f"Raw trace size: {len(content)} chars",
        "",
    ]

    step = 0
    for span in spans:
        kind = span.get("span_kind", "")
        name = span.get("name", "")

        if kind == "LLM":
            step += 1
            model = span.get("model", "")
            tokens = span.get("token_usage", {})
            duration = span.get("duration_ms", "")
            header = f"\n## LLM Call {step}"
            if model:
                header += f" ({model})"
            if tokens:
                header += f" [tokens: {tokens.get('prompt', '?')}+{tokens.get('completion', '?')}={tokens.get('total', '?')}]"
            if duration:
                header += f" [{duration}ms]"
            lines.append(header)

            for msg in span.get("input_messages", []):
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "?")
                msg_content = str(msg.get("content", ""))
                if len(msg_content) > _MSG_TRUNCATE:
                    msg_content = msg_content[:_MSG_TRUNCATE] + "... [truncated]"
                lines.append(f"  [{role}]: {msg_content}")
                for tc in msg.get("tool_calls", []):
                    if isinstance(tc, dict):
                        args_str = str(tc.get("args", ""))
                        if len(args_str) > 500:
                            args_str = args_str[:500] + "..."
                        lines.append(f"    -> tool_call: {tc.get('name', '?')}({args_str})")

            out = span.get("output_text", "")
            if out:
                if len(out) > _MSG_TRUNCATE:
                    out = out[:_MSG_TRUNCATE] + "... [truncated]"
                lines.append(f"  output: {out}")

            for tc in span.get("tool_calls", []):
                if isinstance(tc, dict):
                    args_str = str(tc.get("args", ""))
                    if len(args_str) > 500:
                        args_str = args_str[:500] + "..."
                    lines.append(f"  -> {tc.get('name', '?')}({args_str})")

        elif kind == "TOOL":
            out = span.get("output_text", "")
            if out:
                if len(out) > _TOOL_TRUNCATE:
                    out = out[:_TOOL_TRUNCATE] + "... [truncated]"
                lines.append(f"  <- {name}: {out}")

    # Final agent output (terminate span)
    lines.append("\n## Final Output")
    for span in spans:
        if span.get("name") == "terminate":
            out = span.get("output_text", "")
            if len(out) > _FINAL_TRUNCATE:
                out = out[:_FINAL_TRUNCATE] + "... [truncated]"
            lines.append(f"  {out}")
            break
    else:
        lines.append("  (no terminate span found — may have hit MAX_ITERATIONS)")

    # Issues detection
    issues = []
    if len(content) > 100_000:
        issues.append(f"Very large trace ({len(content)} chars) — likely looping")
    tool_counts: dict[str, int] = {}
    for span in spans:
        for tc in span.get("tool_calls", []):
            if isinstance(tc, dict):
                n = tc.get("name", "")
                tool_counts[n] = tool_counts.get(n, 0) + 1
    for tn, count in tool_counts.items():
        if count > 3:
            issues.append(f"Tool '{tn}' called {count} times (possible repetition)")
    if issues:
        lines.append("\n## Issues Detected")
        for issue in issues:
            lines.append(f"  - {issue}")

    result = "\n".join(lines)
    max_chars = int(os.environ.get("COACH_MAX_CHARS", _MAX_MESSAGES_CHARS))
    if len(result) > max_chars:
        result = (
            result[:max_chars]
            + "\n\n... [truncated — use get_agent_messages for a specific datapoint]"
        )
    return result


def _find_output_file(agent_dir: Path, datapoint_name: str, run_number: int) -> Path | None:
    """Find the output.json file for a given datapoint and run."""
    name = glob.escape(datapoint_name.replace("/", "_"))  # "?" or "[" in a name is literal
    for pattern in [
        f"**/{name}/run_{run_number}/output.json",
        f"**/{name}/output.json",
    ]:
        matches = list(agent_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


# ---------------------------------------------------------------------------
# OTLP output.json parsing
# ---------------------------------------------------------------------------


def _otlp_attrs_to_dict(attributes: list[dict]) -> dict[str, object]:
    """Convert OTLP attribute array to a flat dict."""
    result: dict[str, object] = {}
    for attr in attributes:
        key = attr.get("key", "")
        value_obj = attr.get("value", {})
        if "string_value" in value_obj:
            result[key] = value_obj["string_value"]
        elif "int_value" in value_obj:
            result[key] = int(value_obj["int_value"])
        elif "double_value" in value_obj:
            result[key] = float(value_obj["double_value"])
        elif "bool_value" in value_obj:
            result[key] = value_obj["bool_value"]
    return result


def _parse_output_json(content: str) -> list[dict]:
    """Parse output.json content (OTLP JSON array) and extract meaningful spans."""
    if not content.strip():
        return []

    otlp_data = json.loads(content)
    # output.json is a JSON array of OTLP resource span objects
    if isinstance(otlp_data, dict):
        otlp_data = [otlp_data]

    spans: list[dict] = []
    for obj in otlp_data:
        for resource_span in obj.get("resource_spans", []):
            for scope_span in resource_span.get("scope_spans", []):
                for span_data in scope_span.get("spans", []):
                    parsed = _parse_otlp_span(span_data)
                    if parsed:
                        spans.append(parsed)
    return spans


def _parse_otlp_span(span_data: dict) -> dict | None:
    """Parse a single OTLP span into a readable dict.

    Only returns LLM, TOOL, and key CHAIN spans (like 'terminate').
    """
    attrs = _otlp_attrs_to_dict(span_data.get("attributes", []))
    span_kind = str(attrs.get("openinference.span.kind", ""))
    span_name = span_data.get("name", "")

    if span_kind not in ("LLM", "TOOL") and span_name not in ("terminate", "LangGraph"):
        return None

    result: dict[str, object] = {
        "span_kind": span_kind or "CHAIN",
        "name": span_name,
    }

    model = attrs.get("llm.model_name", "")
    if model:
        result["model"] = model

    input_raw = attrs.get("input.value")
    if isinstance(input_raw, str) and span_kind == "LLM":
        result["input_messages"] = _extract_messages_from_input_value(input_raw)

    output_raw = attrs.get("output.value")
    if isinstance(output_raw, str):
        result.update(_extract_output_value(output_raw, span_kind))

    token_usage: dict[str, int] = {}
    for field in ("llm.token_count.prompt", "llm.token_count.completion", "llm.token_count.total"):
        val = attrs.get(field)
        if isinstance(val, int):
            token_usage[field.split(".")[-1]] = val
    if token_usage:
        result["token_usage"] = token_usage

    start = span_data.get("start_time_unix_nano")
    end = span_data.get("end_time_unix_nano")
    if start and end:
        try:
            result["duration_ms"] = round((int(end) - int(start)) / 1_000_000, 2)
        except (ValueError, TypeError):
            pass

    return result


def _extract_messages_from_input_value(raw: str) -> list[dict]:
    """Extract conversation messages from the input.value attribute."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return [{"role": "raw", "content": raw[:500]}]

    if isinstance(parsed, dict) and "messages" in parsed:
        messages_list = parsed["messages"]
        if isinstance(messages_list, list) and messages_list:
            msgs = messages_list[0] if isinstance(messages_list[0], list) else messages_list
            return _parse_langchain_messages(msgs)

    return [{"role": "raw", "content": str(parsed)[:500]}]


def _parse_langchain_messages(msgs: list) -> list[dict]:
    """Parse LangChain-style message objects into simple role/content dicts."""
    result: list[dict] = []
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        kwargs = msg.get("kwargs", {})
        content = kwargs.get("content", "")
        msg_id = msg.get("id", [])

        role = "unknown"
        if isinstance(msg_id, list) and msg_id:
            class_name = msg_id[-1]
            if "System" in class_name:
                role = "system"
            elif "Human" in class_name:
                role = "user"
            elif "AI" in class_name:
                role = "assistant"
            elif "Tool" in class_name:
                role = "tool"

        entry: dict[str, object] = {"role": role, "content": content}
        tool_calls = kwargs.get("tool_calls", [])
        if tool_calls:
            entry["tool_calls"] = [
                {"name": tc.get("name", "?"), "args": tc.get("args", {})}
                for tc in tool_calls
                if isinstance(tc, dict)
            ]
        result.append(entry)
    return result


def _extract_output_value(raw: str, span_kind: str) -> dict:
    """Extract meaningful output from output.value attribute."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"output_text": raw[:500]}

    if not isinstance(parsed, dict):
        return {"output_text": str(parsed)[:500]}

    result: dict[str, object] = {}

    if span_kind == "LLM":
        generations = parsed.get("generations", [])
        if generations and isinstance(generations[0], list):
            for gen in generations[0]:
                if not isinstance(gen, dict):
                    continue
                msg = gen.get("message", {})
                if isinstance(msg, dict):
                    kwargs = msg.get("kwargs", {})
                    content = kwargs.get("content", "")
                    if content:
                        result["output_text"] = content
                    tool_calls = kwargs.get("tool_calls", [])
                    if tool_calls:
                        result["tool_calls"] = [
                            {"name": tc.get("name", "?"), "args": tc.get("args", {})}
                            for tc in tool_calls
                            if isinstance(tc, dict)
                        ]
    else:
        result["output_text"] = str(parsed)[:1000]

    return result
