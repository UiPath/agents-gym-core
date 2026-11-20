import argparse
import asyncio
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agents_gym_core.hydra_agent import AgentInit, CoreAgentRegistry, EvaluationLoader
from agents_gym_core.loops import LoopDetails
from agents_gym_core.loops.types import DEFAULT_LOOP_VERSION
from agents_gym_core.models.backend import configure_model_resolution, set_default_backend
from agents_gym_core.models.chat import OpenAIModels
from agents_gym_core.models.embeddings import OpenAIEmbeddingModels
from agents_gym_core.models.utils import ModelBackend, default_backend
from agents_gym_core.project import ProjectDirectoryProvider
from agents_gym_core.reporting.run_report import run_notebook
from agents_gym_core.runtime.run_utils import (
    AggregationMode,
    evaluator_exception_lines,
    get_experiment_path,
    load_cached_evaluator_result,
    save_evaluation_results,
)
from agents_gym_core.runtime.runners.experiment_orchestrator import ExperimentOrchestrator
from agents_gym_core.sdk.project_registry import ProjectAgentRegistry
from agents_gym_core.utils.directory_provider import CoreDirectoryProvider, DirectoryProvider
from agents_gym_core.utils.gym_settings_core import GymSettingsCore, gym_settings_core
from agents_gym_core.utils.gym_types import Agent
from agents_gym_core.utils.trace_utils import info


class TestbedRunnerCore:
    """Testbed CLI; a subclass can override the registry, output directory and argument hooks."""

    __test__ = False  # not a pytest test class despite the name

    def __init__(
        self,
        settings: GymSettingsCore | None = None,
        *,
        directory_provider: DirectoryProvider | None = None,
    ) -> None:
        self.settings = settings or gym_settings_core
        self.directory_provider = directory_provider or CoreDirectoryProvider()
        os.environ["UIPATH_LLM_SERVICE"] = self.settings.UIPATH_LLM_SERVICE
        configure_model_resolution()

    def get_registry(self):
        """Registry the agents are loaded from."""
        registry = (
            ProjectAgentRegistry
            if isinstance(self.directory_provider, ProjectDirectoryProvider)
            else CoreAgentRegistry
        )
        return registry(directory_provider=self.directory_provider)

    def default_output_dir(self) -> Path:
        """Default output directory for experiments."""
        return self.directory_provider.get_project_root() / "experiments"

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Run agent evaluation")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
        parser.add_argument(
            "--agents",
            nargs="+",
            default=["calculator:base"],
            help="List of agent names to run (default: calculator)",
        )
        parser.add_argument(
            "--loops",
            nargs="+",
            default=["basic"],
            help="Loop specs to run, e.g. 'basic:any' or 'planning:system-plan' (default: basic, i.e. basic:auto)",
        )
        parser.add_argument(
            "--models",
            nargs="+",
            type=self._parse_models,
            default=[
                {
                    "model": OpenAIModels.gpt_4o_2024_11_20.name,
                    "embeddings": OpenAIEmbeddingModels.text_embedding_3_large.name,
                }
            ],
            help='Model configs as string or JSON, e.g. --models "gpt-4o-2024-11-20" or \'{"model": "gpt-4o", "embeddings": "text-embedding-3-large"}\'',
        )
        parser.add_argument(
            "--max_datapoints", type=int, default=0, help="Maximum number of datapoints to run"
        )
        parser.add_argument(
            "--datapoints",
            nargs="+",
            default=None,
            help="Filter to specific datapoint names (exact match). Only datapoints whose name matches will be run. When provided, --max_datapoints is ignored.",
        )
        parser.add_argument(
            "--splits",
            nargs="+",
            default=None,
            metavar="SPLIT",
            help="Filter to specific split labels (e.g. --splits train val). Derived from eval-set filename suffixes (_train.json → 'train'). Ignored when --datapoints is provided.",
        )
        parser.add_argument(
            "--max_concurrency",
            type=int,
            default=20,
            help="Maximum number of concurrent tasks (default: 20)",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default=self.default_output_dir(),
            help="Directory to save evaluation results (default: experiments/ in the source checkout, the gym --project project, or the working directory for a wheel install)",
        )
        parser.add_argument(
            "--experiment_name",
            type=str,
            default="experiment",
            help="Name of the experiment (default: experiment)",
        )
        parser.add_argument(
            "--print_trace",
            action="store_true",
            help="Print the trace of the agent execution",
        )
        parser.add_argument(
            "--parallel_tool_calls",
            action="store_true",
            help="Enable parallel tool calls",
        )
        parser.add_argument(
            "--mode",
            type=str,
            choices=[
                "overwrite",
                "fill",
                "fill_rerun_evals",
                "fill_skip_errors",
                "fill_skip_run_errors",
            ],
            default="fill",
            help="Run mode: overwrite (rerun all), fill (only missing/failed), fill_rerun_evals (run missing or errored datapoints, rerun all evals), fill_skip_errors (skip all errors), fill_skip_run_errors (skip agent run errors except model endpoint failures, retry eval errors)",
        )
        parser.add_argument(
            "--num_runs",
            type=int,
            default=1,
            help="Number of times to run each experiment (default: 1). Each run is saved in a separate run_N folder under each datapoint.",
        )
        parser.add_argument(
            "--aggregation_mode",
            type=str,
            choices=[m.value for m in AggregationMode],
            default=AggregationMode.DATAPOINT_RUN.value,
            help="Aggregation mode for evaluation results: evaluator_run (per evaluator then across runs), datapoint_run (per datapoint then across runs), run_datapoint (across runs per datapoint)",
        )
        parser.add_argument(
            "--handle_tool_errors",
            action="store_true",
            help="Return a tool exception to the agent as an error message instead of failing the datapoint (default: off, the run fails)",
        )
        parser.add_argument(
            "--skip_saving_traces",
            action="store_true",
            help="Skip saving execution traces (output.json) to disk to save space",
        )
        parser.add_argument(
            "--run_report",
            action="store_true",
            help="Run the notebook report after the experiment completes",
        )
        parser.add_argument(
            "--model_backend",
            type=ModelBackend,
            choices=list(ModelBackend),
            default=default_backend(),
            help=(
                "Model backend: 'uipath' uses the UiPath LLM Gateway, 'langchain' calls providers "
                "with their own API keys (default: $AGENTS_GYM_LLM_BACKEND, else langchain)"
            ),
        )
        parser.add_argument(
            "--reinit",
            action="store_true",
            help=(
                "Run each agent's init script before starting the experiment, even if "
                "already initialised. Equivalent to 'gym init <agent> --force' for each "
                "agent in --agents. Useful for refreshing downloaded data or regenerating "
                "eval sets."
            ),
        )
        return parser

    async def run(self, argv: list[str] | None = None, *, prog: str | None = None) -> None:
        parser = self.build_parser()
        if prog is not None:
            parser.prog = prog
        args = self._resolve_args(parser.parse_args(argv))

        registry = self.get_registry()
        parsed_agents = self._parse_agents(args.agents, registry=registry)
        parsed_loops = self._parse_loops(args.loops)

        # The run's backend becomes the process default, so judges, tool-internal models and
        # subprocesses resolve on it too, not only the --models entries.
        set_default_backend(args.model_backend)
        for model_config in args.models:
            model_config.setdefault("backend", args.model_backend)
        self._validate_models(args.models)

        if args.reinit:
            for agent_name, agent_version in parsed_agents:
                print(f"[--reinit] initialising '{agent_name}:{agent_version}' ...")
                AgentInit.run_for_agent(agent_name, agent_version, registry, force=True)

        orchestrator = ExperimentOrchestrator(
            agents=parsed_agents,
            registry=registry,
            loops=parsed_loops,
            models=args.models,
            max_concurrency=args.max_concurrency,
            verbose=args.verbose,
            max_datapoints=args.max_datapoints,
            datapoint_names=args.datapoints,
            splits=args.splits,
            print_trace=args.print_trace,
            parallel_tool_calls=args.parallel_tool_calls,
            base_output_dir=Path(args.output_dir),
            experiment_name=args.experiment_name,
            mode=args.mode,
            num_runs=args.num_runs,
            handle_tool_errors=args.handle_tool_errors,
            aggregation_mode=AggregationMode(args.aggregation_mode),
            save_traces=not args.skip_saving_traces,
            settings=self.settings,
        )

        results = await orchestrator.run_async()

        await self._save_all_results(
            loaded_agents=results.get("loaded_agents", []),
            results=results,
            args=args,
        )

        if args.run_report:
            self._run_report(results, args.aggregation_mode)

        if failed := results.get("failed_datapoint_runs"):
            print(f"\n{len(failed)} datapoint runs failed and are not scored:", file=sys.stderr)
            for line in failed[:20]:
                print(f"  {line}", file=sys.stderr)
            if len(failed) > 20:
                print(f"  ... and {len(failed) - 20} more", file=sys.stderr)
            print("Rerun the same command to retry them.", file=sys.stderr)
        if raised := self._evaluator_exceptions(results.get("evaluation_results", {})):
            print(f"\n{len(raised)} evaluations raised and are scored zero:", file=sys.stderr)
            for line in raised[:20]:
                print(f"  {line}", file=sys.stderr)
            if len(raised) > 20:
                print(f"  ... and {len(raised) - 20} more", file=sys.stderr)
            print("Rerun in fill mode to evaluate them again.", file=sys.stderr)
        # An agent that failed to load ends the run with its error, once the others are saved.
        if load_errors := results.get("agent_load_errors"):
            raise load_errors[0]
        if failed or raised:
            raise SystemExit(1)

    def _parse_agents(
        self,
        agents: Iterable[str],
        registry: CoreAgentRegistry | None = None,
    ) -> list[tuple[str, str]]:
        """Parse agent specs into (name, version) pairs, dropping duplicates.

        The first occurrence wins, so ``--agents calculator:base @core`` runs calculator once.

        Formats:
            'calculator'            -> ('calculator', 'base')
            'calculator:base'       -> ('calculator', 'base')
            'calculator:*'          -> all versions of calculator
            '*:base'                -> all agents with base version
            '*:*'                   -> all agents, all versions
            'meeting_management:v2' -> ('meeting_management', 'v2')
            '@fast-benchmark'       -> all agents matching that tag
        """
        parsed: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def _add(name: str, version: str) -> None:
            key = (name, version)
            if key not in seen:
                seen.add(key)
                parsed.append(key)

        for agent_arg in agents:
            # @<tag>: a directory provider's implicit tag or a tag in agent metadata
            if agent_arg.startswith("@"):
                tag = agent_arg[1:]
                if registry is None:
                    raise ValueError(
                        f"Cannot resolve '{agent_arg}': registry is required for tag-based selection"
                    )
                tagged = registry.list_agents_by_tag(tag)
                if not tagged:
                    raise ValueError(f"No agents found with tag '{tag}'")
                info(f"Resolved tag '{tag}'", agents=[a for a, _ in tagged])
                for name, ver in tagged:
                    _add(name, ver)
            # name:version, where either part may be *
            elif ":" in agent_arg:
                agent_name, agent_version = agent_arg.split(":", 1)
                names = (
                    registry.list_all_agents() if agent_name == "*" and registry else [agent_name]
                )
                for name in names:
                    versions = (
                        registry.list_agent_versions(name)
                        if agent_version == "*" and registry
                        else [agent_version]
                    )
                    for v in versions:
                        _add(name, v)
            # A plain name means the base version.
            else:
                _add(agent_arg, "base")
        return parsed

    def _parse_loops(self, loops: Iterable[str]) -> list[tuple[str, str]]:
        """Parse loop specs into (loop_name, version) tuples, dropping duplicates.

        The first occurrence wins, so ``--loops basic basic:auto`` runs the basic loop once.

        Examples:
            'basic:any' -> ('basic', 'any')
            'planning' -> ('planning', 'auto')
        """
        parsed: list[tuple[str, str]] = []
        for loop_arg in loops:
            if ":" in loop_arg:
                loop_name, version = loop_arg.split(":", 1)
            else:
                loop_name, version = loop_arg, DEFAULT_LOOP_VERSION
            if (loop_name, version) not in parsed:
                parsed.append((loop_name, version))
        return parsed

    def _resolve_args(self, args: argparse.Namespace) -> argparse.Namespace:
        """Rewrite parsed arguments before the run; identity in core.

        A subclass whose flag expands into other arguments applies it here,
        once every flag has been parsed.
        """
        return args

    def _validate_models(self, models: list[dict[str, Any]]) -> None:
        """Reject model configs this runner cannot honour; no-op in core.

        Called once ``--model_backend`` has been folded into every config, so a
        subclass sees the effective backend.
        """

    def _parse_models(self, value: str) -> dict[str, Any]:
        """Parse a ``{...}`` JSON model config; any other value becomes ``{"model": value}``."""
        value = value.strip()
        if value.startswith("{") and value.endswith("}"):
            try:
                return json.loads(value)
            except json.JSONDecodeError as error:
                raise argparse.ArgumentTypeError(f"Invalid JSON model argument: {value}") from error
        return {"model": value}

    async def _save_all_results(
        self,
        loaded_agents: list[Agent],
        results: dict[str, Any],
        args: argparse.Namespace,
    ) -> None:
        # Every loop in evaluation_results ran, including an agent's own `loop:` (e.g.
        # dspy-program:<module.Class>) in place of basic, so each gets an evaluation_results.json.
        evaluation_results = results.get("evaluation_results", {})

        registry = self.get_registry()
        for agent in loaded_agents:
            # Keyed by the loaded version, which a versions.yaml entry may set apart from its key.
            agent_key = f"{agent.name}__{agent.version}"
            agent_loop_results = evaluation_results.get(agent_key, {})

            for evaluator_results in agent_loop_results.values():
                loop_details = self._extract_loop_details(evaluator_results)
                if not loop_details:
                    continue

                experiment_dir = get_experiment_path(
                    base_output_dir=Path(args.output_dir),
                    experiment_name=args.experiment_name,
                    agent=agent,
                    loop_details=loop_details,
                )

                loop_results = self._load_loop_results(
                    agent=agent,
                    registry=registry,
                    loop_results=evaluator_results,
                    experiment_dir=experiment_dir,
                    max_datapoints=args.max_datapoints,
                    num_runs=args.num_runs,
                )

                if loop_results:
                    output_file = save_evaluation_results(
                        results=loop_results,
                        agent=agent,
                        loop_details=loop_details,
                        output_dir=experiment_dir,
                    )

                    total_evaluations = sum(len(dp_results) for dp_results in loop_results.values())
                    info(
                        "Evaluation run completed",
                        agent_name=agent.name,
                        loop_key=loop_details.loop_key,
                        total_evaluations=total_evaluations,
                        num_evaluators=len(loop_results),
                        results_file=str(output_file),
                        experiment_dir=str(experiment_dir),
                    )

    def _extract_loop_details(
        self,
        evaluator_results: dict[str, dict[str, Any]],
    ) -> LoopDetails | None:
        """Extract LoopDetails from the first result that has loop_details."""
        for datapoint_results in evaluator_results.values():
            for result_data in datapoint_results.values():
                loop_details_dict = result_data.get("loop_details")
                if loop_details_dict:
                    return LoopDetails.model_validate(loop_details_dict)
        return None

    def _evaluator_exceptions(self, evaluation_results: dict[str, Any]) -> list[str]:
        """One line per evaluation of this invocation whose evaluator raised (saved as score 0), for
        agent runs that did not fail.

        A result loaded from disk is a NumericEvaluationResult, so the evaluations kept from an
        earlier run are not listed.
        """
        return evaluator_exception_lines(evaluation_results)

    def _load_loop_results(
        self,
        agent: Agent,
        registry: CoreAgentRegistry,
        loop_results: dict[str, dict[str, Any]],
        experiment_dir: Path,
        max_datapoints: int,
        num_runs: int = 1,
    ) -> dict[str, dict[str, Any]]:
        """Return ``loop_results`` if non-empty, else read the evaluator result files from disk."""
        if loop_results:
            return loop_results

        evaluators = EvaluationLoader.get_evaluators(
            agent_name=agent.name,
            directory_provider=registry.get_directory_provider(),
        )

        loop_results = {}
        for datapoint_name in [
            dp.name
            for i, dp in enumerate(agent.evaluation_set.evaluations)
            if max_datapoints == 0 or i < max_datapoints
        ]:
            for run_number in range(1, num_runs + 1):
                for evaluator_id, evaluator in evaluators.items():
                    try:
                        result = load_cached_evaluator_result(
                            experiment_dir, datapoint_name, evaluator_id, evaluator, run_number
                        )
                        if result:
                            if evaluator_id not in loop_results:
                                loop_results[evaluator_id] = {}
                            datapoint_key = f"{datapoint_name}:run_{run_number}"
                            result["run_number"] = run_number
                            loop_results[evaluator_id][datapoint_key] = result
                    except Exception:
                        continue
        return loop_results

    def _run_report(self, results: dict[str, Any], aggregation_mode: str) -> None:
        """Execute the report notebook on the run's combined_results.json."""
        saved_files = results.get("saved_files", {})
        combined_results_path = saved_files.get("combined_results")

        if combined_results_path and Path(combined_results_path).exists():
            print("Running notebook report")
            info("Running notebook report", results_path=str(combined_results_path))

            # nbconvert runs the kernel from the notebook's directory, so a relative
            # results path would not open.
            combined_abs = Path(combined_results_path).resolve()
            output_dir = combined_abs.parent
            output_path = str(output_dir / "report_executed.ipynb")

            run_notebook(
                results_paths=[str(combined_abs)],
                output_path=output_path,
                export_html=True,
                aggregation_mode=aggregation_mode,
            )

            info(
                "Report generated",
                notebook_path=output_path,
                html_path=output_path.replace(".ipynb", ".html"),
            )
        else:
            print("Skipping report - no combined_results.json found")
            info("Skipping report - no combined_results.json found")


def main(
    argv: list[str] | None = None,
    *,
    directory_provider: DirectoryProvider | None = None,
    prog: str | None = None,
) -> None:
    """CLI entry point."""
    runner = TestbedRunnerCore(directory_provider=directory_provider)
    asyncio.run(runner.run(argv, prog=prog))


if __name__ == "__main__":
    main()
