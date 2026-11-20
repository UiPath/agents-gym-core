"""GEPA retains execution settings while inheriting the base evaluation selection."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import DictConfig, OmegaConf
from uipath.eval.models.evaluation_set import EvaluationItem, EvaluationSet

from agents_gym_core.hydra_agent.core_agent_registry import CoreAgentRegistry
from agents_gym_core.prompt_optimization.gepa.writeback import save_evolved_version
from agents_gym_core.utils.directory_provider import CoreDirectoryProvider


@pytest.fixture
def toy_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CoreAgentRegistry:
    agent_dir = tmp_path / "toy"
    agent_dir.mkdir()
    evaluations = agent_dir / "evaluations" / "eval-sets"
    evaluations.mkdir(parents=True)
    for label in ("default", "train"):
        evaluation_set = EvaluationSet(
            id=label,
            name=label,
            evaluations=[
                EvaluationItem(
                    id=label, name=label, inputs={"query": label}, evaluationCriterias={}
                )
            ],
        )
        (evaluations / f"{label}.json").write_text(evaluation_set.model_dump_json(by_alias=True))
    OmegaConf.save(
        OmegaConf.create(
            {
                "version": "base",
                "agent": {
                    "system_prompt": "base system",
                    "user_prompt": "base {query}",
                    "input_schema": {"fields": {"query": {"type": "str"}}},
                    "output_schema": {"fields": {"answer": {"type": "str"}}},
                    "evaluation_sets": ["core::evaluations/eval-sets/default.json"],
                },
            }
        ),
        agent_dir / "agent.yaml",
    )
    OmegaConf.save(
        OmegaConf.create(
            {
                "seed": {
                    "version": "seed",
                    "loop": "basic:auto",
                    "agent": {
                        "system_prompt": "seed system",
                        "user_prompt": "seed {query}",
                        "evaluator_weights": {"check": 2.0},
                        "evaluation_sets": ["core::evaluations/eval-sets/train.json"],
                        "output_schema": {
                            "fields": {"answer": {"type": "str", "description": "seed answer"}}
                        },
                    },
                }
            }
        ),
        agent_dir / "versions.yaml",
    )
    monkeypatch.setattr(CoreDirectoryProvider, "get_agents_dir", lambda self: tmp_path)
    return CoreAgentRegistry()


@pytest.mark.parametrize("variant", ["gepa", "gepa_dspy"])
def test_gepa_cli_writeback_preserves_seed_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, toy_registry: CoreAgentRegistry, variant: str
) -> None:
    cli = importlib.import_module(f"agents_gym_core.prompt_optimization.{variant}.cli")
    seed = toy_registry.get_agent("toy", "seed")
    assert [case.name for case in seed.evaluation_set.evaluations] == ["train"]
    versions_path = tmp_path / "toy" / "versions.yaml"
    before = OmegaConf.load(versions_path)
    monkeypatch.setattr(cli, "configure_model_resolution", lambda: None)
    monkeypatch.setattr(cli, "get_chat_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli.time, "strftime", lambda *args: "test")
    if variant == "gepa_dspy":
        monkeypatch.setattr(
            cli,
            "load_agent_and_evaluators",
            lambda *args: (toy_registry, seed, {"check": object()}),
        )
        monkeypatch.setattr(cli, "make_dspy_lm", lambda *args, **kwargs: object())
        monkeypatch.setattr(cli, "optimize_dspy", lambda **kwargs: ("evolved system", object()))
    else:

        async def optimize(**kwargs: object) -> tuple:
            return {"system_prompt": "evolved system"}, SimpleNamespace(
                best_idx=lambda: 1,
                candidates=[{}, {}],
                train_mean=lambda index: 0.8,
                val_scores=[0.4, 0.8],
            )

        monkeypatch.setattr(cli, "CoreAgentRegistry", lambda: toy_registry)
        monkeypatch.setattr(
            cli.EvaluationLoader, "get_evaluators", lambda **kwargs: {"check": object()}
        )
        monkeypatch.setattr(cli, "optimize", optimize)
    monkeypatch.setattr(
        sys, "argv", [variant, "toy", "--version", "seed", "--run-dir", str(tmp_path / "runs")]
    )

    cli.main()

    evolved = toy_registry.get_agent("toy", f"{variant}_test")
    assert evolved.system_prompt == "evolved system"
    assert evolved.user_prompt == seed.user_prompt == "seed {query}"
    assert evolved.evaluator_weights == seed.evaluator_weights == {"check": 2.0}
    assert evolved.loop == seed.loop == "basic:auto"
    assert evolved.output_schema.model_json_schema() == seed.output_schema.model_json_schema()
    assert [case.name for case in evolved.evaluation_set.evaluations] == ["default"]
    after = OmegaConf.load(versions_path)
    assert isinstance(after, DictConfig)
    assert "evaluation_sets" not in after[f"{variant}_test"].agent
    assert after.seed == before.seed


def test_writeback_without_source_version_keeps_base_configuration(
    toy_registry: CoreAgentRegistry,
) -> None:
    save_evolved_version(
        directory_provider=toy_registry.get_directory_provider(),
        agent_name="toy",
        version_name="evolved",
        evolved_system_prompt="evolved system",
    )
    evolved = toy_registry.get_agent("toy", "evolved")
    assert evolved.system_prompt == "evolved system"
    assert evolved.user_prompt == "base {query}"
    assert [case.name for case in evolved.evaluation_set.evaluations] == ["default"]
