"""End-to-end smoke test: one rollout of the seed candidate on calculator.

Run with::

    python -m agents_gym_core.prompt_optimization.gepa.smoke_test

Verifies (a) agent loads, (b) graph compiles, (c) graph invokes,
(d) evaluators score the result. Prints the result.
"""

import asyncio
import logging

from agents_gym_core.hydra_agent.core_agent_registry import CoreAgentRegistry
from agents_gym_core.hydra_agent.evaluation_loader import EvaluationLoader
from agents_gym_core.models.backend import configure_model_resolution
from agents_gym_core.models.chat import OpenAIModels, get_chat_model
from agents_gym_core.prompt_optimization.gepa.rollout import SYSTEM_PROMPT_KEY, rollout_one


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    configure_model_resolution()

    registry = CoreAgentRegistry()
    agent = registry.get_agent("calculator", "base")
    evaluators = EvaluationLoader.get_evaluators(
        agent_name=agent.name,
        directory_provider=registry.get_directory_provider(),
    )
    print(f"Loaded agent: {agent.name} v={agent.version}")
    print(f"Evaluators:   {list(evaluators.keys())}")
    print(f"Datapoints:   {len(agent.evaluation_set.evaluations or [])}")
    print(f"Weights:      {agent.evaluator_weights}")

    dp = (agent.evaluation_set.evaluations or [])[0]
    print(f"\nRolling out on datapoint {dp.name!r} ...")
    print(f"  inputs: {dp.inputs}")

    model = get_chat_model(OpenAIModels.gpt_4o_2024_11_20.name)
    seed = {SYSTEM_PROMPT_KEY: agent.system_prompt}

    result = await rollout_one(
        candidate=seed,
        agent=agent,
        datapoint=dp,
        model=model,
        evaluators=evaluators,
        evaluator_weights=dict(agent.evaluator_weights or {}),
    )

    print(f"\nOutput:           {result.output}")
    print(f"Aggregate score:  {result.aggregated_score:.3f}")
    if result.error:
        print(f"Error:            {result.error}")
    print("Per-evaluator:")
    for o in result.evaluator_outcomes:
        print(f"  {o.evaluator_id}: {o.score:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
