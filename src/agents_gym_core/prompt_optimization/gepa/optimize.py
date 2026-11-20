"""Native GEPA variant with greedy candidate selection.

Per iteration:
  1. Pick the best candidate so far (greedy by mean train score).
  2. Sample a minibatch of train datapoints.
  3. Roll out the parent on the minibatch; record per-evaluator feedback.
  4. Ask the proposer LLM for a new instruction.
  5. Roll out the new candidate on the same minibatch.
  6. If the new candidate scores at least as high on the minibatch, full-eval
     it on train+val and add to state. Otherwise discard.

Candidate selection ranks prompts by mean training score, with validation scores breaking
ties. The paper's enterprise GEPA baseline runs upstream ``dspy.GEPA`` through ``gepa_dspy``.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from uipath.eval.evaluators import BaseEvaluator

from agents_gym_core.loops.types import DEFAULT_LOOP_VERSION
from agents_gym_core.prompt_optimization.gepa.feedback import feedback_for_rollout
from agents_gym_core.prompt_optimization.gepa.proposer import propose_new_instruction
from agents_gym_core.prompt_optimization.gepa.rollout import (
    SYSTEM_PROMPT_KEY,
    Candidate,
    rollout_batch,
)
from agents_gym_core.prompt_optimization.gepa.state import OptimizationState
from agents_gym_core.utils.gym_types import Agent

logger = logging.getLogger(__name__)


@dataclass
class OptimizeConfig:
    num_iters: int = 10
    minibatch_size: int = 3
    train_fraction: float = 0.8
    seed: int = 0
    max_concurrency: int = 8
    loop_name: str = "basic"
    loop_version: str = DEFAULT_LOOP_VERSION


async def optimize(
    *,
    agent: Agent,
    model: BaseChatModel,
    proposer_lm: BaseChatModel,
    evaluators: dict[str, BaseEvaluator],
    config: OptimizeConfig | None = None,
    log_dir: Path | None = None,
) -> tuple[Candidate, OptimizationState]:
    """Run GEPA on ``agent`` and return ``(best_candidate, full_state)``.

    Only ``system_prompt`` is mutated. ``model`` is the agent under test;
    ``proposer_lm`` is the LLM that proposes new instructions.
    """
    cfg = config or OptimizeConfig()
    rng = random.Random(cfg.seed)

    all_dps = list(agent.evaluation_set.evaluations or [])
    if not all_dps:
        raise ValueError(f"Agent {agent.name!r} has no evaluation datapoints")

    rng.shuffle(all_dps)
    n_train = max(1, int(len(all_dps) * cfg.train_fraction))
    train_dps = all_dps[:n_train]
    val_dps = all_dps[n_train:]
    logger.info(
        "optimize: agent=%s train=%d val=%d evaluators=%d",
        agent.name,
        len(train_dps),
        len(val_dps),
        len(evaluators),
    )

    weights = dict(agent.evaluator_weights or {})
    seed_candidate: Candidate = {SYSTEM_PROMPT_KEY: agent.system_prompt}

    state = OptimizationState()

    # Full-eval the seed
    logger.info("evaluating seed candidate on full train/val ...")
    seed_train = await rollout_batch(
        seed_candidate,
        agent,
        train_dps,
        model,
        evaluators,
        weights,
        max_concurrency=cfg.max_concurrency,
        loop_name=cfg.loop_name,
        loop_version=cfg.loop_version,
    )
    seed_train_scores = [r.aggregated_score for r in seed_train]
    seed_val_score: float | None = None
    if val_dps:
        seed_val = await rollout_batch(
            seed_candidate,
            agent,
            val_dps,
            model,
            evaluators,
            weights,
            max_concurrency=cfg.max_concurrency,
            loop_name=cfg.loop_name,
            loop_version=cfg.loop_version,
        )
        seed_val_score = sum(r.aggregated_score for r in seed_val) / len(seed_val)
    state.add(
        seed_candidate, parent_idx=None, train_scores=seed_train_scores, val_score=seed_val_score
    )
    logger.info(
        "seed: train_mean=%.3f val_mean=%s",
        state.train_mean(0),
        f"{seed_val_score:.3f}" if seed_val_score is not None else "-",
    )
    if log_dir:
        _log_iter(log_dir, 0, state)

    for it in range(1, cfg.num_iters + 1):
        parent_idx = state.best_idx()
        parent = state.candidates[parent_idx]

        mb_size = min(cfg.minibatch_size, len(train_dps))
        mb_idxs = rng.sample(range(len(train_dps)), mb_size)
        mb = [train_dps[i] for i in mb_idxs]

        # Parent rollout on minibatch
        parent_mb = await rollout_batch(
            parent,
            agent,
            mb,
            model,
            evaluators,
            weights,
            max_concurrency=cfg.max_concurrency,
            loop_name=cfg.loop_name,
            loop_version=cfg.loop_version,
        )
        parent_mb_score = sum(r.aggregated_score for r in parent_mb) / len(parent_mb)

        # Propose
        feedback_blocks = [feedback_for_rollout(r) for r in parent_mb]
        try:
            new_instruction = await propose_new_instruction(
                proposer_lm,
                current_instruction=parent[SYSTEM_PROMPT_KEY],
                feedback_blocks=feedback_blocks,
            )
        except Exception as e:
            logger.warning("iter %d: proposer failed (%s) — skipping", it, e)
            continue

        if not new_instruction.strip():
            logger.info("iter %d: proposer returned empty instruction — skipping", it)
            continue
        if new_instruction.strip() == parent[SYSTEM_PROMPT_KEY].strip():
            logger.info("iter %d: proposer returned unchanged instruction — skipping", it)
            continue

        new_candidate: Candidate = {SYSTEM_PROMPT_KEY: new_instruction}

        # Validate on minibatch
        new_mb = await rollout_batch(
            new_candidate,
            agent,
            mb,
            model,
            evaluators,
            weights,
            max_concurrency=cfg.max_concurrency,
            loop_name=cfg.loop_name,
            loop_version=cfg.loop_version,
        )
        new_mb_score = sum(r.aggregated_score for r in new_mb) / len(new_mb)
        logger.info("iter %d: minibatch parent=%.3f new=%.3f", it, parent_mb_score, new_mb_score)

        if new_mb_score < parent_mb_score:
            logger.info("iter %d: discarded (regression on minibatch)", it)
            continue

        # Full-eval new candidate
        new_train = await rollout_batch(
            new_candidate,
            agent,
            train_dps,
            model,
            evaluators,
            weights,
            max_concurrency=cfg.max_concurrency,
            loop_name=cfg.loop_name,
            loop_version=cfg.loop_version,
        )
        new_train_scores = [r.aggregated_score for r in new_train]
        new_val_score: float | None = None
        if val_dps:
            new_val = await rollout_batch(
                new_candidate,
                agent,
                val_dps,
                model,
                evaluators,
                weights,
                max_concurrency=cfg.max_concurrency,
                loop_name=cfg.loop_name,
                loop_version=cfg.loop_version,
            )
            new_val_score = sum(r.aggregated_score for r in new_val) / len(new_val)
        idx = state.add(
            new_candidate,
            parent_idx=parent_idx,
            train_scores=new_train_scores,
            val_score=new_val_score,
        )
        logger.info(
            "iter %d: accepted candidate %d (parent=%d) train=%.3f val=%s",
            it,
            idx,
            parent_idx,
            state.train_mean(idx),
            f"{new_val_score:.3f}" if new_val_score is not None else "-",
        )
        if log_dir:
            _log_iter(log_dir, it, state)

    best_idx = state.best_idx()
    return state.candidates[best_idx], state


def _log_iter(log_dir: Path, iteration: int, state: OptimizationState) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    idx = len(state.candidates) - 1
    record = {
        "iteration": iteration,
        "candidate_idx": idx,
        "parent_idx": state.parents[idx],
        "train_mean": state.train_mean(idx),
        "train_scores": state.score_matrix[idx],
        "val_mean": state.val_scores[idx],
        "system_prompt": state.candidates[idx].get(SYSTEM_PROMPT_KEY, ""),
    }
    with (log_dir / "iterations.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
