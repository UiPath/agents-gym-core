#!/usr/bin/env bash
# ReAPO on the 6 public benchmarks (defaults: gpt-5 optimizer, GPT-4.1-mini system
# model, skill full, 5 rounds, 6 on IFBench). Every setting is overridable; ROUNDS sets every benchmark.
#
# Run `uv run gym init <agent>` first so the train split exists.
#
# Usage:
#   bash scripts/run_react_benchmarks.sh                         # all 6
#   bash scripts/run_react_benchmarks.sh hover ifbench           # subset
#   OPTIMIZER_MODEL=gpt-4.1-mini-2025-04-14 bash scripts/run_react_benchmarks.sh   # GPT-4.1-mini optimizer table
#   OPTIMIZER_MODEL=anthropic.claude-opus-4-6-v1 bash scripts/run_react_benchmarks.sh
#   SKILL=tool-free ABLATION=score-only bash scripts/run_react_benchmarks.sh hover   # tools ablation
#   SKILL=trimmed bash scripts/run_react_benchmarks.sh hover        # skill prompt ablation
#   OVERWRITE=1 bash scripts/run_react_benchmarks.sh hover          # start an existing workspace over
#   CONTINUE=1 bash scripts/run_react_benchmarks.sh                # resume interrupted runs
#
# Logs: experiments/run_logs/react_<agent>_opt_<optimizer>_skill_<skill>[_ablation_<a>][_aggregate_only].log.
set -euo pipefail
. "$(dirname "$0")/_common.sh"

ROUNDS="${ROUNDS:-}"  # unset: 5 rounds, 6 on IFBench
LOOP_MODEL="${LOOP_MODEL:-gpt-4.1-mini-2025-04-14}"
OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-gpt-5-2025-08-07}"
SKILL="${SKILL:-full}"
ABLATION="${ABLATION:-full}"
# Judges use the model set in each evaluator's JSON (PUPA: gpt-4.1-mini-2025-04-14).

DEFAULT_AGENTS=(aime_2025_agent hotpotqa hover ifbench livebench_math papillon)
OVERWRITE_FLAG=()
if [ "${OVERWRITE:-0}" = 1 ]; then OVERWRITE_FLAG=(-o); fi
# CONTINUE=1: resume each agent's interrupted run (finished ones are skipped).
if [ "${CONTINUE:-0}" = 1 ]; then OVERWRITE_FLAG+=(-c); fi
AGENTS=("${@:-${DEFAULT_AGENTS[@]}}")

for AGENT in "${AGENTS[@]}"; do
    TRAIN_FILES=("$AGENTS_DIR/$AGENT"/evaluations/eval-sets/evaluation-set-*train.json)
    if [ ! -e "${TRAIN_FILES[0]}" ]; then
        echo "!! $AGENT has no train split: run \`uv run gym init $AGENT\` first; skipping"
        continue
    fi
    ABLATION_TAG=""
    [ "$ABLATION" = full ] || ABLATION_TAG="_ablation_${ABLATION}"
    NAME_FLAG=()  # the PUPA side experiment without quality and leakage has its own workspace
    if [ "${COACH_HIDE_EVALUATOR_FIELDS:-}" = 1 ]; then
        NAME_FLAG=(--name react_optimizer_aggregate_only)
        ABLATION_TAG+="_aggregate_only"
    fi
    LOG="$LOGS/react_${AGENT}_opt_${OPTIMIZER_MODEL//\//_}_skill_${SKILL}${ABLATION_TAG}.log"  # Qwen/... has a /
    AGENT_ROUNDS="${ROUNDS:-$([ "$AGENT" = ifbench ] && echo 6 || echo 5)}"
    echo
    echo "# [$(date '+%H:%M:%S')] ReAPO ${AGENT_ROUNDS} rounds: ${AGENT} (loop=${LOOP_MODEL}, optimizer=${OPTIMIZER_MODEL})"
    snapshot_agent_prompts "$AGENT"
    "$PY" -m agents_gym_core.prompt_optimization.main "$AGENT:train" \
        --engine react \
        --rounds "$AGENT_ROUNDS" \
        --skill "$SKILL" \
        --ablation "$ABLATION" \
        --model "$LOOP_MODEL" \
        --optimizer-model "$OPTIMIZER_MODEL" \
        --reset \
        ${OVERWRITE_FLAG[@]+"${OVERWRITE_FLAG[@]}"} \
        ${NAME_FLAG[@]+"${NAME_FLAG[@]}"} \
        > "$LOG" 2>&1 || {
        restore_agent_prompts "$AGENT"
        echo "!! $AGENT: $(tail -n 1 "$LOG")" >&2
        exit 1
    }
    restore_agent_prompts "$AGENT"
    echo "  done (log: ${LOG#"$REPO"/})"
done
