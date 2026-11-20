#!/usr/bin/env bash
# ReAPO on the 5 enterprise agents (paper: gpt-5 optimizer and system model, skill full,
# 5 rounds, from the unoptimized baseline in prompts/baseline/ on the train split).
#
# Usage:
#   bash scripts/run_react.sh                      # all 5
#   bash scripts/run_react.sh sql claim_history    # subset
#   MODEL=gpt-4.1-mini-2025-04-14 ROUNDS=3 bash scripts/run_react.sh sql
#   SKILL=tool-free MODEL=gpt-4.1-mini-2025-04-14 ABLATION=score-only bash scripts/run_react.sh   # tools ablation
#   OVERWRITE=1 bash scripts/run_react.sh sql      # start an existing optimizer workspace over
#   CONTINUE=1 bash scripts/run_react.sh           # resume interrupted runs
#
# Logs: experiments/run_logs/react_<agent>[_ablation_<a>].log. Optimized prompts per round:
# the optimizer workspace tmp_react_optimizer_<agent>_train_.../round_*/ (see docs/REPRODUCING.md).
set -euo pipefail
. "$(dirname "$0")/_common.sh"

MODEL="${MODEL:-gpt-5-2025-08-07}"
OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-gpt-5-2025-08-07}"
ROUNDS="${ROUNDS:-5}"
SKILL="${SKILL:-full}"
ABLATION="${ABLATION:-full}"
# Judges and the SQL tool use the models in their configs (gpt-4o-2024-11-20), as in the paper.

DEFAULT_AGENTS=(sql claim_history customer_support_triage_agent meeting_management product_questions_agent)
AGENTS=("${@:-${DEFAULT_AGENTS[@]}}")

ABLATION_TAG=""
[ "$ABLATION" = full ] || ABLATION_TAG="_ablation_${ABLATION}"
OVERWRITE_FLAG=()
if [ "${OVERWRITE:-0}" = 1 ]; then OVERWRITE_FLAG=(-o); fi
# CONTINUE=1: resume each agent's interrupted run (finished ones are skipped).
if [ "${CONTINUE:-0}" = 1 ]; then OVERWRITE_FLAG+=(-c); fi

for AGENT in "${AGENTS[@]}"; do
    echo
    echo "# [$(date '+%H:%M:%S')] ReAPO ${ROUNDS} rounds: ${AGENT} (model=${MODEL}, optimizer=${OPTIMIZER_MODEL})"
    "$PY" -c "from agents_gym_core.prompt_optimization.gepa.benchmark import ensure_splits; ensure_splits('$AGENT')"
    snapshot_agent_prompts "$AGENT"
    "$PY" -m agents_gym_core.prompt_optimization.main "$AGENT:train" \
        --engine react \
        --rounds "$ROUNDS" \
        --skill "$SKILL" \
        --ablation "$ABLATION" \
        --model "$MODEL" \
        --optimizer-model "$OPTIMIZER_MODEL" \
        --reset \
        ${OVERWRITE_FLAG[@]+"${OVERWRITE_FLAG[@]}"} \
        > "$LOGS/react_${AGENT}${ABLATION_TAG}.log" 2>&1 || {
        restore_agent_prompts "$AGENT"
        echo "!! $AGENT: $(tail -n 1 "$LOGS/react_${AGENT}${ABLATION_TAG}.log")" >&2
        exit 1
    }
    restore_agent_prompts "$AGENT"
    echo "  done (log: experiments/run_logs/react_${AGENT}${ABLATION_TAG}.log)"
done
