#!/usr/bin/env bash
# GEPA-DSPy baseline on the 5 enterprise agents (paper: gpt-5 system and reflection model,
# budget = 5 x train size, no merge).
#
# Usage:
#   bash scripts/run_gepa.sh                  # all 5
#   bash scripts/run_gepa.sh sql              # subset
#
# Logs: experiments/run_logs/gepa_<agent>.log. The evolved system prompt is archived to
# experiments/evolved_prompts/<agent>_system_prompt_gepa.md (the original baseline when GEPA
# keeps the seed); the agent's prompts and versions are then restored to their pre-run state.
set -euo pipefail
. "$(dirname "$0")/_common.sh"

MODEL="${MODEL:-gpt-5-2025-08-07}"
PROMPTS="$REPO/experiments/evolved_prompts"
GEPA_RUN_DIR="$REPO/experiments/run_gepa_benchmark"
mkdir -p "$PROMPTS" "$GEPA_RUN_DIR"
# Judges and the SQL tool use the models in their configs (gpt-4o-2024-11-20), as in the paper.

DEFAULT_AGENTS=(sql claim_history customer_support_triage_agent meeting_management product_questions_agent)
AGENTS=("${@:-${DEFAULT_AGENTS[@]}}")

for AGENT in "${AGENTS[@]}"; do
    RUN_DIR="$GEPA_RUN_DIR/${AGENT}_$(date +%Y%m%d_%H%M%S)"
    "$PY" -c "from agents_gym_core.prompt_optimization.gepa.benchmark import ensure_splits; ensure_splits('$AGENT')"
    TRAIN="$AGENTS_DIR/$AGENT/evaluations/eval-sets/evaluation-set-${AGENT}_train.json"
    TRAIN_SIZE=$("$PY" -c "import json; print(len(json.load(open('$TRAIN'))['evaluations']))")
    BUDGET=$((TRAIN_SIZE * 5))
    echo
    echo "# [$(date '+%H:%M:%S')] GEPA-DSPy: ${AGENT} (train=${TRAIN_SIZE}, budget=${BUDGET}, model=${MODEL})"
    snapshot_agent_prompts "$AGENT"
    "$PY" -m agents_gym_core.prompt_optimization.gepa.benchmark "$AGENT" \
        --variant dspy \
        --dspy-max-metric-calls "$BUDGET" \
        --dspy-no-merge \
        --dspy-num-threads 50 \
        --model "$MODEL" \
        --proposer-model "$MODEL" \
        --reflection-temperature 1.0 \
        --reflection-max-tokens 16384 \
        --max-concurrency 4 \
        --reset \
        --run-dir "$RUN_DIR" \
        > "$LOGS/gepa_$AGENT.log" 2>&1 || { restore_agent_prompts "$AGENT"; exit 1; }

    # Archive exactly what this run produced, as recorded in its rows.json.
    ROWS=$(find "$RUN_DIR" -name rows.json | head -1)
    ARCHIVE="$PROMPTS/${AGENT}_system_prompt_gepa.md"
    "$PY" - "$ROWS" "$AGENTS_DIR/$AGENT/prompts/baseline/system_prompt.md" "$ARCHIVE" <<'PYEOF' || { restore_agent_prompts "$AGENT"; exit 1; }
import json, shutil, sys
rows, baseline, archive = sys.argv[1:]
(row,) = json.load(open(rows))
if row.get("error"):
    sys.exit(f"GEPA failed: {row['error']}")
if row.get("seed_kept"):
    print("  GEPA kept the seed; archiving the original baseline")
    shutil.copy(baseline, archive)
elif row.get("evolved_prompt_path"):
    shutil.copy(row["evolved_prompt_path"], archive)
else:
    sys.exit("rows.json names neither a kept seed nor an evolved prompt")
PYEOF
    restore_agent_prompts "$AGENT"
    echo "  done (log: experiments/run_logs/gepa_$AGENT.log)"
done
