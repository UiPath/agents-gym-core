# Shared setup for the ReAPO reproduction scripts (sourced, not run).
#
# - REPO / PY / AGENTS_DIR / LOGS; PY defaults to the repo's .venv (override with PY=...).
# - Loads <repo>/.env when present, so model credentials and AGENTS_GYM_LLM_BACKEND apply.
# - snapshot_agent_prompts / restore_agent_prompts (below).

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="${PY:-$REPO/.venv/bin/python}"
export AGENTS_DIR="$REPO/src/agents_gym_core/agents"  # used by the sourcing scripts
LOGS="$REPO/experiments/run_logs"
mkdir -p "$LOGS"
cd "$REPO" || exit 1

if [ -f "$REPO/.env" ]; then
    # Variables already set in the shell win over .env, as they do for gym itself.
    _shell_vars=()
    while IFS= read -r _name; do
        if [ -n "${!_name+x}" ]; then _shell_vars+=("$_name=${!_name}"); fi
    done < <(sed -nE 's/^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=.*/\2/p' "$REPO/.env")
    set -a
    # shellcheck disable=SC1091
    . "$REPO/.env"
    set +a
    for _var in ${_shell_vars[@]+"${_shell_vars[@]}"}; do export "${_var?}"; done
    unset _shell_vars _name _var
fi

# The optimizers edit an agent's prompt files in place (--reset first copies prompts/baseline/
# over them); the optimized prompts stay in the optimizer workspace (tmp_*/round_*).
# snapshot_agent_prompts copies the agent's prompts/ and versions.yaml as they are (local edits
# and untracked files included); restore_agent_prompts puts that copy back unless
# KEEP_OPTIMIZED_PROMPTS=1.
snapshot_agent_prompts() {
    local agent=$1
    SNAPSHOT_DIR=$(mktemp -d)
    cp -Rp "$AGENTS_DIR/$agent/prompts" "$SNAPSHOT_DIR/prompts"
    cp -p "$AGENTS_DIR/$agent/versions.yaml" "$SNAPSHOT_DIR/versions.yaml"
    # restore on Ctrl-C / kill / a closed terminal or dropped SSH session too
    trap 'restore_agent_prompts "'"$agent"'"; exit 130' INT TERM HUP
}

restore_agent_prompts() {
    local agent=$1
    trap - INT TERM HUP
    if [ -z "${SNAPSHOT_DIR:-}" ] || [ ! -d "$SNAPSHOT_DIR" ]; then
        return
    fi
    if [ "${KEEP_OPTIMIZED_PROMPTS:-0}" = "1" ]; then
        echo "  (KEEP_OPTIMIZED_PROMPTS=1: leaving $agent's edited prompts; snapshot at $SNAPSHOT_DIR)"
        return
    fi
    rm -rf "$AGENTS_DIR/$agent/prompts"
    cp -Rp "$SNAPSHOT_DIR/prompts" "$AGENTS_DIR/$agent/prompts"
    cp -p "$SNAPSHOT_DIR/versions.yaml" "$AGENTS_DIR/$agent/versions.yaml"
    rm -rf "$SNAPSHOT_DIR"
    SNAPSHOT_DIR=""
}
