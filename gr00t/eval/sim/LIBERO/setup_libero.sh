#!/usr/bin/env bash
# Set up the LIBERO / LIBERO-PRO evaluation client (safe to re-run):
#   1. the submodule external_dependencies/LIBERO (LIBERO with the 3DGnd changes) at the pinned
#      commit; in a source archive without git metadata, a clone of its URL from .gitmodules,
#   2. a repo-local LIBERO config in libero_config/ (~/.libero is not touched),
#   3. the client venv libero_uv/.venv with the pinned package versions of requirements-eval.txt,
#   4. a smoke test.
# `--libero-only` stops after step 2 (e.g. to generate the training data in another environment).
# Requires git; steps 3-4 also need uv, EGL (e.g. `apt install libegl1-mesa-dev libglu1-mesa`) and
# cmake (any version), make and a C++ compiler (e.g. `apt install cmake build-essential`).
set -euo pipefail

LIBERO_ONLY=0
case "$*" in
    "") ;;
    --libero-only) LIBERO_ONLY=1 ;;
    -h | --help) echo "Usage: $0 [--libero-only]"; exit 0 ;;
    *) echo "Usage: $0 [--libero-only]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LIBERO_SUBMODULE="external_dependencies/LIBERO"
LIBERO_DIR="$REPO_ROOT/$LIBERO_SUBMODULE"
LIBERO_COMMIT="bb00f8e87945db05f29338e26b41c1a3a4aa7b91"
VENV="$SCRIPT_DIR/libero_uv/.venv"
PYTHON="$VENV/bin/python"
CONFIG_DIR="$SCRIPT_DIR/libero_config"

if [ "$LIBERO_ONLY" = 0 ] && ! command -v uv >/dev/null; then
    echo "uv is required: https://docs.astral.sh/uv/" >&2
    exit 1
fi

echo "==> LIBERO @ $LIBERO_COMMIT"
if git -C "$REPO_ROOT" ls-files --stage -- "$LIBERO_SUBMODULE" 2>/dev/null | grep -q '^160000'
then
    git -C "$REPO_ROOT" submodule update --init "$LIBERO_SUBMODULE"
else
    if [ ! -e "$LIBERO_DIR/.git" ]; then
        LIBERO_URL="$(git config -f "$REPO_ROOT/.gitmodules" "submodule.$LIBERO_SUBMODULE.url")" ||
            { echo "No URL for $LIBERO_SUBMODULE in $REPO_ROOT/.gitmodules" >&2; exit 1; }
        git clone --no-checkout "$LIBERO_URL" "$LIBERO_DIR"
    fi
    git -C "$LIBERO_DIR" checkout -q "$LIBERO_COMMIT"
fi
LIBERO_HEAD="$(git -C "$LIBERO_DIR" rev-parse HEAD)"
if [ "$LIBERO_HEAD" != "$LIBERO_COMMIT" ]; then
    echo "$LIBERO_SUBMODULE is at $LIBERO_HEAD, not at the pinned commit $LIBERO_COMMIT" >&2
    exit 1
fi

echo "==> LIBERO config $CONFIG_DIR/config.yaml"
mkdir -p "$CONFIG_DIR" "$LIBERO_DIR/libero/datasets"
cat > "$CONFIG_DIR/config.yaml" <<EOF
assets: $LIBERO_DIR/libero/libero/assets
bddl_files: $LIBERO_DIR/libero/libero/bddl_files
benchmark_root: $LIBERO_DIR/libero/libero
datasets: $LIBERO_DIR/libero/datasets
init_states: $LIBERO_DIR/libero/libero/init_files
EOF
if [ "$LIBERO_ONLY" = 1 ]; then
    echo "Done. Before using LIBERO: export LIBERO_CONFIG_PATH=$CONFIG_DIR"
    exit 0
fi

echo "==> Client venv $VENV"
[ -x "$PYTHON" ] || uv venv "$VENV" --python 3.10
# egl-probe (a robomimic dependency) is built from source and predates CMake 4, which needs this
# policy floor to configure it; older CMake versions ignore the variable.
CMAKE_POLICY_VERSION_MINIMUM="${CMAKE_POLICY_VERSION_MINIMUM:-3.5}" \
    uv pip install --python "$PYTHON" --no-deps -r "$SCRIPT_DIR/requirements-eval.txt"
uv pip install --python "$PYTHON" --no-deps -e "$LIBERO_DIR" --config-settings editable_mode=compat
uv pip install --python "$PYTHON" --no-deps -e "$REPO_ROOT"

echo "==> Smoke test"
LIBERO_CONFIG_PATH="$CONFIG_DIR" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$PYTHON" - <<'PY'
import gymnasium as gym

from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

register_libero_envs()
env_id = (
    "libero_sim/libero_pro_spatial_task_pick_up_the_black_bowl_between_the_plate_and_the_ramekin"
    "_and_place_it_on_the_plate"
)
env = gym.make(env_id)
obs, _ = env.reset()
print("Env OK:", env_id, "| grounding.target_delta =", obs["grounding.target_delta"])
env.close()
PY

echo "Done. Before running the evaluation client:"
echo "  source $VENV/bin/activate"
echo "  export LIBERO_CONFIG_PATH=$CONFIG_DIR"
