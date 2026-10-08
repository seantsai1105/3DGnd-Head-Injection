#!/usr/bin/env bash
# Evaluate a 3DGnd GR00T checkpoint on LIBERO and LIBERO-PRO with the paper protocol (Table 1):
# 6 suites x 10 tasks x 50 episodes, at most 512 env steps per episode, 8 of the 16 predicted
# actions executed per policy call, random object placements at every reset.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIBERO_SIM_DIR="$REPO_ROOT/gr00t/eval/sim/LIBERO"

LIBERO_PYTHON="${LIBERO_PYTHON:-$LIBERO_SIM_DIR/libero_uv/.venv/bin/python}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-5555}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SUITES="${SUITES:-libero_object libero_spatial libero_pro_object_task libero_pro_object_swap libero_pro_spatial_task libero_pro_spatial_swap}"
N_EPISODES="${N_EPISODES:-50}"
N_ENVS="${N_ENVS:-10}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-512}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
JOBS="${JOBS:-1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
TASK_TIMEOUT="${TASK_TIMEOUT:-0}"
VIDEO_DIR="${VIDEO_DIR:-}"

usage() {
    cat <<EOF
Usage: OUTPUT_DIR=<dir> bash examples/LIBERO/eval_libero_3dgnd.sh [--dry-run]

Start the policy server first (GR00T environment, e.g. in another terminal):
  uv run python gr00t/eval/run_gr00t_server.py --model-path <checkpoint> \\
      --embodiment-tag NEW_EMBODIMENT --use-sim-policy-wrapper --port $POLICY_PORT

Settings (environment variables, current value in brackets):
  LIBERO_PYTHON      LIBERO client python made by gr00t/eval/sim/LIBERO/setup_libero.sh
                     [$LIBERO_PYTHON]
  POLICY_HOST        policy server host [$POLICY_HOST]
  POLICY_PORT        policy server port [$POLICY_PORT]
  OUTPUT_DIR         results.jsonl and per-task logs; required [${OUTPUT_DIR:-unset}]
  SUITES             subset of the 6 evaluation suites [$SUITES]
  N_EPISODES         episodes per task [$N_EPISODES]
  N_ENVS             parallel simulators per task [$N_ENVS]
  MAX_EPISODE_STEPS  [$MAX_EPISODE_STEPS], N_ACTION_STEPS [$N_ACTION_STEPS]
  JOBS               tasks evaluated in parallel [$JOBS]
  MAX_ATTEMPTS       runs of a task before it counts as failed [$MAX_ATTEMPTS]
  TASK_TIMEOUT       time limit of one run of a task, e.g. 3h; 0 = none [$TASK_TIMEOUT]
  VIDEO_DIR          if set, rollout videos are saved to VIDEO_DIR/<task>/ [${VIDEO_DIR:-none}]

Every task appends one line to OUTPUT_DIR/results.jsonl, and tasks that already have one are
skipped, so running the script again resumes an interrupted evaluation. Use a new OUTPUT_DIR for
every checkpoint; results with other settings are never mixed into one results.jsonl.
A failed run of a task is repeated (up to MAX_ATTEMPTS runs); its log is kept as
OUTPUT_DIR/logs/<task>.attempt<k>.log. --dry-run only lists the tasks.
EOF
}

DRY_RUN=0
case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=1 ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; exit 1 ;;
esac

for name in N_EPISODES N_ENVS MAX_EPISODE_STEPS N_ACTION_STEPS JOBS MAX_ATTEMPTS; do
    if [[ ! ${!name} =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer, got '${!name}'" >&2
        exit 1
    fi
done
if [[ $DRY_RUN == 0 && -z "$OUTPUT_DIR" ]]; then
    echo "Set OUTPUT_DIR, a new directory for every checkpoint (see --help)." >&2
    exit 1
fi
if [[ ! -x "$LIBERO_PYTHON" ]]; then
    echo "LIBERO client python not found: $LIBERO_PYTHON" >&2
    echo "Run gr00t/eval/sim/LIBERO/setup_libero.sh or set LIBERO_PYTHON." >&2
    exit 1
fi
[[ $LIBERO_PYTHON == /* ]] || LIBERO_PYTHON="$PWD/$LIBERO_PYTHON"
if [[ -z "${LIBERO_CONFIG_PATH:-}" && -f "$LIBERO_SIM_DIR/libero_config/config.yaml" ]]; then
    export LIBERO_CONFIG_PATH="$LIBERO_SIM_DIR/libero_config"
elif [[ -n "${LIBERO_CONFIG_PATH:-}" && $LIBERO_CONFIG_PATH != /* ]]; then
    export LIBERO_CONFIG_PATH="$PWD/$LIBERO_CONFIG_PATH"
fi
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# shellcheck disable=SC2086  # SUITES is a list
ENV_NAMES=$("$LIBERO_PYTHON" - $SUITES <<'EOF' | grep '^libero_sim/'
import sys

import gymnasium as gym

from gr00t.eval.sim.LIBERO.libero_env import LIBERO_EVAL_SUITES, register_libero_envs

suites = list(dict.fromkeys(sys.argv[1:]))
if not suites or not set(suites) <= set(LIBERO_EVAL_SUITES):
    sys.exit(f"SUITES={' '.join(suites)!r}: choose from {' '.join(LIBERO_EVAL_SUITES)}")
register_libero_envs()
for suite in suites:
    for env_name, spec in gym.registry.items():
        if env_name.startswith("libero_sim/") and spec.kwargs["suite"] == suite:
            print(env_name)
EOF
)
N_TASKS=$(echo "$ENV_NAMES" | wc -l)
if [[ $DRY_RUN == 1 ]]; then
    echo "$ENV_NAMES"
    echo "$N_TASKS tasks" >&2
    exit 0
fi

RESULTS="$OUTPUT_DIR/results.jsonl"
# Finished tasks are skipped, so a results.jsonl must only hold results of the current settings.
if [[ -f "$RESULTS" ]]; then
    "$LIBERO_PYTHON" - "$RESULTS" "$N_EPISODES" "$N_ENVS" "$MAX_EPISODE_STEPS" \
        "$N_ACTION_STEPS" <<'EOF'
import json
import sys

path, *values = sys.argv[1:]
keys = ("n_episodes", "n_envs", "max_episode_steps", "n_action_steps")
settings = dict(zip(keys, map(float, values)))
with open(path) as f:
    for line in filter(str.strip, f):
        record = json.loads(line)
        other = [
            f"{key} {record.get(key)} (now {settings[key]:g})"
            for key in keys
            if record.get(key) != settings[key]
        ]
        if other:
            sys.exit(
                f"{path} has results with other settings ({record['env_name']}: "
                f"{', '.join(other)}); use another OUTPUT_DIR."
            )
EOF
fi

has_result() {
    [[ -f "$RESULTS" ]] && grep -qF "\"env_name\": \"$1\"" "$RESULTS"
}
n_done=0
for env_name in $ENV_NAMES; do
    if has_result "$env_name"; then
        n_done=$((n_done + 1))
    fi
done
if ((n_done < N_TASKS)) && [[ $MUJOCO_GL == egl && -v CUDA_VISIBLE_DEVICES ]] \
    && [[ -z $CUDA_VISIBLE_DEVICES && -z ${MUJOCO_EGL_DEVICE_ID:-} ]]; then
    echo "CUDA_VISIBLE_DEVICES is empty, but the simulators render on a GPU (EGL);" \
        "unset it or set MUJOCO_EGL_DEVICE_ID." >&2
    exit 1
fi

server_up() {
    # shellcheck disable=SC2016  # $0 and $1 are expanded by the inner bash
    timeout 10 bash -c 'exec 3<>"/dev/tcp/$0/$1"' "$POLICY_HOST" "$POLICY_PORT" 2>/dev/null
}
if ((n_done < N_TASKS)) && ! server_up; then
    echo "No policy server at $POLICY_HOST:$POLICY_PORT; start it first (see --help)." >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR/logs"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
RESULTS="$OUTPUT_DIR/results.jsonl"
if [[ -n "$VIDEO_DIR" ]]; then
    mkdir -p "$VIDEO_DIR"
    VIDEO_DIR="$(cd "$VIDEO_DIR" && pwd)"
fi
TIMEOUT_CMD=()
if [[ $TASK_TIMEOUT != 0 ]]; then
    TIMEOUT_CMD=(timeout --kill-after=1m "$TASK_TIMEOUT")
fi

# Rollouts occasionally die from an intermittent crash in a simulator worker (apparently native
# memory corruption in the LIBERO/robosuite/MuJoCo stack, also seen with the paper-era code): retry.
run_task() {
    local env_name=$1 task=${1#libero_sim/} attempt k status
    local log="$OUTPUT_DIR/logs/$task.log" video_args=()
    if [[ -n "$VIDEO_DIR" ]]; then
        video_args=(--video_dir "$VIDEO_DIR/$task")
    fi
    for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
        if ((attempt > 1)) && ! server_up; then
            echo "$(date '+%F %T') $task: FAILED, no policy server at $POLICY_HOST:$POLICY_PORT"
            return 1
        fi
        status=0
        # Run from OUTPUT_DIR so that simulator log files (MUJOCO_LOG.TXT) end up there.
        (cd "$OUTPUT_DIR" && ${TIMEOUT_CMD[@]+"${TIMEOUT_CMD[@]}"} "$LIBERO_PYTHON" \
            "$REPO_ROOT/gr00t/eval/rollout_policy.py" \
            --env_name "$env_name" \
            --n_episodes "$N_EPISODES" \
            --n_envs "$N_ENVS" \
            --max_episode_steps "$MAX_EPISODE_STEPS" \
            --n_action_steps "$N_ACTION_STEPS" \
            --policy_client_host "$POLICY_HOST" \
            --policy_client_port "$POLICY_PORT" \
            --output_dir "$OUTPUT_DIR" \
            ${video_args[@]+"${video_args[@]}"}) >"$log" 2>&1 || status=$?
        if has_result "$env_name"; then
            echo "$(date '+%F %T') $task: $(grep -m1 '^success rate:' "$log")"
            return 0
        fi
        if ((status == 130 || status == 143)); then
            return 1 # interrupted
        fi
        k=1
        while [[ -e "$OUTPUT_DIR/logs/$task.attempt$k.log" ]]; do
            k=$((k + 1))
        done
        mv "$log" "$OUTPUT_DIR/logs/$task.attempt$k.log"
        if [[ -n "$VIDEO_DIR" && -d "$VIDEO_DIR/$task" ]]; then
            mv "$VIDEO_DIR/$task" "$VIDEO_DIR/$task.attempt$k"
        fi
        echo "$(date '+%F %T') $task: run $attempt/$MAX_ATTEMPTS failed (exit status $status)," \
            "see $OUTPUT_DIR/logs/$task.attempt$k.log"
    done
    return 1
}

descendants() {
    local child
    for child in $(pgrep -P "$1"); do
        echo "$child"
        descendants "$child"
    done
}
# On Ctrl-C or SIGTERM, stop all runs (including their simulator workers) instead of retrying.
stop_all() {
    trap - INT TERM
    echo "Interrupted, stopping all tasks." >&2
    # shellcheck disable=SC2046  # a list of PIDs
    kill $(descendants $$) 2>/dev/null || true
    exit 130
}

echo "Evaluating $N_TASKS tasks -> $RESULTS"
echo "  N_EPISODES=$N_EPISODES N_ENVS=$N_ENVS MAX_EPISODE_STEPS=$MAX_EPISODE_STEPS" \
    "N_ACTION_STEPS=$N_ACTION_STEPS JOBS=$JOBS"
if ((n_done > 0)); then
    echo "Resuming: $n_done task(s) already have a result in $RESULTS and are skipped."
fi
trap stop_all INT TERM
for env_name in $ENV_NAMES; do
    if has_result "$env_name"; then
        continue
    fi
    while (($(jobs -rp | wc -l) >= JOBS)); do
        wait -n || true
    done
    if ! server_up; then
        echo "Policy server $POLICY_HOST:$POLICY_PORT is gone; no further tasks started." >&2
        break
    fi
    run_task "$env_name" &
done
wait
trap - INT TERM

if [[ -f "$RESULTS" ]]; then
    "$LIBERO_PYTHON" "$REPO_ROOT/examples/LIBERO/summarize_results.py" "$OUTPUT_DIR"
fi
failed=()
for env_name in $ENV_NAMES; do
    if ! has_result "$env_name"; then
        failed+=("${env_name#libero_sim/}")
    fi
done
if ((${#failed[@]} > 0)); then
    echo "${#failed[@]} task(s) without a result (logs in $OUTPUT_DIR/logs):" >&2
    printf '  %s\n' "${failed[@]}" >&2
    echo "Run the script again to evaluate them; finished tasks are skipped." >&2
    exit 1
fi
