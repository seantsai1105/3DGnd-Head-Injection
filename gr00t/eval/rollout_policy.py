import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
import json
import os
from pathlib import Path
import time
from typing import Any

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.policy import BasePolicy
import gymnasium as gym
import numpy as np
from tqdm import tqdm


@dataclass
class VideoConfig:
    """Configuration for video recording settings.

    Attributes:
        video_dir: Directory to save videos (if None, no videos are saved)
        steps_per_render: Number of steps between each call to env.render() while recording
            during rollout
        fps: Frames per second for the output video
        codec: Video codec to use for compression
        input_pix_fmt: Input pixel format
        crf: Constant Rate Factor for video compression (lower = better quality)
        thread_type: Threading strategy for video encoding
        thread_count: Number of threads to use for encoding
    """

    video_dir: str | None = None
    steps_per_render: int = 2
    max_episode_steps: int = 720
    fps: int = 20
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1
    overlay_text: bool = True
    n_action_steps: int = 8


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings.

    Attributes:
        video_delta_indices: Indices of video observations to stack
        state_delta_indices: Indices of state observations to stack
        n_action_steps: Number of action steps to execute
        max_episode_steps: Maximum number of steps per episode
    """

    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 720
    terminate_on_success: bool = False


@dataclass
class WrapperConfigs:
    """Container for various environment wrapper configurations.

    Attributes:
        video: Configuration for video recording
        multistep: Configuration for multi-step processing
    """

    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


def get_gym_env(env_name: str) -> gym.Env:
    """Create a LIBERO / LIBERO-PRO environment (`libero_sim/<task name>`) without wrappers."""
    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

    register_libero_envs()
    return gym.make(env_name)


def create_eval_env(env_name: str, env_idx: int, wrapper_configs: WrapperConfigs) -> gym.Env:
    """Create a single evaluation environment with wrappers.

    Args:
        env_name: Name of the gymnasium environment to use
        env_idx: Environment index (prefix of the recorded video files)
        wrapper_configs: Configuration for environment wrappers
    Returns:
        Wrapped gymnasium environment
    """

    env = get_gym_env(env_name)
    if wrapper_configs.video.video_dir is not None:
        from gr00t.eval.sim.wrapper.video_recording_wrapper import (
            VideoRecorder,
            VideoRecordingWrapper,
        )

        video_recorder = VideoRecorder.create_h264(
            fps=wrapper_configs.video.fps,
            codec=wrapper_configs.video.codec,
            input_pix_fmt=wrapper_configs.video.input_pix_fmt,
            crf=wrapper_configs.video.crf,
            thread_type=wrapper_configs.video.thread_type,
            thread_count=wrapper_configs.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(wrapper_configs.video.video_dir),
            steps_per_render=wrapper_configs.video.steps_per_render,
            max_episode_steps=wrapper_configs.video.max_episode_steps,
            overlay_text=wrapper_configs.video.overlay_text,
            file_prefix=f"env{env_idx}_",
        )

    env = MultiStepWrapper(
        env,
        video_delta_indices=wrapper_configs.multistep.video_delta_indices,
        state_delta_indices=wrapper_configs.multistep.state_delta_indices,
        n_action_steps=wrapper_configs.multistep.n_action_steps,
        max_episode_steps=wrapper_configs.multistep.max_episode_steps,
        terminate_on_success=wrapper_configs.multistep.terminate_on_success,
    )
    return env


def run_rollout_gymnasium_policy(
    env_name: str,
    policy: BasePolicy,
    wrapper_configs: WrapperConfigs,
    n_episodes: int = 10,
    n_envs: int = 1,
) -> Any:
    """Run policy rollouts in parallel environments.

    Args:
        env_name: Name of the gymnasium environment to use
        policy: Policy that returns action chunks for batched observations
        wrapper_configs: Configuration for environment wrappers
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
    Returns:
        Collection results from running the episodes
    """
    start_time = time.time()
    n_episodes = max(n_episodes, n_envs)
    print(f"Running collecting {n_episodes} episodes for {env_name} with {n_envs} vec envs")

    env_fns = [
        partial(
            create_eval_env,
            env_idx=idx,
            env_name=env_name,
            wrapper_configs=wrapper_configs,
        )
        for idx in range(n_envs)
    ]

    if n_envs == 1:
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=False,
            context="spawn",
        )

    # Storage for results
    episode_lengths = []
    current_rewards = [0] * n_envs
    current_lengths = [0] * n_envs
    completed_episodes = 0
    current_successes = [False] * n_envs
    episode_successes = []
    episode_infos = defaultdict(list)

    # Initial reset
    observations, _ = env.reset()
    policy.reset()
    i = 0

    pbar = tqdm(total=n_episodes, desc="Episodes")
    while completed_episodes < n_episodes:
        actions, _ = policy.get_action(observations)
        next_obs, rewards, terminations, truncations, env_infos = env.step(actions)
        # NOTE (FY): Currently we don't properly handle policy reset. For now, our policy are stateless,
        # but in the future if we need policy to be stateful, we need to detect env reset and call policy.reset()
        i += 1
        # Update episode tracking
        for env_idx in range(n_envs):
            if "success" in env_infos:
                env_success = env_infos["success"][env_idx]
                if isinstance(env_success, list):
                    env_success = np.any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            else:
                current_successes[env_idx] = False

            if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                env_success = env_infos["final_info"][env_idx]["success"]
                if isinstance(env_success, list):
                    env_success = any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            current_rewards[env_idx] += rewards[env_idx]
            current_lengths[env_idx] += 1

            # If episode ended, store results
            if terminations[env_idx] or truncations[env_idx]:
                if "final_info" in env_infos:
                    current_successes[env_idx] |= any(env_infos["final_info"][env_idx]["success"])
                if "task_progress" in env_infos:
                    episode_infos["task_progress"].append(env_infos["task_progress"][env_idx][-1])
                if "q_score" in env_infos:
                    episode_infos["q_score"].append(np.max(env_infos["q_score"][env_idx]))
                if "valid" in env_infos:
                    episode_infos["valid"].append(all(env_infos["valid"][env_idx]))
                # Accumulate results
                episode_lengths.append(current_lengths[env_idx])
                episode_successes.append(current_successes[env_idx])
                # Reset trackers for this environment.
                current_successes[env_idx] = False
                # only update completed_episodes if valid
                if "valid" in episode_infos:
                    if episode_infos["valid"][-1]:
                        completed_episodes += 1
                        pbar.update(1)
                else:
                    # envs don't return valid
                    completed_episodes += 1
                    pbar.update(1)
                current_rewards[env_idx] = 0
                current_lengths[env_idx] = 0
        observations = next_obs
    pbar.close()

    if wrapper_configs.video.video_dir is not None:
        # Finalizes the video files (deletes those of the unfinished episodes). Skipped otherwise:
        # it hard-resets every simulator, and a worker crash there would discard all results.
        env.reset()
    env.close()
    print(f"Collecting {n_episodes} episodes took {time.time() - start_time} seconds")

    assert len(episode_successes) >= n_episodes, (
        f"Expected at least {n_episodes} episodes, got {len(episode_successes)}"
    )

    episode_infos = dict(episode_infos)  # Convert defaultdict to dict
    for key, value in episode_infos.items():
        assert len(value) == len(episode_successes), (
            f"Length of {key} is not equal to the number of episodes"
        )

    # process valid results
    if "valid" in episode_infos:
        valids = episode_infos["valid"]
        valid_idxs = np.where(valids)[0]
        episode_successes = [episode_successes[i] for i in valid_idxs]
        episode_infos = {k: [v[i] for i in valid_idxs] for k, v in episode_infos.items()}

    return env_name, episode_successes, episode_infos


def create_gr00t_sim_policy(
    model_path: str,
    embodiment_tag: EmbodimentTag,
    policy_client_host: str = "127.0.0.1",
    policy_client_port: int = 5555,
) -> BasePolicy:
    """Load the policy in-process if `model_path` is given, else connect to a policy server."""
    if model_path:
        from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

        return Gr00tSimPolicyWrapper(
            Gr00tPolicy(
                embodiment_tag=embodiment_tag,
                model_path=model_path,
                device=0,
            )
        )
    from gr00t.policy.server_client import PolicyClient

    return PolicyClient(host=policy_client_host, port=policy_client_port)


def run_gr00t_sim_policy(
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    model_path: str = "",
    policy_client_host: str = "127.0.0.1",
    policy_client_port: int = 5555,
    n_envs: int = 10,
    n_action_steps: int = 8,
    video_dir: str | None = None,
):
    embodiment_tag = get_embodiment_tag_from_env_name(env_name)

    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=video_dir,
            max_episode_steps=max_episode_steps,
        ),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )

    policy = create_gr00t_sim_policy(
        model_path, embodiment_tag, policy_client_host, policy_client_port
    )

    results = run_rollout_gymnasium_policy(
        env_name=env_name,
        policy=policy,
        wrapper_configs=wrapper_configs,
        n_episodes=n_episodes,
        n_envs=n_envs,
    )
    if video_dir is not None:
        print("Video saved to: ", video_dir)
    return results


def get_task_suite(env_name: str) -> str:
    """Benchmark suite of a registered `libero_sim/<task name>` environment."""
    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

    register_libero_envs()
    return gym.spec(env_name).kwargs["suite"]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a GR00T policy on one LIBERO / LIBERO-PRO task. Run it with the "
        "LIBERO client python and LIBERO_CONFIG_PATH from gr00t/eval/sim/LIBERO/setup_libero.sh "
        "(examples/LIBERO/eval_libero_3dgnd.sh sets both).",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        required=True,
        help="e.g. libero_sim/pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=50, help="Episodes per task (default: %(default)s)."
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=10,
        help="Parallel simulators (default: %(default)s).",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=512,
        help="Maximum env steps per episode (default: %(default)s).",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=8,
        help="Actions executed per policy call (default: %(default)s).",
    )
    parser.add_argument(
        "--policy_client_host",
        type=str,
        default="127.0.0.1",
        help="Policy server host (default: %(default)s).",
    )
    parser.add_argument(
        "--policy_client_port",
        type=int,
        default=5555,
        help="Policy server port (default: %(default)s).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Run the policy in this process instead of connecting to a policy server (needs one "
        "environment with both the GR00T model dependencies and the LIBERO client stack; the "
        "client venv of setup_libero.sh has only the latter).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory of results.jsonl (one line per evaluated task is appended).",
    )
    parser.add_argument(
        "--video_dir", type=str, default=None, help="Save rollout videos here (default: none)."
    )
    args = parser.parse_args()

    suite = get_task_suite(args.env_name)
    results = run_gr00t_sim_policy(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        model_path=args.model_path,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        video_dir=args.video_dir,
    )
    all_successes = [bool(success) for success in results[1]]
    successes = all_successes[: args.n_episodes]
    success_rate = float(np.mean(successes))
    if args.model_path:
        policy_info = {"model_path": args.model_path}
    else:
        policy_info = {"host": args.policy_client_host, "port": args.policy_client_port}
    record = {
        "env_name": args.env_name,
        "suite": suite,
        "success_rate": success_rate,
        "n_episodes": args.n_episodes,
        "successes": successes,
        "n_completed_episodes": len(all_successes),
        "all_successes": all_successes,
        "n_envs": args.n_envs,
        "max_episode_steps": args.max_episode_steps,
        "n_action_steps": args.n_action_steps,
        "policy": policy_info,
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "results.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"success rate: {success_rate}")


if __name__ == "__main__":
    main()
