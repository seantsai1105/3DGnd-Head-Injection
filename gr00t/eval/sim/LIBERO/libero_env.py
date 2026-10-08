"""
LIBERO environment

This file wraps LIBERO / LIBERO-PRO tasks as Gymnasium environments with the 3-D grounding
observation used by 3DGnd, and registers them as `libero_sim/<task name>` so that they can be
instantiated via gym.make(...) and work with our distributed evaluation.

Grounding: `grounding.target_delta` = p_target - p_eef (world frame, metres). The target is the
object to pick while LIBERO's `grasped()` (3DGnd LIBERO fork) reports that it is not lifted
(> 1 cm; re-evaluated at every step), otherwise the placement object / region (`:obj_of_interest`
in the BDDL file). p_target is the innermost pixel of the target's instance mask in the agentview
camera, lifted to 3-D with the rendered metric depth; if the target is not visible or not
segmentable (e.g. a region), its simulator position is projected into the image and the depth at
that pixel is lifted instead. This is the same computation as in the training data generation.
"""

import math
import os
import string

import cv2
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
from libero.libero import benchmark


os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from libero.libero.envs import SegmentationRenderEnv
from libero.libero.utils import get_libero_path
import numpy as np
import robosuite.utils.camera_utils as camera_utils


# Benchmark suites of the paper evaluation (10 tasks each).
LIBERO_EVAL_SUITES = (
    "libero_object",
    "libero_spatial",
    "libero_pro_object_task",
    "libero_pro_object_swap",
    "libero_pro_spatial_task",
    "libero_pro_spatial_swap",
)
IMAGE_SIZE = 256
GROUNDING_CAMERA = "agentview"
SETTLE_STEPS = 10  # no-op steps (gripper open) after each reset to let the objects settle


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    (without modifying `quat` in place).

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    w = np.clip(quat[3], -1.0, 1.0)

    den = np.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(w)) / den


def get_camera_matrices(sim, camera_name, image_size):
    """Return the intrinsics, the camera-to-world extrinsics (OpenCV convention) and the metric
    depth map of a camera, with the depth map flipped to the top-down row order of the images."""
    intrinsics = camera_utils.get_camera_intrinsic_matrix(sim, camera_name, image_size, image_size)
    cam_to_world = camera_utils.get_camera_extrinsic_matrix(sim, camera_name)
    depth = sim.render(camera_name=camera_name, height=image_size, width=image_size, depth=True)[1]
    depth = np.flipud(camera_utils.get_real_depth_map(sim, depth))
    return intrinsics, cam_to_world, depth


def lift_pixel(depth, u, v, intrinsics, cam_to_world):
    """Back-project pixel (u = column, v = row) with its metric depth to a 3-D world point."""
    z = float(depth[v, u])
    x = (u - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * z / intrinsics[1, 1]
    return (cam_to_world @ np.array([x, y, z, 1.0]))[:3]


def lift_projected_point(world_pos, depth, intrinsics, cam_to_world):
    """Project a 3-D world point to a pixel (clipped to the image) and lift that pixel's depth."""
    x, y, z = (np.linalg.inv(cam_to_world) @ np.append(world_pos, 1.0))[:3]
    u = int(round(x * intrinsics[0, 0] / z + intrinsics[0, 2]))
    v = int(round(y * intrinsics[1, 1] / z + intrinsics[1, 2]))
    height, width = depth.shape
    u = int(np.clip(u, 0, width - 1))
    v = int(np.clip(v, 0, height - 1))
    return lift_pixel(depth, u, v, intrinsics, cam_to_world)


def innermost_pixel(instance_to_id, name, segmentation):
    """Return the pixel (u, v) of the instance mask of `name` that is farthest from the mask
    boundary (top-down row order), or None if `name` is not visible in `segmentation`."""
    if name not in instance_to_id:
        return None
    mask = np.flipud(segmentation[:, :, 0] == instance_to_id[name])
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, _, _, (u, v) = cv2.minMaxLoc(dist)
    return (u, v) if mask[v, u] else None


def _vector_space(dim):
    return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


class LiberoEnv(gym.Env):
    """LIBERO / LIBERO-PRO task with the 3DGnd grounding observation.

    Args:
        task_bddl_file: BDDL problem file of the task.
        task_description: Language instruction of the task (LIBERO benchmark `task.language`).
        suite: Benchmark suite of the task. For LIBERO-PRO suites (`libero_pro_*`) the instruction
            is read from the BDDL file instead (the task names of the task-perturbation suites
            still describe the original task).

    Object placements are sampled by LIBERO from the global NumPy RNG at every reset (the paper
    evaluation does not use fixed initial states).
    """

    def __init__(self, task_bddl_file: str, task_description: str, suite: str):
        self._env = SegmentationRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=IMAGE_SIZE,
            camera_widths=IMAGE_SIZE,
            camera_depths=True,
            camera_segmentations="instance",
        )
        target_object = self._env.obj_of_interest[0]
        if not hasattr(self._env.env.object_states_dict[target_object], "grasped"):
            raise RuntimeError(
                "LIBERO is not the 3DGnd fork (ObjectState.grasped is missing); install "
                "external_dependencies/LIBERO with gr00t/eval/sim/LIBERO/setup_libero.sh"
            )
        if suite.startswith("libero_pro_"):
            task_description = self._env.language_instruction
        self._task_description = task_description

        self.observation_space = gym.spaces.Dict(
            {
                "video.image": gym.spaces.Box(
                    low=0, high=255, shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8
                ),
                "video.wrist_image": gym.spaces.Box(
                    low=0, high=255, shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8
                ),
                "state.x": _vector_space(1),
                "state.y": _vector_space(1),
                "state.z": _vector_space(1),
                "state.roll": _vector_space(1),
                "state.pitch": _vector_space(1),
                "state.yaw": _vector_space(1),
                "state.gripper": _vector_space(2),
                "grounding.target_delta": _vector_space(3),
                "annotation.human.action.task_description": gym.spaces.Text(
                    max_length=512, charset=string.printable
                ),
            }
        )
        self.action_space = spaces.Dict(
            {
                "action.x": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.y": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.z": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.roll": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.pitch": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.yaw": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.gripper": spaces.Box(low=-1, high=1, shape=(1,)),
            }
        )

    def close(self):
        self._env.close()

    def _get_target_position(self, obs):
        """3-D world position of the current sub-goal: the object to pick while it is not lifted,
        otherwise the placement target."""
        target_object, placement = self._env.obj_of_interest
        intrinsics, cam_to_world, depth = get_camera_matrices(
            self._env.sim, GROUNDING_CAMERA, IMAGE_SIZE
        )
        segmentation = obs[f"{GROUNDING_CAMERA}_segmentation_instance"]
        object_states = self._env.env.object_states_dict

        # `grasped()` is stateful: it must be called exactly once per processed observation.
        if not object_states[target_object].grasped():
            pixel = innermost_pixel(self._env.instance_to_id, target_object, segmentation)
            if pixel is None:
                sim_pos = obs[f"{target_object}_pos"]
                return lift_projected_point(sim_pos, depth, intrinsics, cam_to_world)
        else:
            pixel = innermost_pixel(self._env.instance_to_id, placement, segmentation)
            if pixel is None:
                sim_pos = object_states[placement].get_geom_state()["pos"]
                return lift_projected_point(sim_pos, depth, intrinsics, cam_to_world)
        return lift_pixel(depth, *pixel, intrinsics, cam_to_world)

    def _process_observation(self, obs):
        xyz = obs["robot0_eef_pos"]
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        # Computed in float64 (like the training data) and returned as float32 (observation space).
        vectors = {
            "state.x": xyz[0:1],
            "state.y": xyz[1:2],
            "state.z": xyz[2:3],
            "state.roll": rpy[0:1],
            "state.pitch": rpy[1:2],
            "state.yaw": rpy[2:3],
            "state.gripper": obs["robot0_gripper_qpos"],
            "grounding.target_delta": self._get_target_position(obs) - xyz,
        }
        return {
            "video.image": np.flipud(obs["agentview_image"]),
            "video.wrist_image": np.flipud(obs["robot0_eye_in_hand_image"]),
            **{key: value.astype(np.float32) for key, value in vectors.items()},
            "annotation.human.action.task_description": self._task_description,
        }

    def reset(self, seed=None, options=None):
        observation = self._env.reset()
        for _ in range(SETTLE_STEPS):
            self._env.step([0, 0, 0, 0, 0, 0, -1])
        # The paper evaluation processes the observation returned by reset(), i.e. from before
        # the objects settled (only the depth map is rendered after settling). This only affects
        # the first action chunk and is kept on purpose to reproduce the reported results.
        observation = self._process_observation(observation)
        info = {"success": self._env.check_success()}
        return observation, info

    def step(self, action):
        # Raw LIBERO actions (OSC_POSE deltas, gripper -1 = open / +1 = close), as in the
        # 3DGnd training data.
        action_vector = np.concatenate(
            [
                action["action.x"],
                action["action.y"],
                action["action.z"],
                action["action.roll"],
                action["action.pitch"],
                action["action.yaw"],
                action["action.gripper"],
            ],
            axis=0,
        )
        observation, reward, done, info = self._env.step(action_vector)
        observation = self._process_observation(observation)
        info["success"] = self._env.check_success()
        truncated = False
        return observation, reward, done, truncated, info


def register_libero_envs():
    """Register `libero_sim/<task name>` for all tasks of LIBERO_EVAL_SUITES (idempotent)."""
    benchmark_dict = benchmark.get_benchmark_dict()
    missing = [suite for suite in LIBERO_EVAL_SUITES if suite not in benchmark_dict]
    if missing:
        raise RuntimeError(
            f"LIBERO is not the 3DGnd fork (suites {missing} are not registered); install "
            "external_dependencies/LIBERO with gr00t/eval/sim/LIBERO/setup_libero.sh"
        )
    for task_suite_name in LIBERO_EVAL_SUITES:
        task_suite = benchmark_dict[task_suite_name]()
        for task_id in range(task_suite.get_num_tasks()):
            task = task_suite.get_task(task_id)
            env_id = f"libero_sim/{task.name}"
            if env_id in registry:
                continue
            task_bddl_file = os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
            )
            register(
                id=env_id,
                entry_point=f"{__name__}:LiberoEnv",
                kwargs={
                    "task_bddl_file": task_bddl_file,
                    "task_description": task.language,
                    "suite": task_suite_name,
                },
            )
