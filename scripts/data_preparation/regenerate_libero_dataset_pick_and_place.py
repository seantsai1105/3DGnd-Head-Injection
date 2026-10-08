"""Regenerate the 3DGnd LIBERO pick-and-place data by replaying the raw LIBERO demonstrations.

Every task of the task list is replayed in its own environment, seeded once with 0, with its demos
in order (LIBERO re-samples the fixture poses from the global RNG on every reset, so the result of
a demo depends on the demos replayed before it). No-op actions are dropped. For every kept frame
the robot state, the action, both camera images and the grounding signal

    target_delta = p_target - p_eef

are recorded. p_target is the placement target in every frame in which the LIBERO fork's
``grasped()`` reports the object to pick more than 1 cm above its reference height (its height in
the 12th frame), and the object to pick in all other frames. It is the innermost visible pixel of
the target's segmentation mask, lifted to 3-D with the rendered agentview depth; targets without a
visible mask (placement regions, occluded objects) are lifted at the projection of their position
instead. ``gr00t/eval/sim/LIBERO/libero_env.py`` computes the target in the same way for the
evaluation; keep the two in sync. A demo is kept only if the replay succeeds and reaches the
placement stage.

The replay (initial state, 10 settle steps, no-op filter, success filter) follows OpenVLA's
``experiments/robot/libero/regenerate_libero_dataset.py`` (https://github.com/openvla/openvla, MIT
License, Copyright (c) 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti).

Output: ``<output_dir>/<task>_demo.hdf5`` per task, demos keyed by their raw demo index::

    data                                attrs: task, suite, bddl_file, language, num_raw_demos, ...
    data/demo_<i>/actions               (T, 7) float64  raw LIBERO action
    data/demo_<i>/robot_states          (T, 8) float64  eef xyz, eef axis-angle, gripper qpos
    data/demo_<i>/obs/target_delta      (T, 3) float64  p_target - p_eef
    data/demo_<i>/obs/agentview_rgb     (T, 256, 256, 3) uint8, upright
    data/demo_<i>/obs/eye_in_hand_rgb   (T, 256, 256, 3) uint8, upright

Requirements (to reproduce the original training data, see README.md):
    - the LIBERO fork in ``external_dependencies/LIBERO`` (``grasped()`` in
      ``libero/libero/envs/object_states/base_object_states.py`` and the ``obj_of_interest`` edits
      of the libero_90 bddl files), importable and selected with ``LIBERO_CONFIG_PATH``;
    - mujoco 3.4.0, robosuite 1.4.0, numpy 1.22.4 (``requirements-data.txt``);
    - NVIDIA EGL rendering: ``MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
      __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json``.

Usage:
    python scripts/data_preparation/regenerate_libero_dataset_pick_and_place.py \\
        --libero_raw_data_dir <raw LIBERO datasets dir> --output_dir <output dir>
"""

import argparse
import math
import os
from pathlib import Path

import cv2
import h5py
from libero.libero import get_libero_path
from libero.libero.envs import SegmentationRenderEnv
import numpy as np
import robosuite.utils.camera_utils as camera_utils


DEFAULT_TASK_LIST = Path(__file__).with_name("libero_pick_and_place_tasks.txt")
IMAGE_RESOLUTION = 256
CAMERA_NAME = "agentview"
NUM_SETTLE_STEPS = 10


def read_task_list(path):
    """Return the (suite, task) pairs of a task list with lines ``.../<suite>/<task>.bddl``."""
    with open(path) as f:
        bddl_paths = [Path(line.strip()) for line in f if line.strip()]
    return [(p.parent.name, p.stem) for p in bddl_paths]


def get_camera_matrices(sim, camera_name, resolution):
    """Return the intrinsics K, the camera-to-world pose T_c2w (OpenCV convention) and the metric
    z-depth map (flipped upright like the stored images) of ``camera_name``."""
    K = camera_utils.get_camera_intrinsic_matrix(
        sim=sim, camera_name=camera_name, camera_height=resolution, camera_width=resolution
    )
    T_c2w = camera_utils.get_camera_extrinsic_matrix(sim, camera_name)
    raw_depth = sim.render(
        camera_name=camera_name, height=resolution, width=resolution, depth=True
    )[1]
    depth_map = np.flipud(camera_utils.get_real_depth_map(sim, raw_depth))
    return K, T_c2w, depth_map


def lift_pixel(depth_map, u, v, K, T_c2w):
    """Back-project pixel (u = column, v = row) with its metric depth to a 3-D world point."""
    z = float(depth_map[v, u])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_c = (u - cx) * z / fx
    y_c = (v - cy) * z / fy
    p_world = T_c2w @ np.array([x_c, y_c, z, 1.0])
    return p_world[:3]


def lift_projected_point(world_pos, depth_map, K, T_c2w):
    """Project a 3-D world point to the nearest pixel (clipped to the image) and lift that pixel
    with its depth."""
    p_cam = np.linalg.inv(T_c2w) @ np.append(world_pos, 1.0)
    x, y, z = p_cam[:3]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = int(round(x * fx / z + cx))
    v = int(round(y * fy / z + cy))
    height, width = depth_map.shape[:2]
    u = int(np.clip(u, 0, width - 1))
    v = int(np.clip(v, 0, height - 1))
    return lift_pixel(depth_map, u, v, K, T_c2w)


def innermost_pixel(instance_to_id, name, segmentation):
    """Return the pixel (u, v) of the upright instance mask of ``name`` that is farthest from the
    mask boundary (maximum of the L2 distance transform), or None if ``name`` is not visible."""
    if name not in instance_to_id:
        return None
    mask = np.zeros(segmentation.shape[:2], dtype=np.uint8)
    mask[segmentation[:, :, 0] == instance_to_id[name]] = 1
    mask = np.flipud(mask).copy()  # cv2 needs a contiguous array
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, _, _, (u, v) = cv2.minMaxLoc(dist_transform)
    return (u, v) if mask[v, u] else None


def compute_target_point(env, obs, depth_map, K, T_c2w):
    """Return the 3-D target point of the current frame and whether it is the placement target."""
    target_object, placement_area = env.obj_of_interest
    segmentation = obs[f"{CAMERA_NAME}_segmentation_instance"]
    object_states = env.env.object_states_dict

    # grasped() is stateful (it counts its calls): call it exactly once per recorded frame.
    if not object_states[target_object].grasped():
        pixel = innermost_pixel(env.instance_to_id, target_object, segmentation)
        if pixel is None:
            world_pos = obs[f"{target_object}_pos"]
            return lift_projected_point(world_pos, depth_map, K, T_c2w), False
        return lift_pixel(depth_map, *pixel, K, T_c2w), False

    pixel = innermost_pixel(env.instance_to_id, placement_area, segmentation)
    if pixel is None:
        world_pos = object_states[placement_area].get_geom_state()["pos"]
        return lift_projected_point(world_pos, depth_map, K, T_c2w), True
    return lift_pixel(depth_map, *pixel, K, T_c2w), True


def quat2axisangle(quat):
    """robosuite's ``transform_utils.quat2axisangle`` (MIT License), without modifying ``quat``."""
    quat = quat.copy()
    quat[3] = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def is_noop(action, prev_action=None, threshold=1e-4):
    """OpenVLA's no-op test: no arm motion and, if there is a previous action, the same gripper
    command."""
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    return np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1]


def replay_demo(env, actions, init_state):
    """Replay one raw demo. Return its recorded frames, or None if the demo is not kept."""
    env.reset()
    env.set_init_state(init_state)
    for _ in range(NUM_SETTLE_STEPS):  # let the objects settle, gripper open
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    frames = {
        key: []
        for key in ("actions", "robot_states", "target_delta", "agentview_rgb", "eye_in_hand_rgb")
    }
    reached_placement = done = False
    for action in actions:
        if is_noop(action, frames["actions"][-1] if frames["actions"] else None):
            continue

        frames["actions"].append(action)
        frames["robot_states"].append(
            np.hstack(
                (
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            )
        )
        frames["agentview_rgb"].append(np.flipud(obs["agentview_image"]))
        frames["eye_in_hand_rgb"].append(np.flipud(obs["robot0_eye_in_hand_image"]))

        K, T_c2w, depth_map = get_camera_matrices(env.sim, CAMERA_NAME, IMAGE_RESOLUTION)
        target_point, placing = compute_target_point(env, obs, depth_map, K, T_c2w)
        reached_placement |= placing
        frames["target_delta"].append(target_point - obs["robot0_eef_pos"])

        obs, _, done, _ = env.step(action.tolist())

    if not (done and reached_placement):
        return None
    return {key: np.stack(values) for key, values in frames.items()}


def write_demo(group, demo):
    group.create_dataset("actions", data=demo["actions"])
    group.create_dataset("robot_states", data=demo["robot_states"])
    group.create_dataset("obs/target_delta", data=demo["target_delta"])
    for key in ("agentview_rgb", "eye_in_hand_rgb"):
        images = demo[key]
        group.create_dataset(
            f"obs/{key}", data=images, chunks=(1, *images.shape[1:]), compression="gzip"
        )


def task_paths(args, suite, task):
    """Return the raw demo file and the output file of a task."""
    raw_path = os.path.join(args.libero_raw_data_dir, suite, f"{task}_demo.hdf5")
    out_path = os.path.join(args.output_dir, f"{task}_demo.hdf5")
    return raw_path, out_path


def written_with_same_settings(out_path, args):
    """Whether an existing output file was written with the replay settings of ``args``."""
    with h5py.File(out_path, "r") as f:
        attrs = dict(f["data"].attrs)
    num_raw_demos = attrs.get("num_raw_demos")
    if num_raw_demos is None:
        return False
    num_demos = num_raw_demos if args.max_demos is None else min(num_raw_demos, args.max_demos)
    return attrs.get("num_replayed_demos") == num_demos


def regenerate_task(suite, task, args):
    """Replay the demos of one task into its output file; return (kept, replayed) demo counts."""
    bddl_file = os.path.join(get_libero_path("bddl_files"), suite, f"{task}.bddl")
    raw_path, out_path = task_paths(args, suite, task)
    tmp_path = f"{out_path}.tmp"

    env = SegmentationRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=IMAGE_RESOLUTION,
        camera_widths=IMAGE_RESOLUTION,
        camera_depths=True,
        camera_segmentations="instance",
    )
    env.seed(0)
    try:
        with h5py.File(raw_path, "r") as raw_file, h5py.File(tmp_path, "w") as out_file:
            raw_data = raw_file["data"]
            num_demos = len(raw_data)
            if args.max_demos is not None:
                num_demos = min(num_demos, args.max_demos)

            data = out_file.create_group("data")
            target_object, placement_area = env.obj_of_interest
            data.attrs.update(
                task=task,
                suite=suite,
                bddl_file=f"{suite}/{task}.bddl",
                language=env.language_instruction,
                target_object=target_object,
                placement_area=placement_area,
                num_raw_demos=len(raw_data),
                num_replayed_demos=num_demos,
            )
            num_kept = 0
            for i in range(num_demos):
                raw_demo = raw_data[f"demo_{i}"]
                demo = replay_demo(env, raw_demo["actions"][()], raw_demo["states"][0])
                if demo is None:
                    print(
                        f"  demo_{i}: dropped (replay failed or placement not reached)", flush=True
                    )
                    continue
                write_demo(data.create_group(f"demo_{i}"), demo)
                num_kept += 1
                print(f"  demo_{i}: kept, {len(demo['actions'])} frames", flush=True)
    finally:
        env.close()
    os.replace(tmp_path, out_path)
    return num_kept, num_demos


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--libero_raw_data_dir",
        "--libero-raw-data-dir",
        required=True,
        help="Raw LIBERO demos, laid out as <dir>/<suite>/<task>_demo.hdf5.",
    )
    parser.add_argument(
        "--output_dir", "--output-dir", required=True, help="Output directory for the HDF5 files."
    )
    parser.add_argument(
        "--task_list",
        "--task-list",
        default=str(DEFAULT_TASK_LIST),
        help="Task list, one .../<suite>/<task>.bddl per line.",
    )
    parser.add_argument(
        "--tasks", nargs="+", help="Only regenerate these tasks of the task list (bddl names)."
    )
    parser.add_argument(
        "--max_demos",
        "--max-demos",
        type=int,
        help="Only replay the first N demos of each task (for testing).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate tasks whose output file exists. By default they are skipped, so that an "
        "interrupted run can be resumed with the same command.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = read_task_list(args.task_list)
    if args.tasks:
        selected = {Path(name).stem for name in args.tasks}
        unknown = selected - {task for _, task in tasks}
        if unknown:
            raise SystemExit(f"Tasks not in {args.task_list}: {sorted(unknown)}")
        tasks = [(suite, task) for suite, task in tasks if task in selected]

    # Check all inputs and existing outputs before replaying anything.
    todo, skipped, missing, mismatched = [], [], [], []
    for suite, task in tasks:
        raw_path, out_path = task_paths(args, suite, task)
        if os.path.exists(out_path) and not args.overwrite:
            if written_with_same_settings(out_path, args):
                skipped.append(task)
                continue
            mismatched.append(out_path)
        if not os.path.exists(raw_path):
            missing.append(raw_path)
        todo.append((suite, task))
    if missing:
        raise SystemExit("Raw demo files not found:\n  " + "\n  ".join(missing))
    if mismatched:
        raise SystemExit(
            "These existing files do not match the --max_demos setting of this run "
            "(use --overwrite or another --output_dir):\n  " + "\n  ".join(mismatched)
        )
    if skipped:
        print(
            f"Skipping {len(skipped)} of {len(tasks)} tasks: their output files exist "
            "(--overwrite regenerates them)."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    total_kept = total_replayed = 0
    for n, (suite, task) in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] {suite}/{task}", flush=True)
        num_kept, num_demos = regenerate_task(suite, task, args)
        total_kept += num_kept
        total_replayed += num_demos
        print(f"[{n}/{len(todo)}] {task}: kept {num_kept}/{num_demos} demos", flush=True)
        if num_kept == 0:
            print(f"WARNING: no demo of {task} was kept.", flush=True)
    print(
        f"Done: kept {total_kept}/{total_replayed} replayed demos in {args.output_dir} "
        f"(tasks: {len(todo)} replayed, {len(skipped)} skipped)",
        flush=True,
    )


if __name__ == "__main__":
    main()
