"""Convert the regenerated LIBERO pick-and-place HDF5 files to a LeRobot v3.0 dataset.

Episodes are written in task-list order and, within a task, in lexicographic demo order (demo_0,
demo_1, demo_10, ...), which reproduces the episode and task indices of the original training
data. Features:

    observation.images.image        agentview camera, 256x256 video
    observation.images.wrist_image  eye-in-hand camera, 256x256 video
    observation.state               (8,) eef xyz, eef axis-angle, gripper qpos
    observation.grounding           (3,) target_delta = p_target - p_eef
    action                          (7,) raw LIBERO action
    task                            bddl language instruction (lower case)

GR00T reads LeRobot v2.1: convert the result with ``convert_v3_to_v2.py`` and copy
``modality.json`` to ``<root>/<repo_id>/meta/`` (see README.md).

Usage:
    python scripts/data_preparation/convert_libero_to_lerobot.py \\
        --hdf5_dir <regenerated HDF5 dir> --root <output parent dir> --repo_id <dataset name>
"""

import argparse
import os
from pathlib import Path

import h5py
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np


DEFAULT_TASK_LIST = Path(__file__).with_name("libero_pick_and_place_tasks.txt")
FPS = 20
# Video features carry no "info": LeRobot fills it in from the encoded files (AV1 by default).
FEATURES = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.wrist_image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_0", "gripper_1"],
    },
    "observation.grounding": {
        "dtype": "float32",
        "shape": (3,),
        "names": ["delta_x", "delta_y", "delta_z"],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    },
}


def read_task_list(path):
    """Return the (suite, task) pairs of a task list with lines ``.../<suite>/<task>.bddl``."""
    with open(path) as f:
        bddl_paths = [Path(line.strip()) for line in f if line.strip()]
    return [(p.parent.name, p.stem) for p in bddl_paths]


def has_target_delta(hdf5_path):
    """Whether every demo of an HDF5 file has the grounding signal ``obs/target_delta``."""
    with h5py.File(hdf5_path, "r") as f:
        return all("target_delta" in demo["obs"] for demo in f["data"].values())


def add_episode(dataset, demo, language):
    obs = demo["obs"]
    grounding = obs["target_delta"][()].astype(np.float32)
    states = demo["robot_states"][()].astype(np.float32)
    actions = demo["actions"][()].astype(np.float32)
    images = obs["agentview_rgb"][()]
    wrist_images = obs["eye_in_hand_rgb"][()]

    for t in range(len(actions)):
        dataset.add_frame(
            {
                "observation.images.image": images[t],
                "observation.images.wrist_image": wrist_images[t],
                "observation.state": states[t],
                "observation.grounding": grounding[t],
                "action": actions[t],
                "task": language,
            }
        )
    dataset.save_episode()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--hdf5_dir", "--hdf5-dir", required=True, help="Directory with the <task>_demo.hdf5 files."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Parent directory; the dataset is written to <root>/<repo_id>.",
    )
    parser.add_argument("--repo_id", "--repo-id", required=True, help="Dataset name.")
    parser.add_argument(
        "--task_list",
        "--task-list",
        default=str(DEFAULT_TASK_LIST),
        help="Task list, one .../<suite>/<task>.bddl per line; defines the episode order.",
    )
    parser.add_argument(
        "--tasks", nargs="+", help="Only convert these tasks of the task list (bddl names)."
    )
    parser.add_argument(
        "--image_writer_processes",
        "--image-writer-processes",
        type=int,
        default=0,
        help="LeRobot image writer processes (frames are written as PNG before video encoding).",
    )
    parser.add_argument(
        "--image_writer_threads",
        "--image-writer-threads",
        type=int,
        default=8,
        help="LeRobot image writer threads.",
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

    # Check the inputs before the dataset directory is created.
    hdf5_paths = [os.path.join(args.hdf5_dir, f"{task}_demo.hdf5") for _, task in tasks]
    missing = [path for path in hdf5_paths if not os.path.exists(path)]
    if missing:
        raise SystemExit("HDF5 files not found:\n  " + "\n  ".join(missing))
    without_grounding = [path for path in hdf5_paths if not has_target_delta(path)]
    if without_grounding:
        raise SystemExit(
            "HDF5 files without obs/target_delta (write them with "
            "regenerate_libero_dataset_pick_and_place.py):\n  " + "\n  ".join(without_grounding)
        )
    output_dir = Path(args.root) / args.repo_id
    if output_dir.exists():
        raise SystemExit(
            f"{output_dir} already exists; delete it or choose another --root/--repo_id."
        )
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_dir,
        fps=FPS,
        robot_type="panda",
        features=FEATURES,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    for (suite, task), hdf5_path in zip(tasks, hdf5_paths):
        with h5py.File(hdf5_path, "r") as f:
            data = f["data"]
            language = data.attrs["language"]
            demo_keys = sorted(data)  # lexicographic (demo_0, demo_1, demo_10, ...)
            for key in demo_keys:
                add_episode(dataset, data[key], language)
        print(f'{suite}/{task}: {len(demo_keys)} episodes, "{language}"', flush=True)

    dataset.finalize()
    print(f"Wrote {dataset.meta.total_episodes} episodes to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
