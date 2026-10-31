# Data preparation

These scripts build the 3DGnd LIBERO pick-and-place training data from the raw LIBERO
demonstrations:

| Step | Script | Output |
|---|---|---|
| 1 | `regenerate_libero_dataset_pick_and_place.py` | one HDF5 file per task with the replayed demos and the grounding target |
| 2 | `convert_libero_to_lerobot.py` | LeRobot v3.0 dataset |
| 3 | `convert_v3_to_v2.py` (unchanged copy of Isaac-GR00T's [`scripts/lerobot_conversion/convert_v3_to_v2.py`](https://github.com/NVIDIA/Isaac-GR00T/blob/d483f00b1c13116bda020bead9d16dca497b2f6d/scripts/lerobot_conversion/convert_v3_to_v2.py)) | LeRobot v2.1 dataset, the format GR00T reads |

`libero_pick_and_place_tasks.txt` lists the 69 tasks (49 LIBERO-90, 10 LIBERO-Object, 10
LIBERO-Spatial) in dataset order. The paper reports 69 training tasks, but we later found that all
replays of one task, LIVING_ROOM_SCENE2 "pick up the butter and put it in the basket", fail. The
data therefore contains 68 tasks: 3,092 episodes, 425,471 frames and 63 distinct instructions.

## Data

| Key | Shape | Content |
|---|---|---|
| `observation.images.image` | 256×256×3, 20 fps video | agentview camera |
| `observation.images.wrist_image` | 256×256×3, 20 fps video | eye-in-hand camera |
| `observation.state` | 8 | end-effector position, end-effector orientation (axis-angle, named `roll`, `pitch`, `yaw` in `modality.json`), gripper joint positions |
| `observation.grounding` | 3 | `target_delta` = p_target − p_eef |
| `action` | 7 | raw LIBERO action |
| `task` | – | instruction from the task's BDDL file |

`modality.json` (step 4 below) maps these columns to the GR00T modalities: `observation.grounding`
to `grounding.target_delta`, `observation.state` to the state keys `x`, `y`, `z`, `roll`, `pitch`,
`yaw`, `gripper`, and `action` to the action keys of the same names.

p_target is the placement target in every frame in which `grasped()` of our LIBERO fork reports the
object to pick more than 1 cm above its reference height (its height in the 12th frame), and the
object to pick in all other frames. It is the innermost visible pixel of the target's segmentation
mask, lifted to 3-D with the rendered depth. Targets without a visible mask (placement regions,
occluded objects) are lifted at the projection of their position instead. Demos that fail on replay
or never reach the placement stage are dropped. The replay itself (initial state, settle steps,
no-op and success filters) follows `regenerate_libero_dataset.py` of
[OpenVLA](https://github.com/openvla/openvla) (MIT License).

## Setup

1. LIBERO: `bash gr00t/eval/sim/LIBERO/setup_libero.sh --libero-only` checks out the git submodule
   `external_dependencies/LIBERO`, our LIBERO fork, and writes the LIBERO config
   `gr00t/eval/sim/LIBERO/libero_config/` (only `git` is needed). The fork adds the sub-goal switch
   `grasped()` and the placement targets of 19 LIBERO-90 tasks, both of which the data needs
   ([`3DGND.md`](https://github.com/3dgnd-head-injection/LIBERO/blob/3dgnd/3DGND.md)). Without
   `--libero-only`, the script also builds the evaluation client environment.
2. Create a Python 3.10 environment with the package versions that produced the original training
   data. `ffmpeg` must be on the `PATH` (`convert_v3_to_v2.py` calls it).

   ```bash
   conda create -n libero_data python=3.10 -y && conda activate libero_data
   pip install -r scripts/data_preparation/requirements-data.txt
   pip install numpy==1.22.4  # pip reports that opencv and rerun-sdk want numpy>=2; this is expected
   pip install --no-deps -e external_dependencies/LIBERO
   ```

3. Download the raw LIBERO demonstrations of the 69 tasks (48 GB) into `data/libero_raw` (in the
   data environment, which provides the `hf` command):

   ```bash
   sed -e 's|.*/bddl_files/||' -e 's|\.bddl$|_demo.hdf5|' \
       scripts/data_preparation/libero_pick_and_place_tasks.txt |
       xargs hf download --repo-type dataset --revision f13aa24a3da8c43c7225569f28c562979fa0e35a \
           --local-dir data/libero_raw yifengzhu-hf/LIBERO-datasets
   ```

Run the steps from the repository root, in a shell with:

```bash
export LIBERO_CONFIG_PATH=$PWD/gr00t/eval/sim/LIBERO/libero_config
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

Rendering must go through the NVIDIA EGL driver (last line); Mesa EGL renders slightly different
images, which also changes the targets. With this setup we regenerated demos of several tasks on an
RTX A6000: actions, robot states, grounding targets and agentview images were bit-identical to the
original training data. Eye-in-hand images matched except for rare pixels that differ by one grey
level, because the GPU rendering of that camera is not exactly reproducible.

## Steps

```bash
# 1. Replay the demos (one HDF5 file per task)
python scripts/data_preparation/regenerate_libero_dataset_pick_and_place.py \
    --libero_raw_data_dir data/libero_raw --output_dir data/libero_pick_and_place_hdf5

# 2. Convert to LeRobot v3.0 (writes data/libero_pick_and_place_3dgnd)
python scripts/data_preparation/convert_libero_to_lerobot.py \
    --hdf5_dir data/libero_pick_and_place_hdf5 --root data --repo_id libero_pick_and_place_3dgnd

# 3. Convert to LeRobot v2.1 in place (the v3.0 version moves to data/libero_pick_and_place_3dgnd_v3.0)
python scripts/data_preparation/convert_v3_to_v2.py --root data --repo-id libero_pick_and_place_3dgnd

# 4. Add the GR00T modality file
cp scripts/data_preparation/modality.json data/libero_pick_and_place_3dgnd/meta/
```

`data/libero_pick_and_place_3dgnd` is the training dataset (`DATASET_PATH` of
`examples/LIBERO/finetune_libero_3dgnd.sh`).

- Every task is replayed in its own environment, seeded with 0, so tasks are independent. Step 1
  can be split over several processes with `--tasks <task> ...`. It skips tasks whose output file
  exists, so an interrupted run is resumed by running the same command again. It stops if such a
  file was written with another `--max_demos` setting; `--overwrite` regenerates existing files.
  Step 2 converts all tasks of the list and stops if an HDF5 file is missing; give it the same
  `--tasks` to convert a subset.
- Within a task, the demos must be replayed in order, because LIBERO re-samples the fixture poses
  at every reset. `--max_demos N` therefore replays the first N demos (for a quick test; use a
  separate `--output_dir` and give step 2 that directory as `--hdf5_dir`).
- Step 2 writes the episodes in the order of the original training data: task-list order, then
  demo order `demo_0, demo_1, demo_10, …`.
