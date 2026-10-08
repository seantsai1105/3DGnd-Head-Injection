# Direct Action-Head Injection of A Grounded 3D Point Unlocks Spatial and Task Generalization

Shiang-Feng Tsai<sup>1</sup>, Jin-Cheng Jhang<sup>1</sup>, Yen-Ling Tai<sup>2</sup>,
Jia-Hong Lai<sup>1\*</sup>, Shih-Yun Wong<sup>1\*</sup>, Kang-Tung Hsu<sup>1\*</sup>,
Yi-Ting Chen<sup>2</sup>, Min Sun<sup>1</sup>

<sup>1</sup>National Tsing Hua University, <sup>2</sup>National Yang Ming Chiao Tung University,
<sup>\*</sup>equal contribution

**Conference on Robot Learning (CoRL) 2026** | [Paper](https://arxiv.org/abs/2606.27663) |
[Project page](https://3dgnd-head-injection.github.io)

Vision-language-action (VLA) models remain brittle when objects are moved or when a familiar scene
comes with a different instruction. We find that how a grounding signal is represented and injected
matters more than the signal itself: a 2-D target point lifted to 3-D and injected directly into
the action head unlocks spatial and task generalization. On LIBERO-PRO, this raises the average
success rate of GR00T-N1.6 from 31.2% to 77.5% under task perturbation and from 28.1% to 60.2%
under position perturbation.

**Method (3DGnd).** A 2-D grounding point of the current sub-goal target is lifted to 3-D with the
metric depth and the camera parameters. In simulation the point is an oracle: the innermost pixel of
the target's segmentation mask in the agent-view camera (for a placement region without a mask, the
projection of its center). The target is the object to pick, or the placement target while that
object is lifted more than 1 cm above its initial height. The displacement
d = p_target - p_gripper is encoded by a two-layer MLP and added to the timestep embedding of
GR00T N1.6's DiT action head, so it conditions the AdaLN modulation of every DiT block. The
vision-language backbone is unchanged.

This repository contains the code of the GR00T-N1.6 experiments on LIBERO and LIBERO-PRO. It is
based on [NVIDIA Isaac GR00T N1.6](https://github.com/NVIDIA/Isaac-GR00T) (commit `d483f00`) and
adds:

- a `grounding` observation modality: `grounding.target_delta` = d (3-D, in metres), stored in
  LeRobot datasets as the column `observation.grounding`;
- the grounding encoder of the DiT action head, enabled with `--use_grounding`;
- scripts that regenerate the LIBERO pick-and-place training data with the grounding target
  ([`scripts/data_preparation`](scripts/data_preparation/README.md));
- a LIBERO / LIBERO-PRO environment that computes the grounding target at every step, and the
  evaluation protocol of the paper ([`examples/LIBERO`](examples/LIBERO)).

**Contents:** [Results](#results) | [Repository layout](#repository-layout) |
[Installation](#installation) | [Data preparation](#data-preparation) | [Training](#training) |
[Checkpoints](#checkpoints) | [Evaluation](#evaluation) | [Troubleshooting](#troubleshooting) |
[Acknowledgements and license](#acknowledgements-and-license) | [Citation](#citation)

## Results

Success rates (%) from Table 1 of the paper, GR00T-N1.6 fine-tuned on the same data with and
without 3DGnd (50 episodes per task, 10 tasks per suite):

| Method | LIBERO<br>Object | LIBERO<br>Spatial | Task pert.<br>Object | Task pert.<br>Spatial | Position pert.<br>Object | Position pert.<br>Spatial |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| GR00T-N1.6 | 93.8 | 93.2 | 10.0 | 52.4 | 29.2 | 27.0 |
| GR00T-N1.6 + Ours | 99.6 | 92.4 | 70.4 | 84.6 | 79.4 | 41.0 |

Initial object placements are randomized at every reset of the evaluation environment, so your own
evaluation runs will give slightly different numbers; small deviations from this table are normal.

LIBERO-Object and LIBERO-Spatial are in distribution (their tasks are part of the training data).
Task perturbation and position perturbation are the LIBERO-PRO suites
`libero_pro_{object,spatial}_task` and `libero_pro_{object,spatial}_swap`. The protocol is described
under [Evaluation](#evaluation).

## Repository layout

```text
gr00t/                         GR00T N1.6 package with the grounding modality
  data/ model/ policy/         grounding input: loading, normalization, encoder, policy
  eval/run_gr00t_server.py     policy server
  eval/rollout_policy.py       rollouts of one LIBERO / LIBERO-PRO task
  eval/sim/LIBERO/             3DGnd LIBERO environment, setup_libero.sh, client requirements
examples/LIBERO/               modality config, fine-tuning and evaluation scripts
scripts/data_preparation/      regeneration of the training data
external_dependencies/LIBERO/  git submodule: our LIBERO fork with the 3DGnd changes and LIBERO-PRO
```

## Installation

```bash
git clone --recurse-submodules https://github.com/seantsai1105/3DGnd-Head-Injection.git
cd 3DGnd-Head-Injection
```

All further commands run from the repository root. We tested the release on Ubuntu 22.04 with
NVIDIA RTX A6000 GPUs. The model and the simulators need different package versions, so there are
three Python environments:

| Environment | Used for | Setup |
|---|---|---|
| GR00T (`.venv`) | training, policy server | `uv sync` |
| LIBERO client (`gr00t/eval/sim/LIBERO/libero_uv/.venv`) | evaluation rollouts | `setup_libero.sh` |
| Data | regenerating the training data (optional) | [data preparation README](scripts/data_preparation/README.md#setup) |

Evaluating a checkpoint needs only the first two environments (see [Checkpoints](#checkpoints) and
[Evaluation](#evaluation)); data preparation and training are optional.

### GR00T environment

Install [uv](https://docs.astral.sh/uv/) (v0.8.4 or newer, which reads the flash-attn build settings
in `pyproject.toml`), then:

```bash
uv sync --python 3.10
```

The package versions are pinned (Python 3.10, PyTorch 2.7.1, transformers 4.51.3, flash-attn
2.7.4.post1, DeepSpeed 0.17.6). The release was verified with these versions; other versions or
GPUs can change the predicted actions slightly.

### LIBERO evaluation client

Prerequisites: git, uv, EGL, and cmake, make and a C++ compiler to build `egl-probe` (a robomimic
dependency without a wheel). On Ubuntu:

```bash
sudo apt install libegl1-mesa-dev libglu1-mesa cmake build-essential
bash gr00t/eval/sim/LIBERO/setup_libero.sh
```

`setup_libero.sh` checks out the git submodule `external_dependencies/LIBERO` at the pinned commit
(also after a clone without `--recurse-submodules`; in a source archive without git metadata it
clones the URL from `.gitmodules`), writes the repository-local LIBERO config
`gr00t/eval/sim/LIBERO/libero_config/` (your `~/.libero` is not touched), builds the client venv
`gr00t/eval/sim/LIBERO/libero_uv/.venv` with the package versions of the paper evaluation (pinned
in `gr00t/eval/sim/LIBERO/requirements-eval.txt`, e.g. MuJoCo 3.8.0, robosuite 1.4.0) and runs a
smoke test. It is safe to re-run. Rendering is headless (EGL); no X server is needed.

The submodule is our [LIBERO fork](https://github.com/3dgnd-head-injection/LIBERO/tree/3dgnd):
LIBERO with the sub-goal switch of 3DGnd, the placement targets of 19 LIBERO-90 tasks and the four
LIBERO-PRO suites, documented in its
[`3DGND.md`](https://github.com/3dgnd-head-injection/LIBERO/blob/3dgnd/3DGND.md).

### Data environment

Only needed to regenerate the training data: LIBERO and its config from
`bash gr00t/eval/sim/LIBERO/setup_libero.sh --libero-only` (needs only git) and a Python 3.10
environment from `scripts/data_preparation/requirements-data.txt`, as described in the
[data preparation README](scripts/data_preparation/README.md#setup).

## Data preparation

The training data are built from the LIBERO demonstrations of 69 pick-and-place tasks (49 from
LIBERO-90, 10 from LIBERO-Object, 10 from LIBERO-Spatial; listed in
[`libero_pick_and_place_tasks.txt`](scripts/data_preparation/libero_pick_and_place_tasks.txt)),
replayed in our LIBERO fork to record the grounding target in every frame. The paper reports 69
training tasks, but we later found that all replays of one task, LIBERO-90 `LIVING_ROOM_SCENE2`
"pick up the butter and put it in the basket", fail. The training data therefore contain 68 tasks
(3,092 episodes, 425,471 frames).

The [data preparation README](scripts/data_preparation/README.md) has the setup (LIBERO fork,
data environment, NVIDIA EGL rendering, 48 GB of raw LIBERO demonstrations from Hugging Face) and
the commands of the three steps:

1. `regenerate_libero_dataset_pick_and_place.py` replays the demonstrations and writes one HDF5 file
   per task (can be split over several processes with `--tasks`).
2. `convert_libero_to_lerobot.py` converts the HDF5 files to a LeRobot v3.0 dataset.
3. `convert_v3_to_v2.py` converts it to LeRobot v2.1, the format GR00T reads; then `modality.json`
   is copied into its `meta/` directory.

The result, `data/libero_pick_and_place_3dgnd`, is the training dataset.

**Dataset format.** A 3DGnd dataset is a GR00T LeRobot v2.1 dataset with the extra column
`observation.grounding` (float32, 3 values: `target_delta` = d = p_target - p_gripper in metres),
mapped in `meta/modality.json` by
`"grounding": {"target_delta": {"start": 0, "end": 3, "original_key": "observation.grounding"}}`.
The LIBERO data further contain `observation.state` (8 values: end-effector position, end-effector
axis-angle under the keys `roll`, `pitch`, `yaw`, and the two gripper joint positions), `action`
(the raw 7-D LIBERO action) and the 256x256 videos `observation.images.image` and
`observation.images.wrist_image` at 20 fps; see
[`modality.json`](scripts/data_preparation/modality.json).

## Training

```bash
DATASET_PATH=data/libero_pick_and_place_3dgnd uv run bash examples/LIBERO/finetune_libero_3dgnd.sh
```

[`finetune_libero_3dgnd.sh`](examples/LIBERO/finetune_libero_3dgnd.sh) fine-tunes
[nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) with `--use_grounding`, the
modality config [`libero_3dgnd_config.py`](examples/LIBERO/libero_3dgnd_config.py) and the
embodiment tag `NEW_EMBODIMENT`, using the hyper-parameters of the paper checkpoint: 2 GPUs x 192
samples (batch size 384), 20k steps, AdamW with learning rate 1e-4, cosine schedule with 5% warmup,
weight decay 1e-5 and color-jitter augmentation. A checkpoint is written every 1,000 steps; the
paper evaluates `checkpoint-20000`.

| Variable | Default | Meaning |
|---|---|---|
| `DATASET_PATH` | required | LeRobot v2.1 dataset with `meta/modality.json` |
| `OUTPUT_DIR` | `./outputs/libero_3dgnd` | checkpoint directory |
| `NUM_GPUS` | `2` | number of GPUs |
| `GRAD_ACCUM` | `1` | gradient accumulation steps |
| `BASE_MODEL_PATH` | `nvidia/GR00T-N1.6-3B` | base model (Hugging Face id or local directory) |
| `MASTER_PORT` | `29500` | torchrun port |

- The effective batch size stays 384 for any `NUM_GPUS` x `GRAD_ACCUM` that divides 384 (every GPU
  then processes 384 / (`NUM_GPUS` x `GRAD_ACCUM`) samples per forward/backward pass).
- Extra arguments are passed on to `gr00t/experiment/launch_finetune.py`, e.g. `--use_wandb` or
  `--save_total_limit 3`.
- The paper checkpoint was fine-tuned from revision `d0814e7` of the base model (the main branch
  at the time of the release). To pin it, download that revision and set `BASE_MODEL_PATH`:

  ```bash
  uv run hf download nvidia/GR00T-N1.6-3B --revision d0814e7ecb19202e7c8468b46098b0b7ef3a6d61 \
      --local-dir checkpoints/GR00T-N1.6-3B
  BASE_MODEL_PATH=checkpoints/GR00T-N1.6-3B DATASET_PATH=data/libero_pick_and_place_3dgnd \
      uv run bash examples/LIBERO/finetune_libero_3dgnd.sh
  ```

- GR00T writes the normalization statistics into `meta/` of the dataset on first use, so the
  dataset directory must be writable.
- For other data, add a `grounding` modality to the modality config as in `libero_3dgnd_config.py`
  and pass `--use_grounding` (and `--grounding_dim` if the grounding is not 3-D). The grounding
  encoder is trained together with the action-head projectors.

Open-loop check of a checkpoint on training episodes (plots of predicted and recorded actions are
written to `/tmp/open_loop_eval/`):

```bash
uv run python gr00t/eval/open_loop_eval.py --dataset-path data/libero_pick_and_place_3dgnd \
    --model-path outputs/libero_3dgnd/checkpoint-20000 --embodiment-tag NEW_EMBODIMENT \
    --traj-ids 0 --action-horizon 16
```

## Checkpoints

The GR00T-N1.6 + 3DGnd checkpoint of the paper (trained on the 68-task data) will be released at
CHECKPOINT_URL_TBD. The examples below assume it in `checkpoints/3dgnd_libero`.

### Using the policy in your own code

```python
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

policy = Gr00tPolicy(EmbodimentTag.NEW_EMBODIMENT, "checkpoints/3dgnd_libero", device="cuda")
action, _ = policy.get_action(obs)
```

`obs` is a batch of B observations of one time step each:

| Key | Type and shape | Content |
|---|---|---|
| `obs["video"]["image"]`, `obs["video"]["wrist_image"]` | uint8 (B, 1, 256, 256, 3) | agent-view and wrist RGB images, upright |
| `obs["state"][k]` for k in `x`, `y`, `z` | float32 (B, 1, 1) | end-effector position (m) |
| `obs["state"][k]` for k in `roll`, `pitch`, `yaw` | float32 (B, 1, 1) | end-effector orientation as an axis-angle vector |
| `obs["state"]["gripper"]` | float32 (B, 1, 2) | gripper joint positions |
| `obs["grounding"]["target_delta"]` | float32 (B, 1, 3) | d = p_target - p_gripper (m) |
| `obs["language"]["annotation.human.action.task_description"]` | B lists of one string | instruction |

`action[k]` is a float32 array (B, 16, 1) for k in `x`, `y`, `z`, `roll`, `pitch`, `yaw`,
`gripper`: 16 steps of the 7-D LIBERO action (end-effector delta for the OSC_POSE controller,
gripper -1 open / +1 close), of which the evaluation executes the first 8.

## Evaluation

The evaluation uses GR00T's server-client setup: the policy runs in the GR00T environment, the
simulators in the LIBERO client environment.

1. Start the policy server (GR00T environment). The model path can also be
   `outputs/libero_3dgnd/checkpoint-20000` from training.

   ```bash
   uv run python gr00t/eval/run_gr00t_server.py --model-path checkpoints/3dgnd_libero \
       --embodiment-tag NEW_EMBODIMENT --use-sim-policy-wrapper --port 5555
   ```

2. In a second terminal, run the paper protocol (it uses the client venv and the LIBERO config made
   by `setup_libero.sh`). It ends with a table of the success rates next to the paper's;
   `summarize_results.py` prints it again at any time, also for an unfinished run:

   ```bash
   OUTPUT_DIR=outputs/eval_libero_3dgnd bash examples/LIBERO/eval_libero_3dgnd.sh
   python3 examples/LIBERO/summarize_results.py outputs/eval_libero_3dgnd
   ```

**Protocol.** 6 suites x 10 tasks x 50 episodes (500 episodes per suite): `libero_object`,
`libero_spatial`, the task-perturbation suites `libero_pro_object_task` and
`libero_pro_spatial_task`, and the position-perturbation suites `libero_pro_object_swap` and
`libero_pro_spatial_swap`. An episode has at most 512 environment steps; the policy predicts 16
actions per call, of which 8 are executed. LIBERO suites use the benchmark instruction, LIBERO-PRO
suites the instruction of the task's BDDL file. The environment computes the grounding at every
step from the simulator's agent-view segmentation and depth, as in the training data. The reported
numbers were obtained with `--n_envs 10` (10 simulators per task, the default of the script).
The evaluation environment re-samples the object placements at every reset (there are no fixed
initial states) and the placements are not seeded, so every evaluation run sees different initial
states. Re-running the evaluation therefore gives slightly different success rates than the
[Results](#results) table; this is expected (the standard error is up to about 2 points per suite,
and individual tasks vary more).

The script is configured with environment variables (`--help` lists them, `--dry-run` lists the 60
tasks):

| Variable | Default | Meaning |
|---|---|---|
| `POLICY_HOST`, `POLICY_PORT` | `127.0.0.1`, `5555` | policy server |
| `OUTPUT_DIR` | required | `results.jsonl` and per-task logs (one directory per checkpoint) |
| `SUITES` | all 6 suites | subset of the suites above |
| `N_EPISODES` | `50` | episodes per task |
| `N_ENVS` | `10` | parallel simulators per task |
| `MAX_EPISODE_STEPS`, `N_ACTION_STEPS` | `512`, `8` | episode length, executed actions per call |
| `JOBS` | `1` | tasks evaluated in parallel |
| `MAX_ATTEMPTS` | `3` | runs of a task before it counts as failed |
| `TASK_TIMEOUT` | none | time limit of one run, e.g. `3h` |
| `VIDEO_DIR` | none | save rollout videos to `VIDEO_DIR/<task>/` |
| `LIBERO_PYTHON` | `gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python` | client Python |

- Every task appends one line to `OUTPUT_DIR/results.jsonl`, and tasks that already have one are
  skipped, so running the script again resumes an interrupted evaluation. The script refuses to
  resume when `results.jsonl` holds results of other settings. Use a new `OUTPUT_DIR` for every
  checkpoint and setting (the checkpoint itself is not recorded).
- A task whose run fails (e.g. a crashed simulator, see [Troubleshooting](#troubleshooting)) is run
  again, up to `MAX_ATTEMPTS` runs in total; the logs of failed runs are kept as
  `OUTPUT_DIR/logs/<task>.attempt<k>.log`. At the end, the script lists the tasks without a result
  and exits with status 1; run it again to evaluate them.
- For a policy server on another machine, start it with `--host 0.0.0.0` and set `POLICY_HOST`.
- `summarize_results.py` accepts several output directories (shown side by side) and `--per_task`.

To evaluate a single task, call the rollout script directly. Task ids are `libero_sim/<task>` for
LIBERO-Object and LIBERO-Spatial and `libero_sim/<suite>_<task>` for LIBERO-PRO:

```bash
export LIBERO_CONFIG_PATH=$PWD/gr00t/eval/sim/LIBERO/libero_config
gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --env_name libero_sim/libero_pro_object_task_pick_up_the_alphabet_soup_and_place_it_in_the_basket \
    --n_episodes 50 --n_envs 10 --output_dir outputs/eval_single_task
```

## Troubleshooting

- **Simulator worker crash.** Rarely, a rollout dies inside a simulator worker with an error such as
  `TypeError: 'cell' object is not callable`, `TypeError: unhashable type: 'dict'` or
  `descriptor 'get' for 'dict' objects doesn't apply to a 'numpy.int32' object`. These are
  intermittent crashes inside the LIBERO / robosuite / MuJoCo stack of the worker (they surface at
  the first render after a simulator reset and also occurred with the code used for the paper); the
  policy is not involved. The evaluation script re-runs such tasks automatically.
- **Client waits forever.** If the policy server dies during a run, the client blocks. For
  unattended evaluations, set `TASK_TIMEOUT` (e.g. `TASK_TIMEOUT=3h`); the script also stops
  starting tasks once the server is unreachable.
- **Rendering.** The simulators render headless with EGL (`MUJOCO_GL=egl`); `xvfb` is not needed.
  robosuite renders on the GPU given by `MUJOCO_EGL_DEVICE_ID` (if `CUDA_VISIBLE_DEVICES` is
  non-empty, it must list that GPU, otherwise robosuite stops with an `AssertionError`), else on
  the first GPU listed in `CUDA_VISIBLE_DEVICES`, else on GPU 0. An empty `CUDA_VISIBLE_DEVICES`
  without `MUJOCO_EGL_DEVICE_ID` makes it fail with `ValueError: invalid literal for int()`.
- **LIBERO config.** LIBERO reads its paths from `$LIBERO_CONFIG_PATH/config.yaml`, or from
  `~/.libero/config.yaml` if `LIBERO_CONFIG_PATH` is not set; the latter may point to another LIBERO
  installation. If the file does not exist, LIBERO asks interactively and creates it. The evaluation
  script uses the repository-local config; export
  `LIBERO_CONFIG_PATH=$PWD/gr00t/eval/sim/LIBERO/libero_config` when you call `rollout_policy.py`
  or LIBERO directly.
- **CMake 4.** `egl-probe` predates CMake 4. `setup_libero.sh` sets
  `CMAKE_POLICY_VERSION_MINIMUM=3.5`, so any CMake version works; set it yourself when you install
  `gr00t/eval/sim/LIBERO/requirements-eval.txt` by hand.
- **uv cache.** `uv sync` and `setup_libero.sh` install several GB of packages through uv's cache
  (default `~/.cache/uv`; set `UV_CACHE_DIR` to move it). If the cache and the repository are on
  different file systems, uv warns that it cannot hardlink and copies the files instead;
  `export UV_LINK_MODE=copy` silences the warning.
- **Harmless messages.** robosuite's "No private macro file found" warning, the Gym deprecation
  notice and LIBERO's "[info] using task orders" lines in the client logs, and the albumentations
  update notice in the GR00T environment can be ignored.

## Acknowledgements and license

This code builds on [NVIDIA Isaac GR00T N1.6](https://github.com/NVIDIA/Isaac-GR00T) (commit
`d483f00`) and is released under the NVIDIA License in [`LICENSE`](LICENSE), which permits
non-commercial research use only. The checkpoint is fine-tuned from
[nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) and subject to its license.
Third-party components keep their licenses:

- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (MIT): our
  [fork](https://github.com/3dgnd-head-injection/LIBERO/tree/3dgnd) (the git submodule
  `external_dependencies/LIBERO`) adds the 3DGnd changes for data generation and evaluation.
- [LIBERO-PRO](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro) (MIT): the task files of the
  four LIBERO-PRO suites are included in the LIBERO fork, renamed and edited as documented in its
  [`3DGND.md`](https://github.com/3dgnd-head-injection/LIBERO/blob/3dgnd/3DGND.md#libero-pro-suites),
  which also lists the licenses and the BibTeX entries of LIBERO and LIBERO-PRO.
- [OpenVLA](https://github.com/openvla/openvla) (MIT, [`licenses/OpenVLA.txt`](licenses/OpenVLA.txt)):
  the demonstration replay and no-op filtering of the data regeneration follow its
  `regenerate_libero_dataset.py`.
- [robosuite](https://github.com/ARISE-Initiative/robosuite) (MIT,
  [`licenses/robosuite.txt`](licenses/robosuite.txt)): the simulation framework of LIBERO;
  `quat2axisangle` is taken from it.

## Citation

```bibtex
@inproceedings{tsai2026direct,
  title     = {Direct Action-Head Injection of A Grounded {3D} Point Unlocks Spatial and Task Generalization},
  author    = {Tsai, Shiang-Feng and Jhang, Jin-Cheng and Tai, Yen-Ling and Lai, Jia-Hong and Wong, Shih-Yun and Hsu, Kang-Tung and Chen, Yi-Ting and Sun, Min},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```

Please also cite [GR00T N1](https://arxiv.org/abs/2503.14734), LIBERO and LIBERO-PRO if you use
this code (BibTeX entries of the latter two:
[`3DGND.md`](https://github.com/3dgnd-head-injection/LIBERO/blob/3dgnd/3DGND.md#license-and-attribution)
of the LIBERO fork).
