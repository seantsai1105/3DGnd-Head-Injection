# Direct Action-Head Injection of A Grounded 3D Point Unlocks Spatial and Task Generalization

Shiang-Feng Tsai<sup>1</sup>, Jin-Cheng Jhang<sup>1</sup>, Yen-Ling Tai<sup>2</sup>,
Jia-Hong Lai<sup>1\*</sup>, Shih-Yun Wong<sup>1\*</sup>, Kang-Tung Hsu<sup>1\*</sup>,
Yi-Ting Chen<sup>2</sup>, Min Sun<sup>1</sup>

<sup>1</sup>National Tsing Hua University, <sup>2</sup>National Yang Ming Chiao Tung University,
<sup>\*</sup>equal contribution

[Paper](https://arxiv.org/abs/2606.27663) | [Project page](https://3dgnd-head-injection.github.io)

## Contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements and license](#acknowledgements-and-license)
- [Citation](#citation)

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

All commands run from the repository root. We tested the release on Ubuntu 22.04 with NVIDIA RTX
A6000 GPUs. The model and the LIBERO simulator need different package versions, so they live in two
environments: the GR00T environment (training and policy server) and the LIBERO client
environment (evaluation rollouts). Regenerating the training data is optional and uses a third
environment, described in the [data preparation README](scripts/data_preparation/README.md#setup).

### GR00T environment

Install [uv](https://docs.astral.sh/uv/) (v0.8.4 or newer), then:

```bash
uv sync --python 3.10
```

This installs the pinned versions (Python 3.10, PyTorch 2.7.1, transformers 4.51.3, flash-attn
2.7.4.post1, DeepSpeed 0.17.6) into `.venv`.

### LIBERO client environment

```bash
sudo apt install libegl1-mesa-dev libglu1-mesa cmake build-essential
bash gr00t/eval/sim/LIBERO/setup_libero.sh
```

`setup_libero.sh` checks out the git submodule `external_dependencies/LIBERO` (our
[LIBERO fork](https://github.com/3dgnd-head-injection/LIBERO/tree/3dgnd) with the 3DGnd changes and
the LIBERO-PRO suites, see its
[`3DGND.md`](https://github.com/3dgnd-head-injection/LIBERO/blob/3dgnd/3DGND.md)), writes the
LIBERO config `gr00t/eval/sim/LIBERO/libero_config/` and builds the client venv
`gr00t/eval/sim/LIBERO/libero_uv/.venv` with the pinned versions of
`gr00t/eval/sim/LIBERO/requirements-eval.txt`. Rendering is headless (EGL), so no X server is
needed.

## Data preparation

The training data are built from the LIBERO demonstrations of 69 pick-and-place tasks (49 from
LIBERO-90, 10 from LIBERO-Object, 10 from LIBERO-Spatial), replayed in our LIBERO fork to record
the grounding target in every frame. The
[data preparation README](scripts/data_preparation/README.md) describes the setup, the dataset
format and the three steps (replay, conversion to LeRobot v3.0, conversion to LeRobot v2.1). The
result, `data/libero_pick_and_place_3dgnd`, is the training dataset.

## Training

```bash
DATASET_PATH=data/libero_pick_and_place_3dgnd uv run bash examples/LIBERO/finetune_libero_3dgnd.sh
```

[`finetune_libero_3dgnd.sh`](examples/LIBERO/finetune_libero_3dgnd.sh) fine-tunes
[nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) with `--use_grounding`, the
modality config [`libero_3dgnd_config.py`](examples/LIBERO/libero_3dgnd_config.py) and the
embodiment tag `NEW_EMBODIMENT`, using the hyper-parameters of the paper checkpoint: 2 GPUs x 192
samples (batch size 384), 20k steps, AdamW with learning rate 1e-4, cosine schedule with 5% warmup,
weight decay 1e-5 and color-jitter augmentation. A checkpoint is written every 1,000 steps to
`OUTPUT_DIR` (default `outputs/libero_3dgnd`); the paper evaluates `checkpoint-20000`. The script's
header lists its other settings (`NUM_GPUS`, `GRAD_ACCUM`, `BASE_MODEL_PATH`, ...).

## Evaluation

The GR00T-N1.6 + 3DGnd checkpoint of the paper will be released at CHECKPOINT_URL_TBD; the commands
below assume it in `checkpoints/3dgnd_libero` (a checkpoint from [Training](#training), e.g.
`outputs/libero_3dgnd/checkpoint-20000`, works the same way). The evaluation uses GR00T's
server-client setup: the policy runs in the GR00T environment, the simulators in the LIBERO client
environment.

1. Start the policy server (GR00T environment):

   ```bash
   uv run python gr00t/eval/run_gr00t_server.py --model-path checkpoints/3dgnd_libero \
       --embodiment-tag NEW_EMBODIMENT --use-sim-policy-wrapper --port 5555
   ```

2. In a second terminal, run the paper protocol. It ends with a table of the success rates next to
   the paper's; `summarize_results.py` prints it again at any time, also for an unfinished run:

   ```bash
   OUTPUT_DIR=outputs/eval_libero_3dgnd bash examples/LIBERO/eval_libero_3dgnd.sh
   python3 examples/LIBERO/summarize_results.py outputs/eval_libero_3dgnd
   ```

Protocol (Table 1 of the paper):

- 6 suites x 10 tasks x 50 episodes: `libero_object` and `libero_spatial` (in distribution), the
  task-perturbation suites `libero_pro_object_task` and `libero_pro_spatial_task`, and the
  position-perturbation suites `libero_pro_object_swap` and `libero_pro_spatial_swap`.
- At most 512 environment steps per episode; the policy predicts 16 actions per call, of which 8
  are executed; 10 parallel simulators per task.
- The grounding target is computed at every step from the simulator's agent-view segmentation and
  depth, as in the training data.
- Object placements are re-sampled at every reset and are not seeded, so every run sees different
  initial states and the success rates differ slightly between runs (the standard error is up to
  about 2 points per suite).

Success rates (%) of the paper (GR00T-N1.6 fine-tuned on the same data with and without 3DGnd):

| Method | LIBERO<br>Object | LIBERO<br>Spatial | Task pert.<br>Object | Task pert.<br>Spatial | Position pert.<br>Object | Position pert.<br>Spatial |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| GR00T-N1.6 | 93.8 | 93.2 | 10.0 | 52.4 | 29.2 | 27.0 |
| GR00T-N1.6 + Ours | 99.6 | 92.4 | 70.4 | 84.6 | 79.4 | 41.0 |

Notes on the evaluation script (`--help` lists all settings, `--dry-run` lists the 60 tasks):

- Every task appends one line to `OUTPUT_DIR/results.jsonl`, and tasks that already have one are
  skipped, so running the script again resumes an interrupted evaluation. Use a new `OUTPUT_DIR`
  for every checkpoint.
- `SUITES`, `N_EPISODES`, `N_ENVS` and `JOBS` (tasks evaluated in parallel) select a subset or
  change the parallelism; `VIDEO_DIR` saves rollout videos.
- A task whose run fails (e.g. a crashed simulator, see [Troubleshooting](#troubleshooting)) is run
  again, up to `MAX_ATTEMPTS` (default 3) runs; the script lists the tasks without a result at the
  end and exits with status 1. Run it again to evaluate them.
- For a policy server on another machine, start it with `--host 0.0.0.0` and set `POLICY_HOST`.

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
