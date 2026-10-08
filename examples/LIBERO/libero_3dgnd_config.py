"""Modality config for the 3DGnd LIBERO pick-and-place data (embodiment tag NEW_EMBODIMENT).

`grounding.target_delta` is the 3-D displacement d = p_target - p_gripper (world frame, metres).
It is normalized with the dataset statistics like state and fed only to the action head (added to
the DiT timestep embedding); training with this config requires `--use_grounding`.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig


libero_3dgnd_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["image", "wrist_image"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    ),
    "grounding": ModalityConfig(
        delta_indices=[0],
        modality_keys=["target_delta"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(libero_3dgnd_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
