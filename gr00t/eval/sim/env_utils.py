from gr00t.data.embodiment_tags import EmbodimentTag


def get_embodiment_tag_from_env_name(env_name: str) -> EmbodimentTag:
    if env_name.startswith("libero_sim/"):
        # The 3DGnd LIBERO checkpoints are fine-tuned with the NEW_EMBODIMENT tag.
        return EmbodimentTag.NEW_EMBODIMENT

    return EmbodimentTag(env_name.split("/")[0])
