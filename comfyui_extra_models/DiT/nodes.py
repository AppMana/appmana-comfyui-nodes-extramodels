import json
from importlib import resources

import torch

from comfy.model_downloader import get_or_download, add_known_models, get_filename_list_with_downloadable, \
    KNOWN_CHECKPOINTS
from comfy.model_downloader_types import HuggingFile
from .conf import dit_conf
from .loader import load_dit

KNOWN_DIT_MODELS = [
    HuggingFile("city96/DiT", "DiT-XL-2-256x256-fp16.safetensors"),
    HuggingFile("city96/DiT", "DiT-XL-2-512x512-fp16.safetensors"),
]
add_known_models("checkpoints", KNOWN_CHECKPOINTS, *KNOWN_DIT_MODELS)


class DitCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (get_filename_list_with_downloadable("checkpoints", KNOWN_DIT_MODELS),),
                "model": (list(dit_conf.keys()),),
                "image_size": ([256, 512],),
                # "num_classes": ("INT", {"default": 1000, "min": 0,}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "ExtraModels/DiT"
    TITLE = "DitCheckpointLoader"

    def load_checkpoint(self, ckpt_name, model, image_size):
        ckpt_path = get_or_download("checkpoints", ckpt_name, KNOWN_DIT_MODELS)
        model_conf = dit_conf[model]
        model_conf["unet_config"]["input_size"] = image_size // 8
        # model_conf["unet_config"]["num_classes"] = num_classes
        dit = load_dit(
            model_path=ckpt_path,
            model_conf=model_conf,
        )
        return (dit,)


# todo: this needs frontend code to display properly
def get_label_data():
    label_data = {0: "None"}
    with resources.open_text("comfyui_extra_models.DiT.labels", "imagenet1000.json") as file:
        label_data.update(json.load(file))
    return label_data


label_data = get_label_data()


class DiTCondLabelSelect:
    @classmethod
    def INPUT_TYPES(s):
        global label_data
        return {
            "required": {
                "model": ("MODEL",),
                "label_name": (list(label_data.values()),),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("class",)
    FUNCTION = "cond_label"
    CATEGORY = "ExtraModels/DiT"
    TITLE = "DiTCondLabelSelect"

    def cond_label(self, model, label_name):
        global label_data
        class_labels = [int(k) for k, v in label_data.items() if v == label_name]
        y = torch.tensor([[class_labels[0]]]).to(torch.int)
        return ([[y, {}]],)


class DiTCondLabelEmpty:
    @classmethod
    def INPUT_TYPES(s):
        global label_data
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("empty",)
    FUNCTION = "cond_empty"
    CATEGORY = "ExtraModels/DiT"
    TITLE = "DiTCondLabelEmpty"

    def cond_empty(self, model):
        # [ID of last class + 1] == [num_classes]
        y_null = model.model.model_config.unet_config["num_classes"]
        y = torch.tensor([[y_null]]).to(torch.int)
        return ([[y, {}]],)


NODE_CLASS_MAPPINGS = {
    "DitCheckpointLoader": DitCheckpointLoader,
    "DiTCondLabelSelect": DiTCondLabelSelect,
    "DiTCondLabelEmpty": DiTCondLabelEmpty,
}
