import torch

from comfy.cmd import folder_paths
from comfy.model_downloader import get_filename_list_with_downloadable, get_or_download
from comfy.model_downloader_types import HuggingFile
from .loader import load_t5
from ..utils.dtype import string_to_dtype

FOLDER_NAME = "t5"
KNOWN_T5_MODELS = [
    HuggingFile("city96/t5-v1_1-xxl-encoder-bf16",
                "model.safetensors",
                save_with_filename="t5-v1_1-xxl-encoder-bf16.safetensors")
]

dtypes = [
    "default",
    "auto (comfy)",
    "FP32",
    "FP16",
    # Note: remove these at some point
    "bnb8bit",
    "bnb4bit",
]
try:
    torch.float8_e5m2
except AttributeError:
    print("Torch version too old for FP8")
else:
    dtypes += ["FP8 E4M3", "FP8 E5M2"]


class T5v11Loader:
    @classmethod
    def INPUT_TYPES(s):
        devices = ["auto", "cpu", "gpu"]
        # hack for using second GPU as offload
        for k in range(1, torch.cuda.device_count()):
            devices.append(f"cuda:{k}")
        return {
            "required": {
                "t5v11_name": (get_filename_list_with_downloadable(FOLDER_NAME, KNOWN_T5_MODELS),),
                "t5v11_ver": (["xxl"],),
                "device": (devices, {"default": "auto"}),
                "dtype": (dtypes, {"default": "auto (comfy)"}),
            }
        }

    RETURN_TYPES = ("T5",)
    FUNCTION = "load_model"
    CATEGORY = "ExtraModels/T5"
    TITLE = "T5v1.1 Loader"

    def load_model(self, t5v11_name, t5v11_ver, device, dtype):
        if "bnb" in dtype:
            assert device == "gpu" or device.startswith("cuda"), "BitsAndBytes only works on CUDA! Set device to 'gpu'."
        dtype = string_to_dtype(dtype, "text_encoder")
        if device == "cpu":
            assert dtype in [None, torch.float32], f"Can't use dtype '{dtype}' with CPU! Set dtype to 'default'."

        return (load_t5(
            model_type="t5v11",
            model_ver=t5v11_ver,
            model_path=get_or_download("t5", t5v11_name, KNOWN_T5_MODELS),
            device=device,
            dtype=dtype,
        ),)


class T5TextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "T5": ("T5",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "ExtraModels/T5"
    TITLE = "T5 Text Encode"

    def encode(self, text, T5=None):
        tokens = T5.tokenize(text)
        cond = T5.encode_from_tokens(tokens)
        return ([[cond, {}]],)


NODE_CLASS_MAPPINGS = {
    "T5v11Loader": T5v11Loader,
    "T5TextEncode": T5TextEncode,
    # the previous node is deprecated
    "PixArtT5TextEncode": T5TextEncode,
}
