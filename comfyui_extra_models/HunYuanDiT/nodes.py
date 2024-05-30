from copy import deepcopy

from comfy.model_downloader import get_or_download, get_filename_list_with_downloadable, add_known_models, \
    KNOWN_CHECKPOINTS, KNOWN_CLIP_MODELS
from comfy.model_downloader_types import HuggingFile
from .conf import hydit_conf
from .loader import load_hydit

KNOWN_HUNYUANDIT_CLIP_MODELS = [
    HuggingFile("Tencent-Hunyuan/HunyuanDiT",
                "t2i/clip_text_encoder/pytorch_model.bin",
                save_with_filename="chinese-roberta-wwm-ext-large.bin")
]

KNOWN_HUNYUANDIT_T5_MODELS = [
    HuggingFile("Tencent-Hunyuan/HunyuanDiT",
                "t2i/mt5/pytorch_model.bin",
                save_with_filename="mT5-xl.bin")
]

KNOWN_HUNYUANDIT_CHECKPOINTS = [
    HuggingFile("Tencent-Hunyuan/HunyuanDiT", "t2i/model/pytorch_model_module.pt", save_with_filename="HunYuanDiT.pt")
]

add_known_models("checkpoints", KNOWN_CHECKPOINTS, *KNOWN_HUNYUANDIT_CHECKPOINTS)
add_known_models("clip", KNOWN_CLIP_MODELS, *KNOWN_HUNYUANDIT_CHECKPOINTS)


class HYDiTCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (get_filename_list_with_downloadable("checkpoints", KNOWN_HUNYUANDIT_CHECKPOINTS),),
                "model": (list(hydit_conf.keys()), {"default": "G/2"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "ExtraModels/HunyuanDiT"
    TITLE = "Hunyuan DiT Checkpoint Loader"

    def load_checkpoint(self, ckpt_name, model):
        ckpt_path = get_or_download("checkpoints", ckpt_name, KNOWN_HUNYUANDIT_CHECKPOINTS)
        model_conf = hydit_conf[model]
        model = load_hydit(
            model_path=ckpt_path,
            model_conf=model_conf,
        )
        return (model,)


#### temp stuff for the text encoder ####
import torch
from .tenc import load_clip, load_t5
from ..utils.dtype import string_to_dtype

dtypes = [
    "default",
    "auto (comfy)",
    "FP32",
    "FP16",
    "BF16"
]


class HYDiTTextEncoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        devices = ["auto", "cpu", "gpu"]
        # hack for using second GPU as offload
        for k in range(1, torch.cuda.device_count()):
            devices.append(f"cuda:{k}")
        return {
            "required": {
                "clip_name": (get_filename_list_with_downloadable("clip", KNOWN_HUNYUANDIT_CLIP_MODELS),),
                "mt5_name": (get_filename_list_with_downloadable("t5", KNOWN_HUNYUANDIT_T5_MODELS),),
                "device": (devices, {"default": "auto"}),
                "dtype": (dtypes, {"default": "auto (comfy)"}),
            }
        }

    RETURN_TYPES = ("CLIP", "T5")
    FUNCTION = "load_model"
    CATEGORY = "ExtraModels/HunyuanDiT"
    TITLE = "Hunyuan DiT Text Encoder Loader"

    def load_model(self, clip_name, mt5_name, device, dtype):
        dtype = string_to_dtype(dtype, "text_encoder")
        if device == "cpu":
            assert dtype in [None, torch.float32,
                             torch.bfloat16], f"Can't use dtype '{dtype}' with CPU! Set dtype to 'default' or 'bf16'."

        clip = load_clip(
            model_path=get_or_download("clip", clip_name, KNOWN_HUNYUANDIT_CLIP_MODELS),
            device=device,
            dtype=dtype,
        )
        t5 = load_t5(
            model_path=get_or_download("t5", mt5_name, KNOWN_HUNYUANDIT_T5_MODELS),
            device=device,
            dtype=dtype,
        )
        return clip, t5


class HYDiTTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "text_t5": ("STRING", {"multiline": True}),
                "CLIP": ("CLIP",),
                "T5": ("T5",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "ExtraModels/HunyuanDiT"
    TITLE = "Hunyuan DiT Text Encode"

    def encode(self, text, text_t5, CLIP, T5):
        if text_t5 is None or text_t5 == "":
            text_t5 = text
        # T5
        T5.load_model()
        t5_pre = T5.tokenizer(
            text_t5,
            text,
            max_length=T5.cond_stage_model.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        t5_mask = t5_pre["attention_mask"]
        with torch.no_grad():
            t5_outs = T5.cond_stage_model.transformer(
                input_ids=t5_pre["input_ids"].to(T5.load_device),
                attention_mask=t5_mask.to(T5.load_device),
                output_hidden_states=True,
            )
            # to-do: replace -1 for clip skip
            t5_embs = t5_outs["hidden_states"][-1].float().cpu()

        # "clip"
        CLIP.load_model()
        clip_pre = CLIP.tokenizer(
            text,
            max_length=CLIP.cond_stage_model.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        clip_mask = clip_pre["attention_mask"]
        with torch.no_grad():
            clip_outs = CLIP.cond_stage_model.transformer(
                input_ids=clip_pre["input_ids"].to(CLIP.load_device),
                attention_mask=clip_mask.to(CLIP.load_device),
            )
            # to-do: add hidden states
            clip_embs = clip_outs[0].float().cpu()

        # combined cond
        return ([[
            clip_embs, {
                "context_t5": t5_embs,
                "context_mask": clip_mask.float(),
                "context_t5_mask": t5_mask.float()
            }
        ]],)


class HYDiTTextEncodeSimple(HYDiTTextEncode):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "CLIP": ("CLIP",),
                "T5": ("T5",),
            }
        }

    FUNCTION = "encode_simple"
    TITLE = "Hunyuan DiT Text Encode (simple)"

    def encode_simple(self, text, **args):
        return self.encode(text=text, text_t5=text, **args)


class HYDiTSrcSizeCond:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond": ("CONDITIONING",),
                "width": ("INT", {"default": 1024.0, "min": 0, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 1024.0, "min": 0, "max": 8192, "step": 16}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("cond",)
    FUNCTION = "add_cond"
    CATEGORY = "ExtraModels/HunyuanDiT"
    TITLE = "Hunyuan DiT Size Conditioning (advanced)"

    def add_cond(self, cond, width, height):
        cond = deepcopy(cond)
        for c in range(len(cond)):
            cond[c][1].update({
                "src_size_cond": [[height, width]],
            })
        return (cond,)


NODE_CLASS_MAPPINGS = {
    "HYDiTCheckpointLoader": HYDiTCheckpointLoader,
    "HYDiTTextEncoderLoader": HYDiTTextEncoderLoader,
    "HYDiTTextEncode": HYDiTTextEncode,
    "HYDiTTextEncodeSimple": HYDiTTextEncodeSimple,
    "HYDiTSrcSizeCond": HYDiTSrcSizeCond,
}
