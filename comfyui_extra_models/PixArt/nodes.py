from comfy import utils
from comfy.cmd import folder_paths
from comfy.model_downloader import add_known_models, get_or_download, get_filename_list_with_downloadable, \
    KNOWN_CHECKPOINTS
from comfy.model_downloader_types import HuggingFile
from .conf import pixart_conf, pixart_res
from .loader import load_pixart
from .lora import load_pixart_lora

PIXART_CHECKPOINTS = [HuggingFile("PixArt-alpha/PixArt-alpha", "PixArt-XL-2-1024-MS.pth"),
                      HuggingFile("PixArt-alpha/PixArt-Sigma", 'PixArt-Sigma-XL-2-1024-MS.pth'),
                      HuggingFile("PixArt-alpha/PixArt-Sigma", 'PixArt-Sigma-XL-2-2K-MS.pth'),
                      ]

add_known_models("checkpoints", KNOWN_CHECKPOINTS,
                 *PIXART_CHECKPOINTS
                 )


class PixArtCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (get_filename_list_with_downloadable("checkpoints", PIXART_CHECKPOINTS),),
                "model": (list(pixart_conf.keys()),),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "ExtraModels/PixArt"
    TITLE = "PixArt Checkpoint Loader"

    def load_checkpoint(self, ckpt_name, model):
        ckpt_path = get_or_download("checkpoints", ckpt_name, PIXART_CHECKPOINTS)
        model_conf = pixart_conf[model]
        model = load_pixart(
            model_path=ckpt_path,
            model_conf=model_conf,
        )
        return (model,)


class PixArtResolutionSelect():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (list(pixart_res.keys()),),
                # keys are the same for both
                "ratio": (list(pixart_res["PixArtMS_XL_2"].keys()), {"default": "1.00"}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_res"
    CATEGORY = "ExtraModels/PixArt"
    TITLE = "PixArt Resolution Select"

    def get_res(self, model, ratio):
        width, height = pixart_res[model][ratio]
        return (width, height)


class PixArtLoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "ExtraModels/PixArt"
    TITLE = "PixArt Load LoRA"

    def load_lora(self, model, lora_name, strength, ):
        if strength == 0:
            return (model)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora = load_pixart_lora(model, lora, lora_path, strength, )
        return (model_lora,)


class PixArtResolutionCond:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond": ("CONDITIONING",),
                "width": ("INT", {"default": 1024.0, "min": 0, "max": 8192}),
                "height": ("INT", {"default": 1024.0, "min": 0, "max": 8192}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("cond",)
    FUNCTION = "add_cond"
    CATEGORY = "ExtraModels/PixArt"
    TITLE = "PixArt Resolution Conditioning"

    def add_cond(self, cond, width, height):
        for c in range(len(cond)):
            cond[c][1].update({
                "img_hw": [[height, width]],
                "aspect_ratio": [[height / width]],
            })
        return (cond,)


class PixArtControlNetCond:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond": ("CONDITIONING",),
                "latent": ("LATENT",),
                # "image": ("IMAGE",),
                # "vae": ("VAE",),
                # "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("cond",)
    FUNCTION = "add_cond"
    CATEGORY = "ExtraModels/PixArt"
    TITLE = "PixArt ControlNet Conditioning"

    def add_cond(self, cond, latent):
        for c in range(len(cond)):
            cond[c][1]["cn_hint"] = latent["samples"] * 0.18215
        return (cond,)


NODE_CLASS_MAPPINGS = {
    "PixArtCheckpointLoader": PixArtCheckpointLoader,
    "PixArtResolutionSelect": PixArtResolutionSelect,
    "PixArtLoraLoader": PixArtLoraLoader,
    "PixArtResolutionCond": PixArtResolutionCond,
    "PixArtControlNetCond": PixArtControlNetCond,
}
