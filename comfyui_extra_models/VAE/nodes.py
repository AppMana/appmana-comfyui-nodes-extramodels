from comfy.model_downloader import get_or_download, get_filename_list_with_downloadable
from comfy.model_downloader_types import HuggingFile
from .conf import vae_conf
from .loader import EXVAE
from ..utils.dtype import string_to_dtype

KNOWN_VAES = [
    HuggingFile("mrsteyk/consistency-decoder-sd15", "stk_consistency_decoder_amalgamated.safetensors")
]

# todo: incorporate models from https://github.com/CompVis/latent-diffusion/tree/main#pretrained-autoencoding-models

dtypes = [
    "auto",
    "FP32",
    "FP16",
    "BF16"
]


class ExtraVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (get_filename_list_with_downloadable("vae", KNOWN_VAES),),
                "vae_type": (list(vae_conf.keys()), {"default": "kl-f8"}),
                "dtype": (dtypes,),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "ExtraModels"
    TITLE = "ExtraVAELoader"

    def load_vae(self, vae_name, vae_type, dtype):
        model_path = get_or_download("vae", vae_name, KNOWN_VAES)
        model_conf = vae_conf[vae_type]
        vae = EXVAE(model_path, model_conf, string_to_dtype(dtype, "vae"))
        return (vae,)


NODE_CLASS_MAPPINGS = {
    "ExtraVAELoader": ExtraVAELoader,
}
