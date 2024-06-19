import comfy.diffusers_load
import comfy.sd
from comfy.cmd import folder_paths
from comfy.model_downloader import get_or_download, get_filename_list_with_downloadable, KNOWN_HUGGINGFACE_MODEL_REPOS
from comfy.model_downloader_types import HuggingFile
from comfy.nodes.base_nodes import DiffusersLoader
from .miaobi_tokenizer import MiaoBiTokenizer

KNOWN_MIAOBI_CLIP_MODELS = [
    HuggingFile("ShineChen1024/MiaoBi", "miaobi_beta0.9/text_encoder/model.safetensors",
                save_with_filename="MiaoBi_CLIP.safetensors")
]

KNOWN_MIAOBI_UNET = [
    HuggingFile("ShineChen1024/MiaoBi", "miaobi_beta0.9/unet/diffusion_pytorch_model.safetensors",
                save_with_filename="MiaoBi.safetensors")
]

KNOWN_HUGGINGFACE_MODEL_REPOS.add("ShineChen1024/MiaoBi")


class MiaoBiCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": (get_filename_list_with_downloadable("clip", KNOWN_MIAOBI_CLIP_MODELS),),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_mbclip"
    CATEGORY = "ExtraModels/MiaoBi"
    TITLE = "MiaoBi CLIP Loader"

    def load_mbclip(self, clip_name):
        clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
        clip_path = get_or_download("clip", clip_name, KNOWN_MIAOBI_CLIP_MODELS)
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type
        )
        # override tokenizer
        clip.tokenizer.clip_l = MiaoBiTokenizer()
        return clip,


class MiaoBiDiffusersLoader(DiffusersLoader):
    CATEGORY = "ExtraModels/MiaoBi"
    TITLE = "MiaoBi Checkpoint Loader (Diffusers)"

    def load_checkpoint(self, model_path, output_vae=True, output_clip=True):
        unet, clip, vae = super().load_checkpoint(model_path, output_vae, output_clip)
        # override tokenizer
        clip.tokenizer.clip_l = MiaoBiTokenizer()
        return unet, clip, vae


NODE_CLASS_MAPPINGS = {
    "MiaoBiCLIPLoader": MiaoBiCLIPLoader,
    "MiaoBiDiffusersLoader": MiaoBiDiffusersLoader,
}
