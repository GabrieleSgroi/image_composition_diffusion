from typing import Dict

from diffusers import StableDiffusionUpscalePipeline, AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer


def upscale_image(image: Image, prompt: str, model_id="stabilityai/stable-diffusion-x4-upscaler",
                  enable_model_cpu_offload: bool = True) -> Image:
    """Upscale the image using the stable diffusion upscale pipeline.

    Args:
        image: the image to be upscaled.
        prompt: a text prompt describing the image.
        model_id: path on the Hugging Face hub or local path to the pretrained upscale stable diffusion model.
        enable_model_cpu_offload: allows the model to keep the models on GPU only when used, slows down generation but
                                  reduces needed GPU memory.
    """
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id, revision="fp16", torch_dtype=torch.float16
    )
    pipeline = pipeline.to('cuda')
    if enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    upscaled_image = pipeline(prompt=prompt, image=image).images[0]
    return upscaled_image


def load_models(model_path="stabilityai/stable-diffusion-2-base") -> Dict:
    """Downlaod the models from Hugging Face or load them from local repo."""
    text_encoder = CLIPTextModel.from_pretrained(model_path,
                                                 subfolder="text_encoder",
                                                 torch_dtype=torch.float16,
                                                 revision="fp16")
    tokenizer = CLIPTokenizer.from_pretrained(model_path,
                                              subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(model_path,
                                        torch_dtype=torch.float16,
                                        revision="fp16",
                                        subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_path,
                                                torch_dtype=torch.float16,
                                                revision="fp16",
                                                subfolder="unet")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path,
                                                            subfolder="scheduler")
    return {'vae': vae, 'unet': unet, 'scheduler': scheduler, 'text_encoder': text_encoder, 'tokenizer': tokenizer}
