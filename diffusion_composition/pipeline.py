from typing import Tuple
from diffusers.image_processor import VaeImageProcessor
from diffusion_composition.prompting import BoundingBoxPromptSetter
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DPMSolverMultistepScheduler
import torch
import numpy as np
from tqdm import tqdm


class DiffusionCompositionPipeline:
    """Pipeline implementing MultiDiffusion for image composition. Based on the stable diffusion pipeline of the
       diffusers library by Hugging Face."""

    def __init__(self, vae: AutoencoderKL, unet: UNet2DConditionModel, scheduler: DPMSolverMultistepScheduler,
                 img_size: Tuple[int, int] = (512, 512)):
        """
        Args:
            vae: pre-trained variational autoencoder of the latent diffusion model.
            unet: pre-trained unet predicting noise.
            scheduler: diffusers scheduler for image generation.
            img_size: size of the image to generate.
        """
        self.vae = vae
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.latent_size = (int(img_size[0] / self.vae_scale_factor), int(img_size[1] / self.vae_scale_factor))
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.unet = unet
        self.scheduler = scheduler
        self.img_size = img_size
        self.random_colors = [np.random.randint(low=0, high=255, size=3) for i in range(10)]
        self.constant_latents = torch.cat([self.get_constant_latents(color).unsqueeze(0) for color in
                                           self.random_colors])

    def get_constant_latents(self, color: np.ndarray | None = 0, device: str = 'cuda') -> torch.Tensor:
        """Get the latents corresponding to a uniform color."""
        self.vae.to(device)
        with torch.no_grad():
            constant_color_img = torch.Tensor(np.ones(shape=(3, self.img_size[0], self.img_size[1])
                                                      ) * color.reshape(-1, 1, 1)).half()
            constant_color_img = self.image_processor.preprocess(constant_color_img)
            constant_color_latents = self.vae.encode(constant_color_img.to(device))

        constant_color_latents = constant_color_latents.latent_dist.sample()
        constant_color_latents = constant_color_latents * self.vae.config.scaling_factor

        # Offload vae to CPU to save GPU memory
        self.vae.to('cpu')

        return constant_color_latents

    def batched_prediction(self, latents: torch.Tensor, prompt_embeds: torch.Tensor, t: torch.Tensor,
                           guidance: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Predict the denoised latents after one diffusion step in batches."""
        n_batches = int(prompt_embeds.shape[0] / batch_size)
        residual = prompt_embeds.shape[0] % batch_size
        noise = []
        for i in range(n_batches):
            batched_latents = latents[i * batch_size:(i + 1) * batch_size, ...]
            batched_prompts = prompt_embeds[i * batch_size: (i + 1) * batch_size, ...]
            batched_latents = batched_latents.view([batch_size * 2] + list(latents.shape)[2:])
            batched_prompts = batched_prompts.view([batch_size * 2] + list(prompt_embeds.shape)[2:])
            noise_preds = self.unet(batched_latents, t, encoder_hidden_states=batched_prompts)[0]
            noise.append(noise_preds)

        if residual > 0:
            batched_latents = latents[-residual:, ...]
            batched_prompts = prompt_embeds[-residual:, ...]
            batched_latents = batched_latents.view([residual * 2] + list(latents.shape)[2:])
            batched_prompts = batched_prompts.view([residual * 2] + list(prompt_embeds.shape)[2:])
            noise_preds = self.unet(batched_latents, t, encoder_hidden_states=batched_prompts)[0]
            noise.append(noise_preds)

        noise = torch.cat(noise)
        noise = noise.view(latents.shape)
        noise = noise[:, 0, ...] + guidance.reshape(latents.shape[0], 1, 1, 1) * (noise[:, 0, ...] - noise[:, 1, ...])
        denoised = (self.scheduler.step(noise, t, latents[:, 0, ...], return_dict=False)[0])

        return denoised

    def denoising_step(self, latents: torch.Tensor, t: torch.Tensor, prompt_embeds: torch.Tensor,
                       bbox_masks: torch.Tensor, guidance_scale: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Denoising step. Denoised latents are predicted for each prompt and then an average weighted by the
           bounding boxes is returned"""
        # Double the latents for classifier free guidance
        latents = torch.cat([latents.unsqueeze(1)] * 2, dim=1)
        latents = self.scheduler.scale_model_input(latents, t)

        #  Get denoised latents for each prompt and mask

        new_latents = self.batched_prediction(latents=latents, prompt_embeds=prompt_embeds, t=t,
                                              guidance=guidance_scale, batch_size=batch_size)
        new_latents = new_latents * bbox_masks

        # Combine latents according to masks
        combined_latents = new_latents.sum(dim=0) / (bbox_masks.sum(dim=0))
        # Clone latents, 1 per prompt
        new_latents = torch.cat([combined_latents.unsqueeze(0)] * len(prompt_embeds))

        return new_latents

    def bootstrap_step(self, latents: torch.Tensor, t: torch.Tensor, prompt_embeds: torch.Tensor,
                       bbox_masks: torch.Tensor, guidance_scale: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Bootstrap step. For each prompt the corresponding latents are cropped using the bounding boxes and the
           latents corresponding to a random constant color are set on the background."""
        # Add noise to constant latents
        noise = torch.randn_like(latents)
        random_constant_latent = self.constant_latents[np.random.randint(0, len(self.constant_latents))]
        noised_constant_latents = self.scheduler.add_noise(random_constant_latent, noise, t)
        # Combine latents with noised constant color latents according to the masks
        bootstrapped_input = latents * bbox_masks + noised_constant_latents * torch.logical_not(bbox_masks)
        bootstrapped_input = self.scheduler.scale_model_input(bootstrapped_input, t)
        # Double the latents for classifier free guidance
        bootstrapped_input = torch.cat([bootstrapped_input.unsqueeze(1)] * 2, dim=1)
        new_latents = self.batched_prediction(latents=bootstrapped_input,
                                              prompt_embeds=prompt_embeds,
                                              guidance=guidance_scale,
                                              batch_size=batch_size,
                                              t=t, )
        return new_latents

    def decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        """Decode the latents using the decoder."""
        latents = 1 / self.vae.config.scaling_factor * latents
        with torch.no_grad():
            image = self.vae.decode(latents.unsqueeze(0), return_dict=False)[0].squeeze()
        images = (image / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(1, 2, 0).float().numpy()
        return images

    def __call__(self, prompt_setter: BoundingBoxPromptSetter, num_inference_steps: int = 50,
                 bootstrap_steps: int = 10, batch_size: int = 1, device: str = 'cuda') -> np.ndarray:
        """Generate the image for the configured local prompts.

        Args:
            prompt_setter:  BoundingBoxPromptSetter instance specifying the prompts position.
            num_inference_steps: total number of steps of the reverse diffusion process.
            bootstrap_steps: number of bootstrap steps at the beginning of the diffusion process.
            batch_size: batch size into which group different prompts.
            device: device used for generation.
        """
        with torch.no_grad():
            self.unet.to(device)
            self.vae.to(device)
            self.unet.eval()
            self.vae.eval()

            # Get prompts, bboxes and guidance scales

            prompts, bbox_masks, guidance_scales = prompt_setter.prepare_prompts_tensors(device=device)

            # Create latents
            latents = torch.randn(len(prompts),
                                  self.unet.config.in_channels,
                                  self.latent_size[0],
                                  self.latent_size[1]).half().to(device)

            # Get prompts data

            self.scheduler.set_timesteps(num_inference_steps, device=device)

            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                if i < bootstrap_steps:
                    latents = self.bootstrap_step(latents=latents.half(),
                                                  t=t,
                                                  prompt_embeds=prompts,
                                                  bbox_masks=bbox_masks,
                                                  guidance_scale=guidance_scales,
                                                  batch_size=batch_size,
                                                  )
                else:
                    latents = self.denoising_step(latents=latents.half(),
                                                  t=t,
                                                  prompt_embeds=prompts,
                                                  bbox_masks=bbox_masks,
                                                  guidance_scale=guidance_scales,
                                                  batch_size=batch_size,
                                                  )
        return self.decode_latents(latents[0, ...].half())
