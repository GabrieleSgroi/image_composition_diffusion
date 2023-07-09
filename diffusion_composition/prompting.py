from typing import Tuple, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from transformers import CLIPTextModel, CLIPTokenizer


class BoundingBoxPromptSetter:
    """Class to configure the text prompts with their bounding boxes."""

    def __init__(self, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer,
                 img_size: Tuple = (512, 512), latent_size: Tuple = (64, 64)):
        """

        Args:
            text_encoder: pre-trained text encoder used by the diffusion model for text conditioning.
            tokenizer: tokenizer for the text encoder.
            img_size: image size to generate.
            latent_size: latent dimension of the latent diffusion model.
        """
        self.img_size = img_size
        self.latent_size = latent_size
        self.background_prompt = None
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.local_prompts = {"prompt": [], "bbox": [], "guidance_scale": [], 'margins': []}
        self.prompts_addition = None

    def check_bbox(self, **margins) -> None:
        """Check that margins of the bounding box are valid."""
        for side, val in margins.items():
            if not 0. <= val < 1.:
                raise ValueError(f"All margins must be between 0 and 1. Found {side} margin with value {val}")
            if (margins["top"] + margins["bottom"]) >= 1.:
                raise ValueError("Sum of top and bottom margin must be smaller than 1.")
            if (margins["left"] + margins["right"]) >= 1.:
                raise ValueError("Sum of left and right margin must be smaller than 1.")

    def set_background_prompt(self, prompt: str, guidance_scale: float = 7.5) -> None:
        """Add a global general prompt on the background"""
        self.background_prompt = {"prompt": prompt, "guidance_scale": guidance_scale}

    def add_local_prompt(self, prompt: str, top_margin: float, bottom_margin: float, left_margin: float,
                         right_margin: float, guidance_scale: float = 7.5) -> None:
        """Add a local prompt limited delimited by the bounding box margins. The level of the prompts in the image
           (background/foreground) in case of intersections depends on the order in which they are added.
            First set prompts are put on the foreground."""
        self.check_bbox(top=top_margin, bottom=bottom_margin, left=left_margin, right=right_margin)
        self.local_prompts["prompt"].append(prompt)
        self.local_prompts["bbox"].append(self.create_bbox_mask(top=top_margin,
                                                                bottom=bottom_margin,
                                                                left=left_margin,
                                                                right=right_margin,
                                                                ))
        self.local_prompts["margins"].append({"top": top_margin,
                                              "bottom": bottom_margin,
                                              "left": left_margin,
                                              "right": right_margin})

        self.local_prompts["guidance_scale"].append(guidance_scale)

    def add_to_all_prompts(self, text: str) -> None:
        """Add text at the end of all prompts when encoding the text."""
        self.background_prompt["prompt"] += " " + text
        for i in range(len(self.local_prompts["prompt"])):
            self.local_prompts["prompt"][i] += " " + text

    def create_bbox_mask(self, top: float, bottom: float, left: float, right: float) -> np.ndarray:
        """Create a bounding box mask with value 1 in the specified rectangle and 0 outside it."""
        mask = np.zeros((1, self.latent_size[0], self.latent_size[1]), dtype=np.float32)
        margin_top = int(top * self.latent_size[1])
        margin_bottom = self.latent_size[1] - int(bottom * self.latent_size[1])
        margin_left = int(self.latent_size[0] * left)
        margin_right = self.latent_size[0] - int(right * self.latent_size[0])
        mask[:, margin_top:margin_bottom, margin_left:margin_right] = 1.
        return mask

    def level_bbox_mask(self, bbox_masks: List[np.ndarray]) -> List[np.ndarray]:
        """Set the level for prompts in the intersections. First set prompts are put on the foreground."""
        complement = np.ones((self.latent_size[0], self.latent_size[1]))
        leveled_mask = []
        for mask in bbox_masks:
            leveled_mask.append(mask * complement)
            new_complement = np.logical_not(mask) * 1
            complement = complement * new_complement
        return leveled_mask

    def draw_bboxes(self, save_path: str | None = None, fig_size: Tuple = (5, 5), colors: List | None = None) -> None:
        """Save a figure specifying the bounding boxes for each prompt."""
        if colors is None:
            colors = ['#F30C0D', '#80F30C', '#0CF3F2', '#7F0CF3', '#06F950', '#0636F9', '#F906AF', '#F9C906']
        fig, ax = plt.subplots(figsize=fig_size)
        min_dim = np.argmin(self.img_size)
        if min_dim == 0:
            v_size = self.img_size[0] / self.img_size[1]
            h_size = 1
        else:
            v_size = 1
            h_size = self.img_size[1] / self.img_size[0]
        ax.add_patch(Rectangle((0, 0), h_size, v_size,
                               label=self.background_prompt["prompt"]))
        for i, margins in enumerate(reversed(self.local_prompts['margins'])):
            ax.add_patch(Rectangle((margins["left"] * h_size, margins["bottom"] * v_size),
                                   (h_size - margins["right"] * h_size - margins["left"] * h_size),
                                   (v_size - margins["top"] * v_size - margins["bottom"] * v_size),
                                   label=list(reversed(self.local_prompts["prompt"]))[i],
                                   color=colors[i]))
        fig.legend()
        ax.set_axis_off()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        return fig

    def encode_prompts(self, prompts, device: str = 'cuda') -> torch.Tensor:
        """Encode the prompts using the text encoder."""
        self.text_encoder.to(device)
        tokenized_prompts = self.tokenizer(prompts,
                                           padding="max_length",
                                           max_length=self.tokenizer.model_max_length,
                                           truncation=True,
                                           return_tensors="pt",
                                           )
        prompts_ids = tokenized_prompts.input_ids
        prompt_embeds = self.text_encoder(prompts_ids.to(device))
        prompt_embeds = prompt_embeds.last_hidden_state
        uncond_prompts = [""] * len(prompts)
        uncond_tokenized = self.tokenizer(uncond_prompts,
                                          padding="max_length",
                                          max_length=self.tokenizer.model_max_length,
                                          truncation=True,
                                          return_tensors="pt",
                                          )
        uncond_ids = uncond_tokenized.input_ids
        uncond_embeds = self.text_encoder(uncond_ids.to(device)).last_hidden_state

        prompt_embeds = prompt_embeds.unsqueeze(1)
        uncond_embeds = uncond_embeds.unsqueeze(1)
        return torch.cat([prompt_embeds, uncond_embeds], dim=1)

    def prepare_prompts_tensors(self, device: str = 'cuda') -> Tuple[torch.Tensor]:
        """Return prompt, mask and guidance scale tensors."""

        if self.background_prompt is None:
            raise ValueError("No background prompt set. Please add a prompt for the background using the"
                             " set_background_prompt method")
        # Encode local_prompts and get bbox masks
        if len(self.local_prompts["bbox"]) > 0:
            global_fill_mask = np.logical_not(np.sum(self.local_prompts["bbox"], axis=0)) * 1.
        else:
            global_fill_mask = np.ones((1, self.latent_size[0], self.latent_size[1]))
        prompts = self.encode_prompts(self.local_prompts["prompt"] + [self.background_prompt["prompt"]], device=device)
        bbox_masks = self.level_bbox_mask(self.local_prompts["bbox"] + [global_fill_mask])
        bbox_masks = torch.FloatTensor(np.concatenate(bbox_masks, axis=0)).half().unsqueeze(1).to(device)
        guidance_scales = torch.Tensor(self.local_prompts["guidance_scale"] + [self.background_prompt["guidance_scale"]]
                                       ).to(device)
        # Offload the text encoder to CPU to save GPU memory
        self.text_encoder.to('cpu')
        return prompts, bbox_masks, guidance_scales
