from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image
from torch import nn

from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionImg2ImgPipeline

from pro_deep import embed_image_with_dino, get_device, get_frozen_model


ImageLike = Union[str, Path, Image.Image]
PromptLike = Union[str, List[str]]


@dataclass
class ImageTextDiffusionConfig:
    """Config for the image+text conditioned diffusion pipeline."""

    model_id: str = "runwayml/stable-diffusion-1-5"
    num_image_tokens: int = 4
    image_cond_scale: float = 1.0
    torch_dtype: Optional[torch.dtype] = None
    scheduler_cls: type = EulerAncestralDiscreteScheduler
    disable_safety_checker: bool = True
    device: Optional[str] = None


class ImageTextDiffusionPipeline:
    """
    Lightweight wrapper around Stable Diffusion img2img that injects a DINO image
    embedding as extra conditioning tokens alongside the text encoder outputs.
    """

    def __init__(self, config: Optional[ImageTextDiffusionConfig] = None):
        self.config = config or ImageTextDiffusionConfig()
        self.device = torch.device(config.device) if config.device else get_device()
        self.dtype = (
            config.torch_dtype
            if config.torch_dtype is not None
            else (torch.float16 if self.device.type in ("cuda", "mps") else torch.float32)
        )

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            config.model_id,
            torch_dtype=self.dtype,
            safety_checker=None if config.disable_safety_checker else None,
            feature_extractor=None if config.disable_safety_checker else None,
        )
        if config.scheduler_cls:
            self.pipe.scheduler = config.scheduler_cls.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)

        self.dino_model, self.dino_processor = get_frozen_model(device=self.device)
        dino_dim = self.dino_model.config.hidden_size
        text_dim = self.pipe.text_encoder.config.hidden_size

        self.num_image_tokens = config.num_image_tokens
        self.image_proj = nn.Sequential(
            nn.LayerNorm(dino_dim),
            nn.Linear(dino_dim, text_dim * config.num_image_tokens),
        ).to(self.device, dtype=self.pipe.unet.dtype)
        self.image_cond_scale = config.image_cond_scale

    @staticmethod
    def _to_pil_image(image: ImageLike) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    def _encode_prompt(
        self,
        prompt: PromptLike,
        negative_prompt: Optional[PromptLike],
        num_images_per_prompt: int,
    ):
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt_list)

        text_inputs = self.pipe.tokenizer(
            prompt_list,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        prompt_embeds = self.pipe.text_encoder(text_input_ids, attention_mask=attention_mask)[0]
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        prompt_embeds = prompt_embeds.to(dtype=self.pipe.unet.dtype)

        neg_prompt = negative_prompt if negative_prompt is not None else ""
        neg_list = [neg_prompt] if isinstance(neg_prompt, str) else neg_prompt
        if len(neg_list) == 1 and batch_size > 1:
            neg_list = neg_list * batch_size

        neg_inputs = self.pipe.tokenizer(
            neg_list,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        neg_input_ids = neg_inputs.input_ids.to(self.device)
        neg_attention_mask = neg_inputs.attention_mask.to(self.device)
        negative_embeds = self.pipe.text_encoder(neg_input_ids, attention_mask=neg_attention_mask)[0]
        negative_embeds = negative_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        negative_embeds = negative_embeds.to(dtype=self.pipe.unet.dtype)

        return prompt_embeds, negative_embeds

    @torch.no_grad()
    def _encode_image_tokens(
        self,
        image: Image.Image,
        batch_size: int,
        num_images_per_prompt: int,
        cond_scale: float,
    ) -> torch.Tensor:
        img_embed = embed_image_with_dino(
            image,
            model=self.dino_model,
            processor=self.dino_processor,
            device=self.device,
            pool=True,
        )
        img_embed = img_embed.to(self.pipe.unet.dtype)
        projected = self.image_proj(img_embed)
        tokens = projected.view(1, self.num_image_tokens, -1)
        tokens = tokens * cond_scale
        tokens = tokens.repeat_interleave(batch_size * num_images_per_prompt, dim=0)
        return tokens

    @torch.no_grad()
    def __call__(
        self,
        image: ImageLike,
        prompt: PromptLike,
        negative_prompt: Optional[PromptLike] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        strength: float = 0.6,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        image_cond_scale: Optional[float] = None,
    ):
        cond_scale = image_cond_scale if image_cond_scale is not None else self.image_cond_scale
        source_image = self._to_pil_image(image)
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt_list)

        prompt_embeds, negative_embeds = self._encode_prompt(
            prompt=prompt_list,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
        )
        img_tokens = self._encode_image_tokens(
            image=source_image,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            cond_scale=cond_scale,
        )

        prompt_embeds = torch.cat([prompt_embeds, img_tokens], dim=1)
        negative_tokens = torch.zeros_like(img_tokens)
        negative_embeds = torch.cat([negative_embeds, negative_tokens], dim=1)

        result = self.pipe(
            prompt=None,
            image=source_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=generator,
            output_type=output_type,
        )
        return result
