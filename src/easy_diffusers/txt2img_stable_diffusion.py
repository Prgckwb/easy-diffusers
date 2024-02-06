import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, PNDMScheduler
from diffusers.image_processor import VaeImageProcessor


class SimpleStableDiffusionPipeline(nn.Module):
    def __init__(
            self,
            model_id: str = 'CompVis/stable-diffusion-v1-4',
            device: str | torch.device = 'cuda',
            torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.model_id = model_id
        self.device = device
        self.dtype = torch_dtype

        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype
        ).to(self.device)

        # Main Components
        self.text_encoder: CLIPTextModel = pipeline.text_encoder
        self.tokenizer: CLIPTokenizer = pipeline.tokenizer
        self.vae: AutoencoderKL = pipeline.vae
        self.unet: UNet2DConditionModel = pipeline.unet

        # Sub Components
        self.scheduler: PNDMScheduler = pipeline.scheduler
        self.image_processor: VaeImageProcessor = pipeline.image_processor

    @torch.inference_mode()
    def __call__(
            self,
            prompt: str,
            height: int = 512,
            width: int = 512,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: str | None = None,
            seed: int = 1117,
    ) -> Image.Image:

        text_inputs = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        )
        text_inputs_ids = text_inputs.input_ids  # (1, 77)

        prompt_embed = self.text_encoder(text_inputs_ids.to(self.device))[0]  # (1, 77, 768)

        if negative_prompt is None:
            uncond_tokens = ['']
        else:
            uncond_tokens = [negative_prompt]

        max_length = prompt_embed.shape[1]  # 77
        uncond_inputs = self.tokenizer(
            uncond_tokens,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_tensors='pt'
        )

        negative_prompt_embed = self.text_encoder(uncond_inputs.input_ids.to(self.device))[0]  # (1, 77, 768)

        # Text embeddings from prompt using CLIP TextEncoder
        prompt_embed = torch.cat([negative_prompt_embed, prompt_embed])  # (2, 77, 768)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Fix random seed
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate random latents
        latents = torch.randn(
            size=(1, 4, width // 8, height // 8),
            device=self.device,
            dtype=self.dtype,
            generator=generator
        )

        # Denoising loop
        for i, t in enumerate(tqdm(timesteps)):
            latent_model_input = torch.cat([latents] * 2)  # (2, 4, 64, 64)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)  # (2, 4, 64, 64)

            # Predict residual noise
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embed)[0]  # (2, 4, 64, 64)

            # Unconditional and conditional noise predictions
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = guidance_scale * noise_pred_cond + (1 - guidance_scale) * noise_pred_uncond

            # Update latents x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)[0]

        # Decode latents to image
        image = self.vae.decode(latents // 0.18215, return_dict=False, generator=generator)[0]  # (1, 3, 512, 512)
        image = self.image_processor.postprocess(image, output_type='pil')[0]

        return image
