import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor


class EasyStableDiffusionXLImg2ImgPipeline(nn.Module):
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.model_id = model_id
        self.device = device
        self.dtype = torch_dtype

        # Main Components
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=self.dtype
        ).to(self.device)
        self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=self.dtype
        ).to(self.device)
        self.text_encoder_2: CLIPTextModelWithProjection = (
            CLIPTextModelWithProjection.from_pretrained(
                model_id, subfolder="text_encoder_2", torch_dtype=self.dtype
            ).to(self.device)
        )
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer", torch_dtype=self.dtype
        )
        self.tokenizer_2: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer_2", torch_dtype=self.dtype
        )
        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=self.dtype
        ).to(self.device)

        # Sub Components
        self.scheduler: EulerDiscreteScheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler", torch_dtype=self.dtype
        )
        self.image_processor: VaeImageProcessor = VaeImageProcessor(
            do_convert_rgb=True,
        )

    def encode_prompt(self, prompt: str, negative_prompt: str | None = None):
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]

        prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(self.device), output_hidden_states=True
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]

            # This "2" because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds.hidden_states[-2]  # (1, 77, 768)
            prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)  # (1, 77, 1536)

        if negative_prompt is None:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        else:
            negative_prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(self.device), output_hidden_states=True
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)
            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)
        negative_prompt_embeds = negative_prompt_embeds.to(
            dtype=self.dtype, device=self.device
        )

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def prepare_latents(self, image, timestep, generator):
        # make sure the VAE is in float32 mode, as it overflows in float16
        image = image.to(device=self.device, dtype=torch.float32)
        self.vae.to(dtype=torch.float32)

        init_latents = self.vae.encode(image).latent_dist.sample(generator=generator)

        # Back to the correct device
        self.vae.to(self.device)

        init_latents = init_latents.to(dtype=self.dtype)
        init_latents = self.vae.config["scaling_factor"] * init_latents

        noise = torch.randn(
            init_latents.shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        latents = self.scheduler.add_noise(init_latents, noise, timestep)
        return latents

    def get_add_time_ids(self, latents):
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        height, width = latents.shape[-2:]
        height *= vae_scale_factor
        width *= vae_scale_factor
        original_size = target_size = (height, width)

        add_time_ids = list(original_size + (0, 0) + target_size)
        add_neg_time_ids = list(original_size + (0, 0) + target_size)

        add_time_ids = torch.tensor([add_time_ids], dtype=self.dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=self.dtype)

        return add_time_ids, add_neg_time_ids

    def __call__(
        self,
        prompt: str,
        image: Image.Image,
        strength: float = 0.3,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: str | None = None,
        seed: int = 1117,
    ) -> Image.Image:
        # 1. Encode the text prompt
        (
            prompt_embeds,  # (1, 77, 2048)
            negative_prompt_embeds,  # (1, 77, 2048)
            pooled_prompt_embeds,  # (1, 1280)
            negative_pooled_prompt_embeds,  # (1, 1280)
        ) = self.encode_prompt(prompt, negative_prompt)

        # 2. Preprocess image
        image = self.image_processor.preprocess(image)

        # 3. Define timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        latent_timestep = timesteps[:1]

        # 4. Prepare latents for U-Net
        generator = torch.Generator(self.device).manual_seed(seed)
        latents = self.prepare_latents(
            image=image, timestep=latent_timestep, generator=generator
        )

        add_text_embeds = pooled_prompt_embeds
        add_time_ids, add_neg_time_ids = self.get_add_time_ids(latents)

        # 5. Concat for classifier-free guidance
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, add_text_embeds], dim=0
        )
        add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = add_time_ids.to(self.device)

        # 6. Denoising loop
        for i, t in enumerate(tqdm(timesteps)):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            }
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # make sure the VAE is in float32 mode, as it overflows in float16
        self.vae.to(dtype=torch.float32)
        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        self.vae.to(dtype=self.dtype)

        image = self.image_processor.postprocess(image)[0]
        return image
