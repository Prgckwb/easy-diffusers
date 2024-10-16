# easy-diffusers
This is a handy repo to help you get the hang of the Huggingface Diffusers library.

## What’s this about?
If you’ve ever wanted to train a Diffusion Model using Diffusers, you might have noticed that the code can get pretty complex because it needs to handle various community implementations. This repo simplifies things by pulling out the key parts without messing up the original diffusers code.

We've defined an `EasyPipeline` for each matching diffuser's pipeline, like so:
![Code Comparison](assets/compare.png)

Here's a table that shows the difference in code size:

| Class Name                    | File Path                                                                 | Lines |
|------------------------------|--------------------------------------------------------------------------|-------|
| StableDiffusionPipeline      | `diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py`      | 1062  |
| EasyStableDiffusionPipeline  | `easy_diffusers/stable_diffusion/txt2img_stable_diffusion.py`            | 133   |

## How to Use
### 1. Learn the Basics of the Diffusers Pipeline
By checking out the simplified pipeline processes, you can learn about the critical parts without the clutter of the original implementation. We aim to show the shapes of important variables in the code as much as possible.

For instance, in the StableDiffusionPipeline's image generation process, you might see code like this:
```python
# Denoising loop
for i, t in enumerate(tqdm(timesteps)):
    latent_model_input = torch.cat([latents] * 2)  # (2, 4, 64, 64)
    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)  # (2, 4, 64, 64)

    # Predicting the noise residual
    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embed)[0]  # (2, 4, 64, 64)

    # Unconditional and conditional noise predictions
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred = guidance_scale * noise_pred_cond + (1 - guidance_scale) * noise_pred_uncond

    # Update latent variables x_t -> x_t-1
    latents = self.scheduler.step(noise_pred, t, latents)[0]

# Decode latent variables to image
image = self.vae.decode(latents // 0.18215, return_dict=False, generator=generator)[0]  # (1, 3, 512, 512)
image = self.image_processor.postprocess(image, output_type='pil')

return image
```

By cutting down on branching with lots of options, it’s easier to see what’s going on.

### 2. Use the Simplified Code as a Reference
You can use the simplified pipeline just like the original one.
```python
from easy_diffusers import EasyStableDiffusionPipeline

model = EasyStableDiffusionPipeline()
prompt = 'a photo of a dog'
image = model(prompt)
image.save('output/sample.png')
```

## Citation
If you use this repo, please cite the original repository like this:
```bibtex
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```
