import os
os.environ["HF_HOME"] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"
os.environ['MPLCONFIGDIR'] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"

import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from tqdm.auto import tqdm

import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor

from accelerate import Accelerator


def get_model(model_id, device):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, cache_dir='./model/')
    # pipe.to(device)
    vae = pipe.vae
    unet = pipe.unet
    image_processor = pipe.image_processor
    scheduler = pipe.scheduler
    return vae, unet, image_processor, scheduler

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def forward(timesteps, num_inference_steps, scheduler, unet, vae, prompt_embeds, image_processor, output_type, latents):
    
    for t in tqdm(timesteps):

        latent_model_input = scheduler.scale_model_input(latents, t)

    
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    latents = latents.to('cuda:0')
    if not output_type == "latent":
        image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[
            0
        ]
    else:
        image = latents
    do_denormalize = [True] * image.shape[0]
    image = image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    return image


def main():

    accelerator = Accelerator()
    model_id = "CompVis/stable-diffusion-v1-4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, unet, image_processor, scheduler = get_model(model_id, device)
    print(0)
    prompt_embeds = torch.randn((1, 77, 768), device=device, dtype=torch.bfloat16)
    height = 256
    width = 256
    num_inference_steps = 50
    timesteps = None

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    batch_size = 1
    num_images_per_prompt = 1


    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler, num_inference_steps, device, timesteps
    )
    print(1)
    num_channels_latents = unet.config.in_channels

    shape = (
        batch_size,
        num_channels_latents,
        int(height) // vae_scale_factor,
        int(width) // vae_scale_factor,
    )

    latents = randn_tensor(shape, device=device, dtype=prompt_embeds.dtype)
    latents.to(device)
    latents = latents * scheduler.init_noise_sigma


    print(2)
    ty = 'PIL.Image'
    timesteps, num_inference_steps, scheduler, unet, vae, prompt_embeds, image_processor, ty, latents = accelerator.prepare(timesteps, num_inference_steps, scheduler, unet, vae, prompt_embeds, image_processor, ty, latents)


    out = forward(timesteps, num_inference_steps, scheduler, unet, vae, prompt_embeds, image_processor, 'PIL.Image', latents)
    print('done')
    
    



if __name__ == "__main__":
    main()



        # callback = kwargs.pop("callback", None)
        # callback_steps = kwargs.pop("callback_steps", None)

        # if callback is not None:
        #     deprecate(
        #         "callback",
        #         "1.0.0",
        #         "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        #     )
        # if callback_steps is not None:
        #     deprecate(
        #         "callback_steps",
        #         "1.0.0",
        #         "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        #     )

        # if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        #     callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # # 0. Default height and width to unet
        # height = height or self.unet.config.sample_size * self.vae_scale_factor
        # width = width or self.unet.config.sample_size * self.vae_scale_factor
        # # to deal with lora scaling and other possible forward hooks

        # # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt,
        #     height,
        #     width,
        #     callback_steps,
        #     negative_prompt,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     ip_adapter_image,
        #     ip_adapter_image_embeds,
        #     callback_on_step_end_tensor_inputs,
        # )

        # self._guidance_scale = guidance_scale
        # self._guidance_rescale = guidance_rescale
        # self._clip_skip = clip_skip
        # self._cross_attention_kwargs = cross_attention_kwargs
        # self._interrupt = False

        # # 2. Define call parameters
        # if prompt is not None and isinstance(prompt, str):
        #     batch_size = 1
        # elif prompt is not None and isinstance(prompt, list):
        #     batch_size = len(prompt)
        # else:
        #     batch_size = prompt_embeds.shape[0]

        # device = self._execution_device

        # # 3. Encode input prompt
        # lora_scale = (
        #     self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        # )

        # prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        #     prompt,
        #     device,
        #     num_images_per_prompt,
        #     self.do_classifier_free_guidance,
        #     negative_prompt,
        #     prompt_embeds=prompt_embeds,
        #     negative_prompt_embeds=negative_prompt_embeds,
        #     lora_scale=lora_scale,
        #     clip_skip=self.clip_skip,
        # )

        # # For classifier free guidance, we need to do two forward passes.
        # # Here we concatenate the unconditional and text embeddings into a single batch
        # # to avoid doing two forward passes
        # if self.do_classifier_free_guidance:
        #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        #     image_embeds = self.prepare_ip_adapter_image_embeds(
        #         ip_adapter_image,
        #         ip_adapter_image_embeds,
        #         device,
        #         batch_size * num_images_per_prompt,
        #         self.do_classifier_free_guidance,
        #     )

        # # 4. Prepare timesteps
        # timesteps, num_inference_steps = retrieve_timesteps(
        #     self.scheduler, num_inference_steps, device, timesteps, sigmas
        # )

        # # 5. Prepare latent variables
        # num_channels_latents = self.unet.config.in_channels
        # latents = self.prepare_latents(
        #     batch_size * num_images_per_prompt,
        #     num_channels_latents,
        #     height,
        #     width,
        #     prompt_embeds.dtype,
        #     device,
        #     generator,
        #     latents,
        # )

        # # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # # 6.1 Add image embeds for IP-Adapter
        # added_cond_kwargs = (
        #     {"image_embeds": image_embeds}
        #     if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
        #     else None
        # )

        # # 6.2 Optionally get Guidance Scale Embedding
        # timestep_cond = None
        # if self.unet.config.time_cond_proj_dim is not None:
        #     guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        #     timestep_cond = self.get_guidance_scale_embedding(
        #         guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        #     ).to(device=device, dtype=latents.dtype)

        # # 7. Denoising loop
        # num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # self._num_timesteps = len(timesteps)
        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        #     for i, t in enumerate(timesteps):
        #         if self.interrupt:
        #             continue

        #         # expand the latents if we are doing classifier free guidance
        #         latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        #         latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        #         # predict the noise residual
        #         noise_pred = self.unet(
        #             latent_model_input,
        #             t,
        #             encoder_hidden_states=prompt_embeds,
        #             timestep_cond=timestep_cond,
        #             cross_attention_kwargs=self.cross_attention_kwargs,
        #             added_cond_kwargs=added_cond_kwargs,
        #             return_dict=False,
        #         )[0]

        #         # perform guidance
        #         if self.do_classifier_free_guidance:
        #             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #             noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        #         if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
        #             # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
        #             noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

        #         # compute the previous noisy sample x_t -> x_t-1
        #         latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        #         if callback_on_step_end is not None:
        #             callback_kwargs = {}
        #             for k in callback_on_step_end_tensor_inputs:
        #                 callback_kwargs[k] = locals()[k]
        #             callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

        #             latents = callback_outputs.pop("latents", latents)
        #             prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
        #             negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

        #         # call the callback, if provided
        #         if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
        #             progress_bar.update()
        #             if callback is not None and i % callback_steps == 0:
        #                 step_idx = i // getattr(self.scheduler, "order", 1)
        #                 callback(step_idx, t, latents)

        # if not output_type == "latent":
        #     image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
        #         0
        #     ]
        #     image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        # else:
        #     image = latents
        #     has_nsfw_concept = None

        # if has_nsfw_concept is None:
        #     do_denormalize = [True] * image.shape[0]
        # else:
        #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        # image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # # Offload all models
        # self.maybe_free_model_hooks()

        # if not return_dict:
        #     return (image, has_nsfw_concept)

        # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)