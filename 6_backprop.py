import os
os.environ["HF_HOME"] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"
os.environ['MPLCONFIGDIR'] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"

import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor


def get_model(model_id, device):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, cache_dir='./model/', force_download=False)
    pipe.to(device)
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
    
    for t in timesteps:

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
    image = image / 2 + 0.5
    image = image.clamp(0, 1)
    # image = image_processor.postprocess(image, do_denormalize=do_denormalize)

    return image


def main():
    tr = 10
    # model_id = "CompVis/stable-diffusion-v1-4"
    model_id = "OFA-Sys/small-stable-diffusion-v0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, unet, image_processor, scheduler = get_model(model_id, device)
    
    height = 256
    width = 256
    num_inference_steps = 50
    

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    batch_size = 1
    num_images_per_prompt = 1

    num_channels_latents = unet.config.in_channels

    shape = (
        batch_size,
        num_channels_latents,
        int(height) // vae_scale_factor,
        int(width) // vae_scale_factor,
    )

    prompt_embeds = torch.randn((1, 77, 768), device=device, dtype=torch.bfloat16)
    prompt_embeds.requires_grad = True
    latents = randn_tensor(shape, device=device, dtype=prompt_embeds.dtype)
    latents.requires_grad = True
    latents.to(device)

    optimizer = torch.optim.AdamW([latents, prompt_embeds], lr=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    targetImg = torch.load('output.pth').to(device)

    
    
    for i in tqdm(range(10)):
        timesteps = None
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, device, timesteps
        )
        out = forward(timesteps, num_inference_steps, scheduler, unet, vae, prompt_embeds, image_processor, 'PIL.Image', latents)

        im = out.to('cpu')
        plt.imshow(im[0].float().detach().permute(1, 2, 0).numpy())
        plt.savefig(f'ims/out{tr}_{i}.png')
        
        loss = torch.nn.functional.mse_loss(out, targetImg)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tqdm.write(str(loss.item()))


    timesteps = None
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler, num_inference_steps, device, timesteps
    )
    out = forward(timesteps, num_inference_steps, scheduler, unet, vae, prompt_embeds, image_processor, 'PIL.Image', latents)
    
    
    torch.save(out, f'output{tr}.pth')
    
    



if __name__ == "__main__":
    main()
