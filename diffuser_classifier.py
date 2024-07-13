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
import torchvision
from transformers import ViTImageProcessor, ViTForImageClassification
import numpy as np



#atk_target = 123
#print("Attack target class:", model.config.id2label[atk_target])

# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# np.random.seed(0)


def get_classifier(model_id='google/vit-base-patch16-224',device="cuda",ftype=torch.bfloat16):
    model = ViTForImageClassification.from_pretrained(model_id, torch_dtype=ftype, cache_dir='./model/', local_files_only=True).to(device)
    return model


def get_model(model_id="OFA-Sys/small-stable-diffusion-v0", device="cuda",ftype=torch.bfloat16):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=ftype, cache_dir='./model/', local_files_only=True)
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


def predict_class(image,classifier_model):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    normalizer = torchvision.transforms.Normalize(mean, std, inplace=False)
    resizer = torchvision.transforms.Resize((224, 224))

    return classifier_model(normalizer(resizer(image)))