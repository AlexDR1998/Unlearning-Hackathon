import os
os.environ["HF_HOME"] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"
os.environ['MPLCONFIGDIR'] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"

import torch

import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from diffusers import StableDiffusionPipeline

from diffusers.utils.torch_utils import randn_tensor


from diffuser_classifier import get_classifier,predict_class
from diffuser_with_grad import *


def get_model_full(model_id="OFA-Sys/small-stable-diffusion-v0", device="cuda",ftype=torch.bfloat16):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=ftype, cache_dir='./model/', local_files_only=True)
    pipe.to(device)
    vae = pipe.vae
    unet = pipe.unet
    image_processor = pipe.image_processor
    scheduler = pipe.scheduler
    return vae, unet, image_processor, scheduler, pipe


def main():
    tr = 30
    atk_target = 293
    # model_id = "CompVis/stable-diffusion-v1-4"
    model_id = "OFA-Sys/small-stable-diffusion-v0"
    classifier_id = 'google/vit-base-patch16-224'
    device = 'cuda'
    batch_size = 1
    num_inference_steps = 1
    vae, unet, image_processor, scheduler, pipe = get_model_full(model_id, device)


    classifier_model = get_classifier(classifier_id)
    print("Attack target class:", classifier_model.config.id2label[atk_target])


    newPipe = OurPipe(pipe.vae, pipe.text_encoder, pipe.tokenizer, pipe.unet, pipe.scheduler, pipe.safety_checker, pipe.feature_extractor, pipe.image_encoder, requires_safety_checker=False)

    
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    height = 512//2
    width = 512//2
    shape = (1, unet.config.in_channels, int(height) // vae_scale_factor, int(width) // vae_scale_factor)
    
    
    latents = randn_tensor(shape, generator=None, device=device, dtype=torch.bfloat16)
    prompt_embeds_org = pipe.encode_prompt('an apple, 4k', device, 1, False)[0].detach()

    prompt_embeds_org.requires_grad = True

    optimizer = torch.optim.Adam([prompt_embeds_org], lr=0.1)
    
    for i in tqdm(range(1)):
        prompt_embeds = prompt_embeds_org.repeat(batch_size, 1, 1)
        
        out = newPipe(prompt_embeds=prompt_embeds, latents=latents, num_inference_steps=num_inference_steps).images

        im = out.to('cpu')
        plt.imshow(im[0].float().detach().permute(1, 2, 0).numpy())
        plt.savefig(f'ims/out{tr}_{i}.png')
        
        prediction = predict_class(out,classifier_model)
        logits = prediction.logits
        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits[0].argmax(-1).item()
        
        
        # logits_nt = torch.cat([logits[:atk_target], logits[atk_target+1:]])# non-target
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor(batch_size * [atk_target]).to(device), reduction='mean') +  10*torch.abs(torch.mean(torch.abs(prompt_embeds_org) - 0.78))
        # loss = logits_nt.max() - logits[atk_target]
        
        tqdm.write(f"Predicted class: {classifier_model.config.id2label[predicted_class_idx]} Loss: {loss.item()}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_embeds_org, 0.1)
        # torch.nn.utils.clip_grad_norm_(latents, 0.1)
        optimizer.step()
        optimizer.zero_grad()


    out = newPipe(prompt_embeds=prompt_embeds, latents=latents, num_inference_steps=num_inference_steps).images
    
    torch.save(out, f'output{tr}.pth')
    
if __name__ == "__main__":
    main()