import os
os.environ["HF_HOME"] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"
os.environ['MPLCONFIGDIR'] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"

import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

import torch

from diffuser_classifier import *



torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)


def get_model_full(model_id="OFA-Sys/small-stable-diffusion-v0", device="cuda",ftype=torch.bfloat16):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=ftype, cache_dir='./model/', local_files_only=True)
    pipe.to(device)
    vae = pipe.vae
    unet = pipe.unet
    image_processor = pipe.image_processor
    scheduler = pipe.scheduler
    return vae, unet, image_processor, scheduler, pipe

def main():

    tr = 25
    atk_target = 293
    # model_id = "CompVis/stable-diffusion-v1-4"
    model_id = "OFA-Sys/small-stable-diffusion-v0"
    classifier_id = 'google/vit-base-patch16-224'

    device = torch.device("cuda:0")
    vae, unet, image_processor, scheduler, pipe = get_model_full(model_id, device)
    classifier_model = get_classifier(classifier_id)
    print("Attack target class:", classifier_model.config.id2label[atk_target])
    height = 256
    width = 256
    num_inference_steps = 50
    
    # class_string = classifier_model.config.id2label[atk_target]
    class_string = 'car'
    class_embed = pipe.encode_prompt(class_string, "cuda",1,False)[0]


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

    # prompt_embeds_org = torch.randn((1, 77, 768), device=device, dtype=torch.bfloat16)
    prompt_embeds_org = class_embed.detach()
    latents = randn_tensor(shape, device=device, dtype=torch.bfloat16)
    latents.to(device)
    # latents.requires_grad=True
    prompt_embeds_org.requires_grad=True

    optimizer = torch.optim.Adam([prompt_embeds_org], lr=0.2)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    # targetImg = torch.load('output.pth').to(device)
    for i in tqdm(range(30)):
        prompt_embeds = prompt_embeds_org.repeat(batch_size, 1, 1)
        timesteps = None
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, device, timesteps
        )
        out = forward(timesteps, num_inference_steps, scheduler, unet, vae, prompt_embeds, image_processor, 'PIL.Image', latents)

        im = out.to('cpu')
        plt.imshow(im[0].float().detach().permute(1, 2, 0).numpy())
        plt.savefig(f'ims/out{tr}_{i}.png')
        
        prediction = predict_class(out,classifier_model)
        logits = prediction.logits
        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits[0].argmax(-1).item()
        
        
        # logits_nt = torch.cat([logits[:atk_target], logits[atk_target+1:]])# non-target
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor(batch_size * [atk_target]).to(device), reduction='mean')
        # loss = logits_nt.max() - logits[atk_target]
        
        tqdm.write(f"Predicted class: {classifier_model.config.id2label[predicted_class_idx]} Loss: {loss.item()}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_embeds_org, 0.1)
        # torch.nn.utils.clip_grad_norm_(latents, 0.1)
        optimizer.step()
        optimizer.zero_grad()


    timesteps = None
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler, num_inference_steps, device, timesteps
    )
    out = forward(timesteps, num_inference_steps, scheduler, unet, vae, prompt_embeds, image_processor, 'PIL.Image', latents)
    
    
    torch.save(out, f'output{tr}.pth')
    
    



if __name__ == "__main__":
    main()
