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


def main():
    tr = 16
    atk_target = 123
    # model_id = "CompVis/stable-diffusion-v1-4"
    model_id = "OFA-Sys/small-stable-diffusion-v0"
    classifier_id = 'google/vit-base-patch16-224'

    device = torch.device("cuda:0")
    vae, unet, image_processor, scheduler = get_model(model_id, device)
    classifier_model = get_classifier(classifier_id)
    print("Attack target class:", classifier_model.config.id2label[atk_target])
    height = 256
    width = 256
    num_inference_steps = 10
    

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
    latents = randn_tensor(shape, device=device, dtype=prompt_embeds.dtype)
    latents.to(device)
    latents.requires_grad=True
    prompt_embeds.requires_grad=True

    optimizer = torch.optim.Adam([latents, prompt_embeds], lr=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    # targetImg = torch.load('output.pth').to(device)
    for i in tqdm(range(20)):
        timesteps = None
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, device, timesteps
        )
        out = forward(timesteps, num_inference_steps, scheduler, unet, vae, prompt_embeds, image_processor, 'PIL.Image', latents)

        im = out.to('cpu')
        plt.imshow(im[0].float().detach().permute(1, 2, 0).numpy())
        plt.savefig(f'ims/out{tr}_{i}.png')
        
        prediction = predict_class(out,classifier_model)
        logits = prediction.logits[0]
        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()
        
        
        # logits_nt = torch.cat([logits[:atk_target], logits[atk_target+1:]])# non-target
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([atk_target]).to(device))
        # loss = logits_nt.max() - logits[atk_target]
        
        tqdm.write(f"Predicted class: {classifier_model.config.id2label[predicted_class_idx]} Loss: {loss.item()}")

        loss.backward()
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
