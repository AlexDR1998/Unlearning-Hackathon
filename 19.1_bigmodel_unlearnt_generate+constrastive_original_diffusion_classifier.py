import os
os.environ["HF_HOME"] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"
os.environ['MPLCONFIGDIR'] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"

import torch
torch.set_default_dtype(torch.bfloat16)
from diffusers import StableDiffusionPipeline

from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection


from matplotlib import pyplot as plt

from diffuser_with_grad import *


def get_model_full(model_id="OFA-Sys/small-stable-diffusion-v0", device="cuda",ftype=torch.bfloat16):
    pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                   torch_dtype=ftype, 
                                                   cache_dir='./model/', 
                                                   local_files_only=True
                                                   )
    

    pipe.to(device)
    vae = pipe.vae
    unet = pipe.unet
    image_processor = pipe.image_processor
    scheduler = pipe.scheduler
    return vae, unet, image_processor, scheduler, pipe
    

def get_model_unlearnt(model_path = "./unlearnt_model/snapshots/38e10e5e71e8fbf717a47a81e7543cd01c1a8140", device="cuda",ftype=torch.bfloat16):
    pipe = StableDiffusionPipeline.from_pretrained(model_path,
                                                    torch_dtype=ftype,
                                                    use_safetensors=False,
                                                    safety_checker=None,
                                                    requires_safety_checker=False,
                                                    local_files_only=True,
                                                    )
    pipe.to(device)
    vae = pipe.vae
    unet = pipe.unet
    image_processor = pipe.image_processor
    scheduler = pipe.scheduler
    return vae, unet, image_processor, scheduler, pipe

def get_bigmodel_full(model_id="CompVis/stable-diffusion-v1-4", device="cuda",ftype=torch.bfloat16):
    pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                   torch_dtype=ftype, 
                                                   cache_dir='./model/', 
                                                   # local_files_only=True
                                                   )
    pipe.to(device)
    vae = pipe.vae
    unet = pipe.unet
    image_processor = pipe.image_processor
    scheduler = pipe.scheduler
    vae, unet, image_processor, scheduler, upipe = get_bigmodel_full(model_id, device=device)
    model_path = './unlearnt_model/diffusers-VanGogh-ESDx1-UNET.pt'

    uunet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet") 
    uunet.load_state_dict(torch.load(model_path)) # unlearnt unet
    upipe.unet = uunet

    return vae, unet, image_processor, scheduler, upipe


from tqdm import tqdm
import torch.nn.functional as F
import sys
import os
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout




def main():
    tr = 1
    filename = "contrastive_uVanGognDog_to_VanGognDog"
    neg_target_prompt = 'empty, blank, simple, single color, blurry'
    target_prompt = 'dog in Van Gogn style'
    initial_prompt = "dog in Van Gogn style"
    ITERATIONS = 30
    num_inference_steps = 10
    
    # classifier_id = 'google/vit-base-patch16-224'
    # model_id = "OFA-Sys/small-stable-diffusion-v0"
    # unlearnt_model_path = "./unlearnt_model/snapshots/38e10e5e71e8fbf717a47a81e7543cd01c1a8140"
    device = 'cuda'
    batch_size = 1
    # _, unet, _, _, _ = get_model_full(model_id, device)
    # uvae, uunet, uimage_processor, uscheduler, upipe = get_model_unlearnt(unlearnt_model_path, device=device)
    
    uvae, unet, uimage_processor, uscheduler, upipe = get_bigmodel_full("CompVis/stable-diffusion-v1-4", device)
    
    target_prompt_embed = upipe.encode_prompt(target_prompt, device, 1, False)[0].detach()
    neg_target_prompt_embed = upipe.encode_prompt(neg_target_prompt, device, 1, False)[0].detach()
    
    # classifier_model = get_classifier(classifier_id)
    # print("Attack target class:", classifier_model.config.id2label[atk_target])


    unewPipe = OurPipe(upipe.vae, upipe.text_encoder, upipe.tokenizer, upipe.unet, upipe.scheduler, upipe.safety_checker, upipe.feature_extractor, upipe.image_encoder, requires_safety_checker=False)

    vae_scale_factor = 2 ** (len(uvae.config.block_out_channels) - 1)
    height = 512
    width = 512
    shape = (1, unet.config.in_channels, int(height) // vae_scale_factor, int(width) // vae_scale_factor)
    
    latents = randn_tensor(shape, generator=None, device=device, dtype=torch.bfloat16)
    prompt_embeds_org = upipe.encode_prompt(initial_prompt, device, 1, False)[0].detach()
    # latents.requires_grad = True
    prompt_embeds_org.requires_grad = True

    optimizer = torch.optim.NAdam([prompt_embeds_org], lr=0.05)

    classifier_sample_number = 20
    for i in tqdm(range(ITERATIONS)):
        prompt_embeds = prompt_embeds_org.repeat(batch_size, 1, 1)
        
        with suppress_stdout():
            out = unewPipe(prompt_embeds=prompt_embeds, latents=latents, num_inference_steps=num_inference_steps, output_type='latent').images

        im = uvae.decode(out / uvae.config.scaling_factor,return_dict=False)[0].to('cpu')
        im = im/2 + 0.5
        im = im.clamp(0, 1)
        # tqdm.write(f'im range: {im.min()}, {im.max()}')
        plt.imshow(im[0].float().detach().permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.savefig(f'ims/out_{filename}_{tr}_{i}.png', bbox_inches='tight', pad_inches=0)

        noise = torch.randn((classifier_sample_number,*latents.shape[1:]), device=device, dtype=torch.bfloat16)
        
        ts = torch.randint(0, uscheduler.num_train_timesteps, (classifier_sample_number,)).to('cuda')
        cumprod = uscheduler.alphas_cumprod.to('cuda')

        noisy_latents = out * cumprod[ts].view(-1, 1, 1, 1)**0.5 + noise * (1 - cumprod[ts]).view(-1, 1, 1, 1)**0.5
        noisy_latents = noisy_latents.to(torch.bfloat16)
        noisy_latents = uscheduler.scale_model_input(noisy_latents, ts)

        noise_estimates = unet(noisy_latents, ts, 
                               encoder_hidden_states = target_prompt_embed.repeat(classifier_sample_number, 1, 1)
                               ).sample
        neg_noise_estimates = unet(noisy_latents, ts, 
                                   encoder_hidden_states = neg_target_prompt_embed.repeat(classifier_sample_number, 1, 1)
                                   ).sample
        loss = F.mse_loss(noise_estimates, noise) - F.mse_loss(neg_noise_estimates, noise)
        tqdm.write(f'Loss: {loss.item()}')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_embeds_org, 0.1)
        # torch.nn.utils.clip_grad_norm_(latents, 0.1)
        print(loss)
        optimizer.step()
        optimizer.zero_grad()

    out = unewPipe(prompt_embeds=prompt_embeds, latents=latents, num_inference_steps=num_inference_steps).images
    im = out.to('cpu')
    plt.imshow(im[0].float().detach().permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.savefig(f'ims/out_{filename}_{tr}_final.png', bbox_inches='tight', pad_inches=0)
    
    

if __name__ == "__main__":
    main()
