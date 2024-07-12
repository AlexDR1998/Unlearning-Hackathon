'''
THIS IS TO TRY TO PLUG IN THE DIFFUSION CLASSIFICATION PAPER CODE
THIS IS UNFINISHED AND CURRENTLY ABANDONED, DUE TO A SIMPLER PIPELINE USED
'''

import os
os.environ["HF_HOME"] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"
os.environ['MPLCONFIGDIR'] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"
import matplotlib.pyplot as plt
import torch
torch.set_default_dtype(torch.bfloat16)
from diffusers import StableDiffusionPipeline

from diffusers.optimization import get_cosine_schedule_with_warmup

#from diffuser_classifier import *
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from transformers import ViTImageProcessor, ViTForImageClassification

#from diffuser_classifier import get_classifier,predict_class
from diffuser_with_grad import *


def get_model_full(model_id="OFA-Sys/small-stable-diffusion-v0", device="cuda",ftype=torch.bfloat16,local_files_only=True):
    pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                   torch_dtype=ftype, 
                                                   cache_dir='./model/', 
                                                   local_files_only=local_files_only
                                                   )
    pipe.to(device)
    vae = pipe.vae
    unet = pipe.unet
    image_processor = pipe.image_processor
    scheduler = pipe.scheduler
    return vae, unet, image_processor, scheduler, pipe



### Diffusion Classifier
from tqdm import tqdm
import torch.nn.functional as F

###
import sys
import os
from contextlib import contextmanager
import torchvision

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def predict_class(image,classifier_model):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    normalizer = torchvision.transforms.Normalize(mean, std, inplace=False)
    resizer = torchvision.transforms.Resize((224, 224))

    return classifier_model(normalizer(resizer(image)))


def main():
    tr = 1
    
    filename= "combined_techniques_table_to_dog"
    neg_target_prompt = 'empty, blank, simple, single color, blurry, low quality'
    initial_prompt = 'photorealistic image of a table'
    random_start = False
    target_prompt = 'dog'
    LEARN_RATE = 0.05
    ITERATIONS = 30
    atk_targets = [160, 193, 181, 239, 156, 232, 182, 195, 233, 215, 151, 236, 167, 217, 248, 245, 235, 210, 246, 257, 238, 173, 213, 184, 221, 170, 171, 152, 183, 208, 189, 255, 204, 153, 268, 256, 185, 174, 186, 229, 154, 263, 259, 234, 247, 176, 258, 199, 177, 190, 230, 155, 250, 179, 220, 244, 200, 166, 178, 218, 203, 187]
    num_inference_steps = 30
    
    
    
    model_id = "OFA-Sys/small-stable-diffusion-v0"
    classifier_id = 'google/vit-base-patch16-224'
    classifier_model = ViTForImageClassification.from_pretrained(classifier_id, torch_dtype=torch.bfloat16, cache_dir='./model/', local_files_only=False).to('cuda')
    device = 'cuda'
    batch_size = 1
    vae, unet, image_processor, scheduler, pipe = get_model_full(model_id, device)

    target_prompt_embed = pipe.encode_prompt(target_prompt, device, 1, False)[0].detach()
    neg_target_prompt_embed = pipe.encode_prompt(neg_target_prompt, device, 1, False)[0].detach()

    newPipe = OurPipe(pipe.vae, pipe.text_encoder, pipe.tokenizer, pipe.unet, pipe.scheduler, pipe.safety_checker, pipe.feature_extractor, pipe.image_encoder, requires_safety_checker=False)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    height = 512
    width = 512
    shape = (1, unet.config.in_channels, int(height) // vae_scale_factor, int(width) // vae_scale_factor)
    
    latents = randn_tensor(shape, generator=None, device=device, dtype=torch.bfloat16)
    if random_start:
        prompt_embeds_org = torch.randn((1, 77, 768), device=device, dtype=torch.bfloat16)
    else:
        prompt_embeds_org = pipe.encode_prompt(initial_prompt, device, 1, False)[0].detach() # Initial prompt
    # latents.requires_grad = True
    prompt_embeds_org.requires_grad = True

    optimizer = torch.optim.NAdam([prompt_embeds_org], lr=LEARN_RATE)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=3, num_training_steps=ITERATIONS)    

    classifier_sample_number = 20
    for i in tqdm(range(ITERATIONS)):
        prompt_embeds = prompt_embeds_org.repeat(batch_size, 1, 1)
        
        with suppress_stdout():
            out = newPipe(prompt_embeds=prompt_embeds, latents=latents, num_inference_steps=num_inference_steps, output_type='latent').images

        outVae = vae.decode(out / vae.config.scaling_factor,return_dict=False)[0]
        im = outVae.to('cpu')
        im = im/2 + 0.5
        im = im.clamp(0, 1)
        # tqdm.write(f'im range: {im.min()}, {im.max()}')
        plt.imshow(im[0].float().detach().permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.savefig(f'ims/out_{filename}_{tr}_{i}.png', bbox_inches='tight', pad_inches=0)
        prediction = predict_class(outVae,classifier_model)
        logits = prediction.logits
        loss1 = torch.tensor([torch.nn.functional.cross_entropy(logits, torch.tensor(batch_size * [tar]).to(device), reduction='mean') for tar in atk_targets]).mean()

        noise = torch.randn((classifier_sample_number,*latents.shape[1:]), device=device, dtype=torch.bfloat16)
        
        ts = torch.randint(0, scheduler.num_train_timesteps, (classifier_sample_number,)).to('cuda')
        cumprod = scheduler.alphas_cumprod.to('cuda')

        noisy_latents = out * cumprod[ts].view(-1, 1, 1, 1)**0.5 + noise * (1 - cumprod[ts]).view(-1, 1, 1, 1)**0.5
        noisy_latents = noisy_latents.to(torch.bfloat16)
        noisy_latents = scheduler.scale_model_input(noisy_latents, ts)

        noise_estimates = unet(noisy_latents, ts, encoder_hidden_states = target_prompt_embed.repeat(classifier_sample_number, 1, 1)).sample
        neg_noise_estimates = unet(noisy_latents, ts, 
                                   encoder_hidden_states = neg_target_prompt_embed.repeat(classifier_sample_number, 1, 1)
                                   ).sample
        
        loss2 = 5 * F.mse_loss(noise_estimates, noise)
        loss3 = - 5 * F.mse_loss(neg_noise_estimates, noise)

        loss4 = torch.abs(torch.mean(torch.abs(prompt_embeds_org) - 0.78))

        tqdm.write(f'Loss: {loss1.item()} + {loss2.item()} + {loss3.item()} + {loss4.item()} = {loss1.item() + loss2.item() + loss3.item() + loss4.item()}')
        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_embeds_org, 0.1)
        # torch.nn.utils.clip_grad_norm_(latents, 0.1)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
    out = newPipe(prompt_embeds=prompt_embeds, latents=latents, num_inference_steps=num_inference_steps).images
    im = out.to('cpu')
    plt.imshow(im[0].float().detach().permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.savefig(f'ims/out_{filename}_{tr}_final.png', bbox_inches='tight', pad_inches=0)
    
    

if __name__ == "__main__":
    main()