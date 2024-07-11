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
    tr = 48
    # atk_target = 293
    # model_id = "CompVis/stable-diffusion-v1-4"
    model_id = "OFA-Sys/small-stable-diffusion-v0"
    classifier_id = 'google/vit-base-patch16-224'
    classifier_model = ViTForImageClassification.from_pretrained(classifier_id, torch_dtype=torch.bfloat16, cache_dir='./model/', local_files_only=False).to(device)
    device = 'cuda'
    batch_size = 1
    num_inference_steps = 20
    vae, unet, image_processor, scheduler, pipe = get_model_full(model_id, device)

    target_prompt = 'photorealistic image of a crisp and delicious green apple'
    atk_target = 948

    initial_prompt = 'Red apple'


    target_prompt_embed = pipe.encode_prompt(target_prompt, device, 1, False)[0].detach()

    # classifier_model = get_classifier(classifier_id)
    # print("Attack target class:", classifier_model.config.id2label[atk_target])


    newPipe = OurPipe(pipe.vae, pipe.text_encoder, pipe.tokenizer, pipe.unet, pipe.scheduler, pipe.safety_checker, pipe.feature_extractor, pipe.image_encoder, requires_safety_checker=False)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    height = 512
    width = 512
    shape = (1, unet.config.in_channels, int(height) // vae_scale_factor, int(width) // vae_scale_factor)
    
    latents = randn_tensor(shape, generator=None, device=device, dtype=torch.bfloat16)
    prompt_embeds_org = pipe.encode_prompt(initial_prompt, device, 1, False)[0].detach() # Initial prompt
    # latents.requires_grad = True
    prompt_embeds_org.requires_grad = True

    optimizer = torch.optim.Adam([prompt_embeds_org], lr=0.01)

    classifier_sample_number = 20
    for i in tqdm(range(30)):
        prompt_embeds = prompt_embeds_org.repeat(batch_size, 1, 1)
        
        with suppress_stdout():
            out = newPipe(prompt_embeds=prompt_embeds, latents=latents, num_inference_steps=num_inference_steps, output_type='latent').images

        out = vae.decode(out / vae.config.scaling_factor,return_dict=False)[0]
        im = out.to('cpu')
        im = im/2 + 0.5
        im = im.clamp(0, 1)
        # tqdm.write(f'im range: {im.min()}, {im.max()}')
        plt.imshow(im[0].float().detach().permute(1, 2, 0).numpy())
        plt.savefig(f'ims/out{tr}_{i}.png')
        prediction = predict_class(out,classifier_model)
        logits = prediction.logits
        loss1 = torch.nn.functional.cross_entropy(logits, torch.tensor(batch_size * [atk_target]).to(device), reduction='mean')

        noise = torch.randn((classifier_sample_number,*latents.shape[1:]), device=device, dtype=torch.bfloat16)
        
        ts = torch.randint(0, scheduler.num_train_timesteps, (classifier_sample_number,)).to('cuda')
        cumprod = scheduler.alphas_cumprod.to('cuda')

        noisy_latents = out * cumprod[ts].view(-1, 1, 1, 1)**0.5 + noise * (1 - cumprod[ts]).view(-1, 1, 1, 1)**0.5
        noisy_latents = noisy_latents.to(torch.bfloat16)
        noisy_latents = scheduler.scale_model_input(noisy_latents, ts)

        noise_estimates = unet(noisy_latents, ts, encoder_hidden_states = target_prompt_embed.repeat(classifier_sample_number, 1, 1)).sample
        loss2 = 5 * F.mse_loss(noise_estimates, noise)
        tqdm.write(f'Loss: {loss1.item()} + {loss2.item()} = {loss1.item() + loss2.item()}')
        loss = loss1 + loss2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_embeds_org, 0.1)
        # torch.nn.utils.clip_grad_norm_(latents, 0.1)
        optimizer.step()
        optimizer.zero_grad()

    out = newPipe(prompt_embeds=prompt_embeds, latents=latents, num_inference_steps=num_inference_steps).images
    im = out.to('cpu')
    plt.imshow(im[0].float().detach().permute(1, 2, 0).numpy())
    plt.savefig(f'ims/out{tr}_final.png')
    
    

if __name__ == "__main__":
    main()