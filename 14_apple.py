
import os
os.environ["HF_HOME"] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"
os.environ['MPLCONFIGDIR'] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"

import torch
torch.set_default_dtype(torch.bfloat16)
from diffusers import StableDiffusionPipeline

from diffuser_classifier import *
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection



from diffuser_classifier import get_classifier,predict_class
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



if __name__ == "__main__":
    tr = 50
    # atk_target = 293
    # model_id = "CompVis/stable-diffusion-v1-4"
    model_id = "OFA-Sys/small-stable-diffusion-v0"
    # classifier_id = 'google/vit-base-patch16-224'
    device = 'cuda'
    batch_size = 1
    num_inference_steps = 30
    vae, unet, image_processor, scheduler, pipe = get_model_full(model_id, device)

    target_prompt = 'apple, 4k'
    target_prompt_embed = pipe.encode_prompt(target_prompt, device, 1, False)[0].detach()

    # classifier_model = get_classifier(classifier_id)
    # print("Attack target class:", classifier_model.config.id2label[atk_target])


    newPipe = OurPipe(pipe.vae, pipe.text_encoder, pipe.tokenizer, pipe.unet, pipe.scheduler, pipe.safety_checker, pipe.feature_extractor, pipe.image_encoder, requires_safety_checker=False)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    height = 512
    width = 512
    shape = (1, unet.config.in_channels, int(height) // vae_scale_factor, int(width) // vae_scale_factor)
    
    latents = randn_tensor(shape, generator=None, device=device, dtype=torch.bfloat16)
    # latents = scheduler.init_noise_sigma * latents
    
    with torch.no_grad():
        out = newPipe(prompt_embeds=target_prompt_embed, latents=latents, num_inference_steps=num_inference_steps).images
    im = out.to('cpu')
    plt.imshow(im[0].float().detach().permute(1, 2, 0).numpy())
    plt.savefig(f'ims/out{tr}_final.png')
    