from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, StableDiffusionPipeline

import torch

device='cpu'
model_id = "CompVis/stable-diffusion-v1-4"

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
    return vae, unet, image_processor, scheduler, pipe

vae, unet, image_processor, scheduler, pipe = get_bigmodel_full(model_id, device=device)
model_path = './unlearnt_model/diffusers-VanGogh-ESDx1-UNET.pt'

uunet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet") 
uunet.load_state_dict(torch.load(model_path)) # unlearnt unet
pipe.unet = uunet


