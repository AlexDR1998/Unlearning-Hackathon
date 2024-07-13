import torch
from diffusers import StableDiffusionPipeline

model_path = "./unlearnt_model/snapshots/38e10e5e71e8fbf717a47a81e7543cd01c1a8140"
pipe = StableDiffusionPipeline.from_pretrained(model_path,
                                               torch_dtype=torch.float16,
                                               use_safetensors=False,
                                               safety_checker=None,
                                               requires_safety_checker=False)
pipe.to("cuda")

image = pipe(prompt="An frog image in Van Gogh style").images[0]
image.save("outputs_ori/frog_van_gogh_ori.png")