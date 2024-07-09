import os
os.environ["HF_HOME"] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"
os.environ['MPLCONFIGDIR'] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"

from diffusers import FlaxStableDiffusionPipeline
import matplotlib.pyplot as plt
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
print(0)
print(jax.devices())
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", revision="flax", dtype=jax.numpy.bfloat16, cache_dir='./model/', device_map='auto'
)
print(1)
prompt = "a photo of an astronaut riding a horse on mars"

prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 10
print(pipeline)
print(params)
num_samples = 1# jax.device_count()
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)

# shard inputs and rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed, num_samples)
# prompt_ids = shard(prompt_ids)
print(prompt_ids.shape)
print(3)
images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
print(4)
plt.imshow(images[0])
plt.savefig('output2.png')
print(5)