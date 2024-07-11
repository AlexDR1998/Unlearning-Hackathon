'''
THIS IS TO TRY TO PLUG IN THE DIFFUSION CLASSIFICATION PAPER CODE
THIS IS UNFINISHED AND CURRENTLY ABANDONED, DUE TO A SIMPLER PIPELINE USED
'''

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



### Diffusion Classifier
from tqdm import tqdm
import torch.nn.functional as F
# def eval_prob_adaptive(unet, latent, text_embeds, scheduler, args_batch_size,
#                        latent_size=64, 
#                        all_noise=None, args_dtype='bfloat16',
#                        args_n_samples=[500]*5,
#                        topk_to_keep=[10]*5,
#                        args_n_trials=5,
#                        args_version='1-4', #TODO: NOT TRUE
#                        args_loss='l2'
#                        ):
    
#     scheduler_config = get_scheduler_config(args_version)
#     T = scheduler_config['num_train_timesteps']
#     max_n_samples = max(args_n_samples)

#     # if all_noise is None:
#     #     all_noise = torch.randn((max_n_samples * args.n_trials, 4, latent_size, latent_size), device=latent.device)
#     # if args.dtype == 'float16':
#     #     all_noise = all_noise.half()
#     #     scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()
        
#     all_noise = torch.randn((max_n_samples * args_n_trials, 4, latent_size, latent_size), device=latent.device, dtype=args_dtype)

#     data = dict()
#     t_evaluated = set()
#     remaining_prmpt_idxs = list(range(len(text_embeds)))
#     start = T // max_n_samples // 2
#     t_to_eval = list(range(start, T, T // max_n_samples))[:max_n_samples]

#     for n_samples, n_to_keep in zip(args_n_samples, topk_to_keep):
#         ts = []
#         noise_idxs = []
#         text_embed_idxs = []
#         curr_t_to_eval = t_to_eval[len(t_to_eval) // n_samples // 2::len(t_to_eval) // n_samples][:n_samples]
#         curr_t_to_eval = [t for t in curr_t_to_eval if t not in t_evaluated]
#         for prompt_i in remaining_prmpt_idxs:
#             for t_idx, t in enumerate(curr_t_to_eval, start=len(t_evaluated)):
#                 ts.extend([t] * args_n_trials)
#                 noise_idxs.extend(list(range(args_n_trials * t_idx, args_n_trials * (t_idx + 1))))
#                 text_embed_idxs.extend([prompt_i] * args_n_trials)
#         t_evaluated.update(curr_t_to_eval)
#         pred_errors = eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs, 
#                                  text_embeds, text_embed_idxs, args_batch_size, args_dtype, args_loss,
#                                  device=latent.device
#                                  )
#         # match up computed errors to the data
#         for prompt_i in remaining_prmpt_idxs:
#             mask = torch.tensor(text_embed_idxs) == prompt_i
#             prompt_ts = torch.tensor(ts)[mask]
#             prompt_pred_errors = pred_errors[mask]
#             if prompt_i not in data:
#                 data[prompt_i] = dict(t=prompt_ts, pred_errors=prompt_pred_errors)
#             else:
#                 data[prompt_i]['t'] = torch.cat([data[prompt_i]['t'], prompt_ts])
#                 data[prompt_i]['pred_errors'] = torch.cat([data[prompt_i]['pred_errors'], prompt_pred_errors])

#         # compute the next remaining idxs
#         errors = [-data[prompt_i]['pred_errors'].mean() for prompt_i in remaining_prmpt_idxs]
#         best_idxs = torch.topk(torch.tensor(errors), k=n_to_keep, dim=0).indices.tolist()
#         remaining_prmpt_idxs = [remaining_prmpt_idxs[i] for i in best_idxs]

#     # organize the output
#     assert len(remaining_prmpt_idxs) == 1
#     pred_idx = remaining_prmpt_idxs[0]

#     return pred_idx, data


# def eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs, device,
#                text_embeds, text_embed_idxs, batch_size=32, dtype='float32', loss='l2'):
#     assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
#     pred_errors = torch.zeros(len(ts), device='cpu')
#     idx = 0
#     with torch.inference_mode():
#         for _ in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
#             batch_ts = torch.tensor(ts[idx: idx + batch_size])
#             noise = all_noise[noise_idxs[idx: idx + batch_size]]
#             noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(device) + \
#                             noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(device)
#             t_input = batch_ts.to(device).half() if dtype == 'float16' else batch_ts.to(device)
#             text_input = text_embeds[text_embed_idxs[idx: idx + batch_size]]
#             noise_pred = unet(noised_latent, t_input, encoder_hidden_states=text_input).sample
#             if loss == 'l2':
#                 error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
#             elif loss == 'l1':
#                 error = F.l1_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
#             elif loss == 'huber':
#                 error = F.huber_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
#             else:
#                 raise NotImplementedError
#             pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
#             idx += len(batch_ts)
#     return pred_errors
###
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
    filename = "contrastive_apple"
    neg_target_prompt = 'empty, blank, simple, single color, lacking'
    target_prompt = 'apple, 4k'
    ITERATIONS = 30
    num_inference_steps = 10
    
    # classifier_id = 'google/vit-base-patch16-224'
    model_id = "OFA-Sys/small-stable-diffusion-v0"
    device = 'cuda'
    batch_size = 1
    vae, unet, image_processor, scheduler, pipe = get_model_full(model_id, device)

    target_prompt_embed = pipe.encode_prompt(target_prompt, device, 1, False)[0].detach()
    neg_target_prompt_embed = pipe.encode_prompt(neg_target_prompt, device, 1, False)[0].detach()
    
    # classifier_model = get_classifier(classifier_id)
    # print("Attack target class:", classifier_model.config.id2label[atk_target])


    newPipe = OurPipe(pipe.vae, pipe.text_encoder, pipe.tokenizer, pipe.unet, pipe.scheduler, pipe.safety_checker, pipe.feature_extractor, pipe.image_encoder, requires_safety_checker=False)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    height = 512
    width = 512
    shape = (1, unet.config.in_channels, int(height) // vae_scale_factor, int(width) // vae_scale_factor)
    
    latents = randn_tensor(shape, generator=None, device=device, dtype=torch.bfloat16)
    prompt_embeds_org = pipe.encode_prompt('apple, 4k', device, 1, False)[0].detach()
    # latents.requires_grad = True
    prompt_embeds_org.requires_grad = True

    optimizer = torch.optim.Adam([prompt_embeds_org], lr=0.01)

    classifier_sample_number = 5
    for i in tqdm(range(ITERATIONS)):
        prompt_embeds = prompt_embeds_org.repeat(batch_size, 1, 1)
        
        with suppress_stdout():
            out = newPipe(prompt_embeds=prompt_embeds, latents=latents, num_inference_steps=num_inference_steps, output_type='latent').images

        im = vae.decode(out / vae.config.scaling_factor,return_dict=False)[0].to('cpu')
        im = im/2 + 0.5
        im = im.clamp(0, 1)
        # tqdm.write(f'im range: {im.min()}, {im.max()}')
        plt.imshow(im[0].float().detach().permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.savefig(f'ims/out_{filename}_{tr}_{i}.png', bbox_inches='tight', pad_inches=0)

        noise = torch.randn((classifier_sample_number,*latents.shape[1:]), device=device, dtype=torch.bfloat16)
        
        ts = torch.randint(0, scheduler.num_train_timesteps, (classifier_sample_number,)).to('cuda')
        cumprod = scheduler.alphas_cumprod.to('cuda')

        noisy_latents = out * cumprod[ts].view(-1, 1, 1, 1)**0.5 + noise * (1 - cumprod[ts]).view(-1, 1, 1, 1)**0.5
        noisy_latents = noisy_latents.to(torch.bfloat16)
        noisy_latents = scheduler.scale_model_input(noisy_latents, ts)

        noise_estimates = unet(noisy_latents, ts, 
                               encoder_hidden_states = target_prompt_embed.repeat(classifier_sample_number, 1, 1)
                               ).sample
        neg_noise_estimates = unet(noisy_latents, ts, 
                                   encoder_hidden_states = neg_target_prompt_embed.repeat(classifier_sample_number, 1, 1)
                                   ).sample
        loss = F.mse_loss(noise_estimates, noise) - 2*F.mse_loss(neg_noise_estimates, noise)
        tqdm.write(f'Loss: {loss.item()}')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_embeds_org, 0.1)
        # torch.nn.utils.clip_grad_norm_(latents, 0.1)
        print(loss)
        optimizer.step()
        optimizer.zero_grad()

    out = newPipe(prompt_embeds=prompt_embeds, latents=latents, num_inference_steps=num_inference_steps).images
    im = out.to('cpu')
    plt.imshow(im[0].float().detach().permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.savefig(f'ims/out_{filename}_{tr}_final.png', bbox_inches='tight', pad_inches=0)
    
    

if __name__ == "__main__":
    main()
