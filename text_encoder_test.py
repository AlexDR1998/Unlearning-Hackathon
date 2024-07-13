import os
os.environ["HF_HOME"] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"
os.environ['MPLCONFIGDIR'] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"

import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

import torch
from einops import rearrange
# from diffusers import StableDiffusionPipeline
# from diffusers.utils.torch_utils import randn_tensor
# import torchvision
# from transformers import ViTImageProcessor, ViTForImageClassification
# import numpy as np
# import nltk
# #nltk.download('abc',download_dir=".venv/nltk_data")




# def get_model(model_id="OFA-Sys/small-stable-diffusion-v0", device="cuda",ftype=torch.bfloat16):
#     pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=ftype, cache_dir='./model/',local_files_only=True)
#     pipe.to(device)
#     return pipe
#     #text_encoder = pipe.text_encoder
#     #tokenizer = pipe.tokenizer

def main():
    # model = get_model()
    # #print(model.encode_prompt("test prompt","cuda",1,False))
    # #print(str(nltk.corpus.brown).replace('\\\\','/'))
    # words_prompts = nltk.corpus.words.words()
    # with torch.no_grad():
    #     n_words = len(words_prompts)
    #     word_embeddings = []
    #     MINI_BATCHES = 1000
    #     print(f"Words per mini-batch {n_words//MINI_BATCHES}")
    #     for i in tqdm(range(235,MINI_BATCHES)):
    #         em = model.encode_prompt(words_prompts[i*n_words//MINI_BATCHES:(i+1)*n_words//MINI_BATCHES],"cuda",n_words//MINI_BATCHES,False)[0].cpu()
    #         word_embeddings.append(em)
    word_array = torch.load("word_embeddings_dictionary.pt",map_location="cpu")[]
    word_array_flat = rearrange(word_array,"W u v -> W (u v)")
    
    print(word_array.shape)
main()    

    
    