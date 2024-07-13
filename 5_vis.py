import os
os.environ["HF_HOME"] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"
os.environ['MPLCONFIGDIR'] = "/scratch/space1/ic084/unlearning/Unlearning-Hackathon/model"

import torch 
import matplotlib.pyplot as plt
with torch.no_grad():
    im = torch.load('output2.pth').to('cpu')
    print(im)
    plt.imshow(im[0].float().detach().permute(1, 2, 0).numpy())
    plt.savefig('oooo2.png')