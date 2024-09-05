import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from corr_vit import ViT
from Data import train_dataset, test_dataset
from configs import config
import math 
from torch.nn import functional as F
from matplotlib.patches import Ellipse, Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

def save_experiment(experiment_name, config, model, train_losses, test_losses, accuracies, base_dir="Cooper_pairs\\experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    
    # Save the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)
#     train_losses_np = [tensor.detach().cpu().numpy() for tensor in train_losses]
#     test_losses_np = [tensor.detach().cpu().numpy() for tensor in test_losses]


    # Save the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
#             'train_losses': [tensor.tolist() for tensor in train_losses_np],
#             'test_losses':  [tensor.tolist() for tensor in train_losses_np],
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)
    
    # Save the model
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)

def save_checkpoint(experiment_name, model, epoch, base_dir="Cooper_pairs\\experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')
    torch.save(model.state_dict(), cpfile)

    
def load_experiment(experiment_name, checkpoint_name="model_final.pt", base_dir="Cooper_pairs\\experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    # Load the config
    # configfile = os.path.join(outdir, 'config.json')
    configfile = "Cooper_pairs\\experiments\\my_exp\\config.json"
    with open(configfile, 'r') as f:
        config = json.load(f)
    # Load the metrics
    # jsonfile = os.path.join(outdir, 'metrics.json')
    jsonfile = "Cooper_pairs\\experiments\\my_exp\\metrics.json"
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']
    # Load the model
    model = ViT(config)
    # cpfile = os.path.join(outdir, checkpoint_name)
    cpfile ="Cooper_pairs\\experiments\\my_exp\\model_final.pt"
    model.load_state_dict(torch.load(cpfile))
    return config, model, train_losses, test_losses, accuracies

def get_attention_map(model, sample_img,patch, head=None, return_raw=False,seed=32):
    """This returns the attentions when CLS token is used as query in the last attention layer, averaged over all attention heads"""
    seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    model.eval()
    logits, attentions,_ = model(sample_img, output_attentions=True)
    patch_size = config['patch_size']
    

    w_featmap = sample_img.shape[-2] // patch_size
    h_featmap = sample_img.shape[1] // patch_size

    attentions=attentions[0]
    print(attentions.shape)
    nh = attentions.shape[1]  # number of heads

    attentions = attentions[0, 0,patch,:].reshape(nh, -1)


    if return_raw:
        return torch.mean(attentions, dim=0).squeeze().detach().cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = torch.nn.functional.interpolate(
        attentions.unsqueeze(0), scale_factor=patch_size, mode="bilinear"
    )[0]
    if head == None:

        mean_attention = attentions.detach().cpu().numpy()
        return mean_attention
    else:
        return attentions[head].squeeze().detach().cpu().numpy()