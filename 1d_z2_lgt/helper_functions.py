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

def save_experiment(experiment_name, config, model, train_losses, test_losses, accuracies, base_dir="1d_z2_lgt\\experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    
    # Save the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)

    # Save the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)
    
    # Save the model
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)

def save_checkpoint(experiment_name, model, epoch, base_dir="1d_z2_lgt\\experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')
    torch.save(model.state_dict(), cpfile)

    
def load_experiment(experiment_name, checkpoint_name="model_final.pt", base_dir="1d_z2_lgt\\experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    # Load the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)
    # Load the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']
    # Load the model
    model = ViT(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile))
    return config, model, train_losses, test_losses, accuracies

def visualize_images():
    trainset = train_dataset
    classes = ('Disordered Phase','Ordered Phase')

    # Pick 30 samples randomly
    indices = torch.randperm(len(trainset))[:3]
    images = [np.asarray(trainset[i][0]) for i in indices]
    labels = [trainset[i][1] for i in indices]
    # Visualize the images using matplotlib
    fig = plt.figure(figsize=(10, 10))
    for i in range(3):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        images[i] = np.reshape(images[i],(40,40,1))
        ax.imshow(images[i][:,:,0],cmap='gray')
        ax.set_title(classes[labels[i]])


def get_attention_map(model, sample_img, head=None, return_raw=False,seed=32):
    """This returns the attentions when CLS token is used as query in the last attention layer, averaged over all attention heads"""
    seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    model.eval()
    logits, attentions,_ = model(sample_img, output_attentions=True)
    patch_size = config['patch_size']

#     w_featmap = sample_img.shape[-2] // patch_size
#     h_featmap = sample_img.shape[-1] // patch_size
    w_featmap=8
    h_featmap = w_featmap
    attentions=attentions[0]
    nh = attentions.shape[1]  # number of heads


    # this extracts the attention when cls is used as query
#     attentions = attentions[0, :, :,:].reshape(nh, -1)
    attentions = attentions[0, :, :].reshape(nh, -1)

    if return_raw:
        return torch.mean(attentions, dim=0).squeeze().detach().cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
#     attentions = torch.nn.functional.interpolate(
#         attentions.unsqueeze(0), scale_factor=patch_size, mode="bilinear"
#     )[0]
    if head == None:
        mean_attention = torch.mean(attentions, dim=0).squeeze().detach().cpu().numpy()
        return mean_attention
    else:
        return attentions[head].squeeze().detach().cpu().numpy()