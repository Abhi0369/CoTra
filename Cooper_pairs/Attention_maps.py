from helper_functions import load_experiment,get_attention_map
import matplotlib.pyplot as plt
import numpy as np
from Data import test_dataset,train_dataset
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse       
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

config, model, train_losses, test_losses, accuracies= load_experiment("my_exp")
ind = np.random.randint(6000)

ind=2577
print(ind)
head=None

patch = 170

size=15
grid_size=30

radius =10
center = grid_size // 2

classes = ('ground state','excited state')

cmap = plt.get_cmap('RdBu')

colors = cmap(np.linspace(0, 1, 256))

# Set the middle color to white
middle = int(len(colors) / 2)
colors[middle] = [1, 1, 1, 1]

# Create a new colormap with the modified colors
new_cmap = mcolors.LinearSegmentedColormap.from_list('custom_RdBu', colors)


sample_img = torch.Tensor(test_dataset[ind][0]).unsqueeze(0)
print(sample_img.shape)

lab = test_dataset[ind][1]

ellipse = Ellipse((center, center), 2 * radius, 2 * radius, edgecolor='black', facecolor='none', linestyle='dotted')
conc_map = get_attention_map(model,sample_img,patch,head=head,seed=ind)

maxm=np.max(conc_map)
minm = np.min(conc_map)

fig = plt.figure(figsize=(16,8))



gs = gridspec.GridSpec(1, 2)
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])

im1=ax1.imshow(sample_img.detach().numpy().squeeze(0),new_cmap)

ax1.add_patch(ellipse)


ax1.axis('off')

ellipse = Ellipse((center, center), 2 * radius, 2 * radius, edgecolor='black', facecolor='none', linestyle='dotted')
ax2.add_patch(ellipse)



patch_x, patch_y = (patch%size)*config["patch_size"], (patch//size)*config["patch_size"]
  
rect = plt.Rectangle((patch_x-0.5 , patch_y-0.5 ),
                     config["patch_size"], config["patch_size"], linewidth=1, edgecolor='black', facecolor='none')
ax1.add_patch(rect)

ax2.axis('off')

norm = plt.Normalize(vmin=conc_map.min(), vmax=conc_map.max())

# Create a colormap from 'RdBu'
cmap = plt.get_cmap('RdBu')

zero_position = norm(0)

colors = cmap(np.linspace(0, 1, 256))


colors[int(256 * zero_position)] = [1, 1, 1, 1] 

new_cmap = mcolors.LinearSegmentedColormap.from_list('custom_RdBu', colors)



im = ax2.imshow(conc_map.reshape((30,30)),aspect='equal',cmap="RdBu",vmin=-24,vmax=24)

plt.tight_layout()
cax = make_axes_locatable(ax2).append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax)
cbar.ax.tick_params(labelsize=20)
hd = config['hidden_size']
pz = config["patch_size"]
lr =config["lr"]
e = config["epochs"]
s = config["seed"]
gamma = config["gamma"]

plt.tight_layout()


plt.show()
