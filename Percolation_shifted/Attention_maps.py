from helper_functions import load_experiment,get_attention_map
import matplotlib.pyplot as plt
import numpy as np
from Data import test_dataset,train_dataset
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse
plt.style.use('seaborn-v0_8-dark-palette')        


config, model, train_losses, test_losses, accuracies= load_experiment("my_exp")

ind = np.random.randint(28000)
ind=1835

print(ind)

head=None

patch = 30

size=8

classes = ('ground state','excited state')


# ind =16500
sample_img = torch.Tensor(test_dataset[ind][0]).unsqueeze(0)
print(sample_img.shape)

lab = test_dataset[ind][1]


conc_map = get_attention_map(model,sample_img,patch,head=head,seed=ind)
fig ,(ax1,ax2) = plt.subplots(1,2,figsize=(10,15))


im1=ax1.imshow(sample_img.detach().numpy().squeeze(0),cmap="RdBu")

cax1 = make_axes_locatable(ax1).append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im1, cax1)
cbar.ax.tick_params(labelsize=20)

ax1.axis('off')

patch_x, patch_y = (patch%size)*config["patch_size"], (patch//size)*config["patch_size"]
  
rect = plt.Rectangle((patch_x-0.5 , patch_y-0.5 ),
                     config["patch_size"], config["patch_size"], linewidth=1, edgecolor='red', facecolor='none')
ax1.add_patch(rect)

ax2.axis('off')

im = ax2.imshow(conc_map.reshape((16,16)),aspect='equal',cmap='RdBu')

# np.save(f"Percolation_shifted\\Perc_GS_Attn_p{patch}_i{ind}_new1.npy",conc_map.reshape((16,16)))


plt.tight_layout()
cax = make_axes_locatable(ax2).append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax)
cbar.ax.tick_params(labelsize=20)
plt.show()
