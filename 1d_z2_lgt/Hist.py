import gc
from helper_functions import load_experiment
import torch
from Data import test_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from cycler import cycler

from matplotlib import rcParams
fontsize=24
rcParams.update({'font.size': fontsize})
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['axes.prop_cycle'] = cycler(color=["#3071B7", "#B22222", "#E3B345", "#753589", "grey"])        
    


config, model, train_losses, test_losses, accuracies= load_experiment("my_exp")


device="cuda" if torch.cuda.is_available() else "cpu"
seed=32
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
model = model.to(device)
model.eval()  


num_images = 4000  # Number of images you want to process


batch_size = 64  # Define an appropriate batch size for your system
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Variables to store results
all_predictions = []
all_logits = []
all_embeddings = []
all_labels=[]
all_M=[]
all_pos =[]
all_attn = []
with torch.no_grad():  # Disable gradient computation for efficiency
    
    for images, labels in test_loader:
        # Move the images to the device
        images = images.to(device, dtype=torch.float32)  # Using float16 for memory efficiency

        logits, attention_maps, emb_out = model(images, output_attentions=True)
        attn = attention_maps[1]
        all_attn.extend(attn.cpu().detach().tolist())

        # Store results
        all_predictions.extend(torch.argmax(logits, dim=1).cpu().detach().tolist())
        all_logits.extend(logits.detach().cpu().tolist())

        all_labels.extend(labels.detach().cpu().tolist())

        del images, logits, attention_maps, emb_out,labels
        gc.collect()
        torch.cuda.empty_cache()  #
        if len(all_predictions) >= num_images:  # Break the loop once we have enough images
            break

# Convert lists to arrays or tensors if needed
all_predictions = np.array(all_predictions[:num_images])
all_logits = np.array(all_logits[:num_images])
# all_embeddings = np.array(all_embeddings[:num_images])
all_labels = np.array(all_labels[:num_images])

attn_all =np.array(all_attn[:num_images])

a = attn_all.mean((-1,-2,-3))
d1 =[]
d2 =[]
attn_all = attn_all.reshape((num_images,64))
for i in range(num_images):
    if all_labels[i]==0:
        d1.append(a[i])
    else:
        d2.append(a[i])
        
data_min = min(np.min(d1), np.min(d2))
data_max = max(np.max(d1), np.max(d2))


plt.figure(figsize=(6, 6))

bins = np.linspace(data_min, data_max, 50) 

plt.hist(d1, bins=bins, alpha=0.8,label='Fulfills GL',color="#E3B345")
plt.hist(d2, bins=bins, alpha=0.8, label='Violates GL',color="#753589")

plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)



# Add legend
plt.legend(fontsize=fontsize)
plt.xlabel("Mean Attention",fontsize=fontsize)
plt.ylabel("Count",fontsize=fontsize)
plt.show()
