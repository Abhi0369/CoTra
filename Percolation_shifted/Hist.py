import gc
from helper_functions import load_experiment
import torch
from Data import test_dataset
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('seaborn-v0_8-talk')    
plt.style.use('seaborn-v0_8-dark-palette')        
    


config, model, train_losses, test_losses, accuracies= load_experiment("my_exp")


device="cuda" if torch.cuda.is_available() else "cpu"
seed=32
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
model = model.to(device)
model.eval()  # Make sure the model is in evaluation mode

# Load random images
num_images = 4000  # Number of images you want to process

# Using a DataLoader to handle batches efficiently

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
        
        # Get the attention maps from the last block
        M= torch.sum(images,axis=(1,2,3))

        logits, attention_maps, emb_out = model(images, output_attentions=True)
        attn = attention_maps[0]
        all_attn.extend(attn.cpu().detach().tolist())

        # Store results
        all_predictions.extend(torch.argmax(logits, dim=1).cpu().detach().tolist())
        all_logits.extend(logits.detach().cpu().tolist())
#         all_embeddings.extend(emb_out.cpu().detach().tolist())  # Mean across sequence length
        all_labels.extend(labels.detach().cpu().tolist())
#         all_M.extend(M.detach().cpu().tolist())
#         all_pos.extend(emb_out[0])
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

all_M = np.array(all_M[:num_images])
# attn = np.empty((2048,17,17))
attn_all =np.array(all_attn[:num_images])

a = attn_all.mean((-1,-2,-3))
d1 =[]
d2 =[]

for i in range(2000):
    if all_labels[i]==0:
        d1.append(a[i])
    else:
        d2.append(a[i])
        
data_min = min(np.min(d1), np.min(d2))
data_max = max(np.max(d1), np.max(d2))

# Calculate the bin width
# bin_width = 0.1  # You can adjust this value as needed

# # Calculate the number of bins based on the bin width
# num_bins = int(np.ceil((data_max - data_min) / bin_width))

# plt.hist(d1, bins=num_bins, alpha=0.5, label='Label 0')
# plt.hist(d2, bins=num_bins, alpha=0.5, label='Label 1')
# plt.figure(figsize=(10, 6))
# plt.hist(d1, bins=50, alpha=0.8, label='Percolated')
# plt.hist(d2, bins=50, alpha=0.8, label='Non Percolated')
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)


# # Add legend
# plt.legend(fontsize=20)
# plt.xlabel("Mean Attention",fontsize=20)
# plt.ylabel("Count",fontsize=20)
# plt.vlines(x=0,ymin=0,ymax =75,colors='red')
# plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
np.save("Percolation_shifted\\Hist_percolated_new.npy",d1)
np.save("Percolation_shifted\\Hist_nonpercolated_new.npy",d2)

# Plot histograms
# ax.hist(d1, bins=50, alpha=0.8, label='Percolated')
# ax.hist(d2, bins=50, alpha=0.8, label='Non Percolated')

bins = np.linspace(data_min, data_max, 100) 

ax.hist(d1, bins=bins, alpha=0.8,label='Percolated',color="#E3B345")
ax.hist(d2, bins=bins, alpha=0.8, label='Non Percolated',color="#753589")

# Set ticks font size
ax.tick_params(axis='both', which='major', labelsize=20)

# Add legend
ax.legend(fontsize=20)

# Set labels
ax.set_xlabel("Mean Attention", fontsize=20)
ax.set_ylabel("Count", fontsize=20)

# Draw a vertical line at x=0 that extends to the top of the plot
ylim = ax.get_ylim()

ax.vlines(x=0, ymin=ylim[0], ymax=ylim[1], colors='red')
ax.set_ylim(top=ylim[1] )

# Show the plot
plt.show()

# plt.savefig("perc_2_mean_attn_coorrvit",bbox_inches='tight')
len(d1),len(d2)
hd = config['hidden_size']
pz = config["patch_size"]
lr =config["lr"]
e = config["epochs"]
s = config["seed"]
gamma = config["gamma"]
