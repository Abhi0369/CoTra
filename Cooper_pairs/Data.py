import numpy as np
import torch.utils.data as data
import torch


dat1 = np.load("C:\\Master' Thesis\\Corr_ViT\\Cooper_pairs\\Data\\cp_gs_2_2_binary_u_r_10.npz")

dat2 = np.load("C:\\Master' Thesis\\Corr_ViT\\Cooper_pairs\\Data\\cp_es_22_binary_u.npz")

dat1 = dat1['arr_0']
dat2 = dat2['arr_0']
BATCH_SIZE = 64
n = 9000
n1 = 6000
n2 = 3000

lab1 = np.zeros((n), dtype=np.int8)
lab2 = np.ones((n), dtype=np.int8)

tdat1 = np.expand_dims(np.array(dat1[:n], dtype=np.float32),-1)
tdat2 = np.expand_dims(np.array(dat2[:n], dtype=np.float32),-1)

train_data = np.concatenate((tdat1[:n1], tdat2[:n1]))
train_labels = np.concatenate((lab1[:n1], lab2[:n1]))
test_data = np.concatenate((tdat1[n1:], tdat2[n1:]))
test_labels = np.concatenate((lab1[n1:], lab2[n1:]))



# Convert to torch tensors with appropriate types
train_data = torch.tensor(train_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.int64)
test_data = torch.tensor(test_data, dtype=torch.float32)

test_labels = torch.tensor(test_labels, dtype=torch.int64)

print(train_data.shape)

# Create DataLoader instances
train_dataset = data.TensorDataset(train_data, train_labels)
test_dataset = data.TensorDataset(test_data, test_labels)

train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


