import numpy as np
import torch.utils.data as data
import torch
import matplotlib.pyplot as plt

dat1=np.load("C:\\Master' Thesis\\Corr_ViT\\Percolation_shifted\\Data\\percolated (1).npy")
dat2=np.load("C:\\Master' Thesis\\Corr_ViT\\Percolation_shifted\\Data\\non_percolated_shifted.npy")
# dat2 = np.load("C:\\Master' Thesis\\Corr_ViT\\Percolation_shifted\\Data\\non_percolating_1.npy")


BATCH_SIZE=64
n=44000
n1=30000
n2=14000

lab1= np.zeros((n))
lab2=np.ones((n))

tdat1=np.asarray(dat1[:n])
tdat2=np.asarray(dat2[:n])

train_data = np.concatenate((tdat1[:n1],tdat2[:n1]))
train_labels = np.concatenate((lab1[:n1],lab2[:n1]))
test_data = np.concatenate((tdat1[n1:],tdat2[n1:]))

test_labels = np.concatenate((lab1[n1:],lab2[n1:]))


train_data = torch.from_numpy(train_data).float()
train_labels = torch.from_numpy(train_labels).to(torch.int64)
test_data = torch.from_numpy(test_data).float()
test_labels = torch.from_numpy(test_labels).to(torch.int64)

train_dataset = data.TensorDataset(train_data, train_labels)
test_dataset = data.TensorDataset(test_data, test_labels)


train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


