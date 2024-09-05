import torch.utils.data as data
import torch
import numpy as np
from configs import config

seed = config['seed']
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_data = np.load("Ising_Gauge_Theory\\Data\\ilgt_training_configs.npy")
train_labels = np.load("Ising_Gauge_Theory\\Data\\ilgt_training_labels.npy")
test_data = np.load("Ising_Gauge_Theory\\Data\\ilgt_test_configs.npy")
test_labels = np.load("Ising_Gauge_Theory\\Data\\ilgt_test_labels.npy")


train_data = torch.from_numpy(train_data).float()
train_labels = torch.from_numpy(train_labels).to(torch.int64)
test_data = torch.from_numpy(test_data).float()
test_labels = torch.from_numpy(test_labels).to(torch.int64)


train_dataset = data.TensorDataset(train_data, train_labels)
test_dataset = data.TensorDataset(test_data, test_labels)

train_loader=torch.utils.data.DataLoader(train_dataset,config['batch_size'],shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,config['batch_size'],shuffle=True)
