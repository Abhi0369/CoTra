import numpy as np
import pickle
import torch
from configs import config
from sklearn.model_selection import train_test_split
seed = 32
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def gen_random(num_snaps, size):
    snaps = torch.zeros(num_snaps, 1, size, size, dtype=torch.float32)
    # Set the spin-up channel to random {0, 1}
    snaps[:, 0, :, :] = torch.randint(2, (num_snaps, size, size))
    snaps[snaps==0]=-1
    # Set the spin-down channel to 1 - spin_up
#     snaps[:, 1, :, :] = 1 - snaps[:, 0, :, :]
    # Split into a train and val set
    num_train = int(0.9 * num_snaps)
    return snaps[:num_train], snaps[num_train:]


def load_afm_data():
    data = pickle.load(open("H_AFM\\Data\\AFM.pkl", "rb"))
    snaps = torch.tensor(np.stack(data["snapshots"], axis=0), dtype=torch.float32)
    train_data = snaps[data['train_idxs']]
    test_data = snaps[data['val_idxs']]
    
    train_data[train_data==0] = -1
    test_data[test_data==0] = -1



    return torch.unsqueeze(train_data[:,0,:,:],1),torch.unsqueeze(test_data[:,0,:,:],1)
#     return train_data,test_data

def gen_stripe(num_snaps, size):
    snaps = torch.zeros(num_snaps, 2, size, size, dtype=torch.float32)
    # Randomly pick the parity of the stripes
    if np.random.randint(2) == 0:
        snaps[:, 0, ::2, :] = 1
    else:
        snaps[:, 0, 1::2, :] = 1
    # Randomly flip spins with probability 0.3 to make mock "thermal noise"
    flips = np.random.choice([0, 1], size=snaps[:, 0, :, :].shape, p=[0.7, 0.3])
    snaps[:, 0, :, :] += torch.tensor(flips) * (1 - 2 * snaps[:, 0, :, :])
    # Set the spin-down channel to 1 - spin_up
    snaps[:, 1, :, :] = 1 - snaps[:, 0, :, :]
    num_train = int(0.9 * num_snaps)
    return snaps[:num_train], snaps[num_train:]


rand_train, rand_val = gen_random(10000, 16)
afm_train, afm_val = load_afm_data()
stripe_train, stripe_val = gen_stripe(10000, 16)

def make_datasets(train_A, train_B, val_A, val_B, batch_size=64):
    train_tensors = torch.cat([train_A, train_B])
    val_tensors = torch.cat([val_A, val_B])
    # Class A is labeled with a 0, class B is labeled with a 1
    train_labels = torch.cat([
        torch.zeros(len(train_A), dtype=torch.int64),
        torch.ones(len(train_B), dtype=torch.int64)
    ])
    val_labels = torch.cat([
        torch.zeros(len(val_A), dtype=torch.int64),
        torch.ones(len(val_B), dtype=torch.int64)
    ])
    
    train_dataset = torch.utils.data.TensorDataset(train_tensors, train_labels)
    test_dataset = torch.utils.data.TensorDataset(val_tensors, val_labels)
    
    pin_memory = True if torch.cuda.is_available() else False
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )
    return train_loader, test_loader,train_dataset, test_dataset


train_loader, test_loader,train_dataset,test_dataset = make_datasets(afm_train, rand_train, afm_val, rand_val)
