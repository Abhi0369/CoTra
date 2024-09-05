from configs import config
from helper_functions import load_experiment
import torch
import numpy as np
from Data import train_loader,test_loader
import sklearn
from sklearn import *
from tqdm import tqdm
import matplotlib.pyplot as plt
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
         


num_cor= config["num_corr"]


def Output(loader):

    device= "cuda" if torch.cuda.is_available() else "cpu"

    config, model, train_losses, test_losses, accuracies= load_experiment("my_exp")
    model=model.to(device)
    model.eval

    Crs=np.zeros((0, num_cor))
        
    lab =np.zeros(0)

    with torch.no_grad():

        for batch in loader:

            batch = [t.to(device) for t in batch]
            images, labels = batch
            logits = model(images)
         
            corrs = model.output

            Crs=np.concatenate((Crs,corrs.cpu()),axis=0)

            lab=np.concatenate((lab,labels.cpu()),axis=0)
            

    return Crs,lab


crs_train, train_labels = Output(train_loader)
crs_test, test_labels= Output(test_loader)


train_mean = crs_train.mean(axis=0, keepdims=True)
train_std = crs_train.std(axis=0, keepdims=True)
test_mean = crs_test.mean(axis=0, keepdims=True)
test_std = crs_test.std(axis=0, keepdims=True)
crs_train = (crs_train - train_mean) / (train_std)
crs_test = (crs_test - test_mean) / (test_std)


logistic_clf = sklearn.linear_model.LogisticRegression(
    penalty='l1', solver='saga', warm_start=True, tol=1e-4,
    fit_intercept=True, max_iter=int(1e4)
)

inv_lmdas = np.logspace(-6, 1, num=500)

Beta_as = np.zeros((len(inv_lmdas), num_cor* 1))
train_accs = np.zeros(len(inv_lmdas))
val_accs = np.zeros(len(inv_lmdas))

for i, inv_lm in tqdm(enumerate(inv_lmdas), total=len(inv_lmdas)):

    logistic_clf.set_params(C=inv_lm)

    logistic_clf.fit(crs_train, train_labels)
 
    Beta_as[i] = logistic_clf.coef_.ravel().copy()

    train_accs[i] = logistic_clf.score(crs_train, train_labels)
    val_accs[i] = logistic_clf.score(crs_test, test_labels)

        

fig, axs = plt.subplots(1, 1, figsize=(6, 6))

axs.set_xscale('log')
axs.set_ylabel(r"$\beta^{n}$",fontsize=fontsize)
axs.set_xscale('log')
axs.set_xlabel(r"$1/\lambda$",fontsize=fontsize)
axs.tick_params(axis='both', which='major', labelsize=fontsize)
axs.tick_params(axis='both', which='minor', labelsize=fontsize)
ifilt=0
for order in range(num_cor):
    iBeta = order 
    label = r"$\beta"  + "^{" + str(order+1) + "}$"
    axs.plot(inv_lmdas, -Beta_as[:, iBeta], lw=2, label=label)
axs.legend(fontsize=fontsize)    

hd = config['hidden_size']
pz = config["patch_size"]
lr =config["lr"]
e = config["epochs"]

fig.tight_layout()

plt.show()

