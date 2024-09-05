import matplotlib.pyplot as plt
from Correlators import corr2, corr3, corr4
import numpy as np
from configs import config
from cycler import cycler
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter


fontsize=24
rcParams.update({'font.size': fontsize})
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['axes.prop_cycle'] = cycler(color=["#3071B7", "#B22222", "#E3B345", "#753589", "grey"])        
    

# op,acc=corr2(0)
lis=[]
for i in range(64):
    op,acc=corr2(i)
    lis.append(op)

op=np.array(lis).mean(0)

# np.save("2nd_order_W.npy",op)
# op = np.load("2nd_order_W.npy")
# plt.style.use('seaborn-v0_8-dark-palette')        

# plt.figure(figsize=(7,7))

fig, ax = plt.subplots()
im = ax.imshow(np.abs(op).transpose(-2,-1), cmap="RdBu")

print(np.abs(op).transpose(-2,-1)[0,0])
# Set ticks for both axes
ax.set_xticks([0, 1, 2, 3])
ax.set_yticks([0, 1, 2, 3])
# np.save("weights_stagg_2nd.npy",np.abs(op).transpose(-2,-1))
# Set tick labels if you want to customize them (optional)
ax.set_yticklabels(['3', '2', '1', '0'])
# ax.set_yticklabels(['0', '1', '2', '3'])

# plt.title(f"H_afm seed_acc_{acc}")
# plt.yticks([0,1,2,3],fontsize=fontsize)
# plt.xticks([3,2,1,0],fontsize=fontsize)
# plt.yticks(np.arange(4),fontsize=fontsize)
# plt.xticks(np.arange(4),fontsize=fontsize)
# plt.xlabel(r'$P_i$',fontsize=20)
# plt.ylabel(r'$P_i$',,fontsize=20)
plt.tight_layout()

# plt.colorbar()

cbar = plt.colorbar(im)

# Set font size for colorbar ticks
# cbar.ax.tick_params(labelsize=fontsize)
# plt.savefig("H_AFM\\corrweight.png",bbox_inches='tight')
plt.show()
# from itertools import permutations, combinations
# lis=[]
# for i in range(64):
#     op,acc=corr3(i)
#     lis.append(op)
# # op,acc=corr3(7)
# lis=np.array(lis)
# print(lis.shape)
# op= lis.mean(0)    

# elements = [0, 1, 2, 3]

# p=[(1, 1, 1),
#  (1, 1, 2),
#  (1, 1, 3),
#  (1, 1, 0),
#  (1, 2, 2),
#  (1, 3, 3),
#  (1, 0, 0),
#  (2, 2, 2),
#  (2, 2, 3),
#  (2, 2, 0),
#  (2, 3, 3),
#  (2, 0, 0),
#  (3, 3, 3),
#  (3, 3, 0),
#  (3, 0, 0),
#  (0, 0, 0)]
# perms = list(combinations(elements, 3))
# perms.extend(p)


# corr=np.zeros((len(perms),1))

# for i in range(len(perms)):
#     index = list(permutations(perms[i]))
#     for ind in index:
#         corr[i]+=(op[ind])
    

        
# fig, ax = plt.subplots(figsize=(8,6))


# # Convert permutation tuples to string for labels
# labels = [''.join(map(str, perm)) for perm in perms]

# # Flatten 'corr' for plotting
# values = abs(corr.flatten())

# # Create bar plot
# ax.bar(labels, values,color="#24305e")

# # Set x-ticks to be the permutation labels, rotate for readability
# plt.xticks(rotation=90,fontsize=fontsize)
# plt.yticks(fontsize=20)
# ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# # Set labels and title
# ax.yaxis.get_offset_text().set_fontsize(fontsize)
# # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(30))
# plt.xlabel(r'$3^{rd} ~\text{order index}$',fontsize=fontsize,labelpad=15)
# plt.ylabel('Weights',fontsize=fontsize,labelpad=15)
# # plt.title('Sum of 3rd order weights over 50 realizations')
# # plt.title(f'3rd order weights_seed_{s}_acc_{acc}')

# plt.tight_layout()  # Adjust layout to make room for the rotated x-tick labels
# d=config['hidden_size']
# plt.show()


# lis=[]
# for i in range(64):
#     op,acc=corr4(i)
#     lis.append(op)
# # op,acc=corr3(7)
# lis=np.array(lis)
# op= lis.mean(0)
# print("4th",op.shape)


# from itertools import permutations, combinations
# elements = [0,1,2,3]


# perms = list(combinations(elements, 4))
# # perms.extend(p)

# corr=np.zeros((len(perms),1))

# for i in range(len(perms)):
#     index = list(permutations(perms[i]))
#     for ind in index:
#         corr[i]+=(op[ind])
    

        
# fig, ax = plt.subplots(figsize=(10,6))

# # Convert permutation tuples to string for labels
# # labels = [''.join(map(str, perm)) for perm in perms]

# # # Flatten 'corr' for plotting
# # values = corr.flatten()

# # labels = np.load("Ising_Gauge_Theory\\ilgt_4th_order_labels.npy")
# # values = np.load("Ising_Gauge_Theory\\ilgt_4th_order_w.npy")

# ax.bar(labels,values,color="#24305e")

# # labels = np.save("Ising_Gauge_Theory\\ilgt_4th_order_labels.npy",np.array(labels))
# # values = np.save("Ising_Gauge_Theory\\ilgt_4th_order_w.npy",abs(values))
# # Create bar plot

# # Set x-ticks to be the permutation labels, rotate for readability
# plt.xticks(rotation=90)
# plt.yticks()

# # Set labels and title
# # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(30))
# plt.xlabel(r'$4^{th} ~\text{order index}$')

# plt.ylabel('Weights')
# # plt.title('Sum of 3rd order weights over 50 realizations')
# # plt.title(f'3rd order weights_seed_{s}_acc_{acc}')

# plt.tight_layout()  # Adjust layout to make room for the rotated x-tick labels
# d=config['hidden_size']
# plt.show()