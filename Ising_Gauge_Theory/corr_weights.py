import matplotlib.pyplot as plt
from Correlators import corr4,corr3,corr2
import numpy as np
from configs import config
from helper_functions import load_experiment
from opt_einsum import contract
from matplotlib.ticker import ScalarFormatter
from cycler import cycler

# plt.style.use('seaborn-v0_8-dark-palette')   
# 
from matplotlib import rcParams
fontsize=24
rcParams.update({'font.size': fontsize})
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['axes.prop_cycle'] = cycler(color=["#3071B7", "#B22222", "#E3B345", "#753589", "grey"])     


lis=[]
for i in range(64):
    op,acc=corr4(i)
    lis.append(op)

lis=np.array(lis)
op= lis.mean(0)
print("4th",op.shape)


from itertools import permutations, combinations
elements = [0,1,2,3,4,5,6,7]


perms = list(combinations(elements, 4))


corr=np.zeros((len(perms),1))

for i in range(len(perms)):
    index = list(permutations(perms[i]))
    for ind in index:
        corr[i]+=(op[ind])
    

        
fig, ax = plt.subplots(figsize=(10,6))

# Convert permutation tuples to string for labels
labels = [''.join(map(str, perm)) for perm in perms]

# Flatten 'corr' for plotting
values = abs(corr.flatten())


ax.bar(labels,values,color="#24305e")


# Set x-ticks to be the permutation labels, rotate for readability
plt.xticks(rotation=90)
plt.yticks()

plt.xlabel(r'$4^{th} ~\text{order index}$')

plt.ylabel('Weights')

plt.tight_layout()  # Adjust layout to make room for the rotated x-tick labels
d=config['hidden_size']
plt.show()

