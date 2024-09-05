import matplotlib.pyplot as plt
from Correlators import corr3
import numpy as np
from configs import config
from matplotlib.ticker import ScalarFormatter
from matplotlib import rcParams
from cycler import cycler
from itertools import permutations, combinations

fontsize=24
rcParams.update({'font.size': fontsize})
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['axes.prop_cycle'] = cycler(color=["#3071B7", "#B22222", "#E3B345", "#753589", "grey"])     
# op=0
lis=[]
for i in range(8):
    op,acc=corr3(i)
    lis.append(op)
# op,acc=corr3(7)
lis=np.array(lis)
print(lis.shape)
op= lis.mean(0)    

elements = [0, 1, 2, 3]

p=[(1, 1, 1),
 (1, 1, 2),
 (1, 1, 3),
 (1, 1, 0),
 (1, 2, 2),
 (1, 3, 3),
 (1, 0, 0),
 (2, 2, 2),
 (2, 2, 3),
 (2, 2, 0),
 (2, 3, 3),
 (2, 0, 0),
 (3, 3, 3),
 (3, 3, 0),
 (3, 0, 0),
 (0, 0, 0)]
perms = list(combinations(elements, 3))
# perms.extend(p)


corr=np.zeros((len(perms),1))

for i in range(len(perms)):
    index = list(permutations(perms[i]))
    for ind in index:
        corr[i]+=(op[ind])
    

        
fig, ax = plt.subplots(figsize=(8,6))


# Convert permutation tuples to string for labels
labels = [''.join(map(str, perm)) for perm in perms]

# Flatten 'corr' for plotting
values = abs(corr.flatten())

# Create bar plot
ax.bar(labels, values,color="#24305e")

# Set x-ticks to be the permutation labels, rotate for readability
plt.xticks(rotation=90,fontsize=fontsize)
plt.yticks(fontsize=20)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# Set labels and title
ax.yaxis.get_offset_text().set_fontsize(fontsize)

plt.xlabel(r'$3^{rd} ~\text{order index}$',fontsize=fontsize,labelpad=15)
plt.ylabel('Weights',fontsize=fontsize,labelpad=15)


plt.tight_layout()  # Adjust layout to make room for the rotated x-tick labels
d=config['hidden_size']
plt.show()



