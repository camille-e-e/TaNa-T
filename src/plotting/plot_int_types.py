import numpy as np
import matplotlib.pyplot as plt

sin_width,doub_width = 3.54,7.48 # standard fig sizes in JTB

int_types = np.load("interaction_types_by_T.npy",allow_pickle=True)
core_int_types = np.load("core_interaction_types_by_T.npy",allow_pickle=True)

temps = np.r_[274:320:3]
int_names = ["mut","comp","pred/par","1way+","1way-","none"]

n_interactions,core_n_interactions = [],[]

# find total number of interactions at each temperature
for i in range(16):
    n_interactions.append(sum(int_types[i]))
    core_n_interactions.append(sum(core_int_types[i]))

# color maps
cmap = plt.get_cmap("Greys")
colors = cmap(np.linspace(.2,1,6))

# for each interaction type, plot its temperature dependence
fig,ax = plt.subplots(1,2,figsize=(doub_width,sin_width),sharey=True)
for i in range(6):
    ax[0].plot(temps,int_types.T[i]/n_interactions,color=colors[i],label=int_names[i])
    ax[1].plot(temps,core_int_types.T[i]/core_n_interactions,color=colors[i],label=int_names[i])

for a in ax:
    a.set_xlabel("Temperature (K)")

ax[0].set_ylabel("Fraction")
ax[1].legend(bbox_to_anchor=(1,1))

ax[0].set_title("All")
ax[1].set_title("Core")

plt.tight_layout()

plt.savefig("Figures/int_type_lineplots.pdf")

plt.show()




