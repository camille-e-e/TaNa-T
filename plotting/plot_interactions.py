"""
Purpose: make lists of interactions

Created: Fall 2023 by Camille Febvre
Last edited: Jan 16, 2024 by Camille Febvre
"""

import os
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import sys
sys.path.append('/home/cfebvre/repos/the_model_y3/plotting')
from test_classes import Species,State


#inputs
#outpath = './'
#outpath = '/home/cfebvre/out_venus/TTNM_outputs/SteadyT/single-TRC/varypmut-Tref303/' # end in '/'
# Path to outputs you want to process:
outpath = "/home/cfebvre/out_venus/TTNM_outputs/SteadyT/var-TRC/constpmut/"
# where to save npy files 
save_to = './model_outputs/var_TRC/constpmut'
multiple_dates = True
dates = ['Feb_05_24/'] #'Jun_21_23/','Dec_14_23/'] # must end in '/'
seed = 1000
seeds = np.r_[1000:1050]
seed_threshold = 1049 # largest number in seeds from first date
T = 301
temps = np.r_[274:320:3]
final_gen = 10000-1
save_stuff = True
sort_by_seed = False

# funcitons
def get_interactions(spc1,spc2,J_combo): #seed,outpath=outpath):
    found = 0
    Jran1,Jran2,Jran3 = J_combo
    interactions = spc1.ID^spc2.ID
    Jij = Jran1[interactions]*Jran2[interactions]*Jran3[spc2.ID]
    return Jij

def format_outputs(seed,T,outpath=outpath):
    T = f"{int(T)}"
    found = 0
    if os.path.exists(outpath+f"modelrun_seed{seed}_{T}K.npy"):
        modelrun = np.load(outpath+f"modelrun_seed{seed}_{T}K.npy",allow_pickle=True)
        all_spcs = np.load(outpath+f"species_objects_seed{seed}_{T}K.npy",allow_pickle=True)
        return modelrun, all_spcs
    else: 
        return [],[]

interactions_by_T = []
mean_interactions_by_T = []
core_int_by_T = []
mean_core_int_by_T = []
std_int_by_T = []
std_core_int_by_T = []
int_type_by_T = []
core_int_type_by_T = []
connectance_by_T = []
core_connectance_by_T = []
con_fig, con_ax = plt.subplots()
con_ax.set_xlabel("Temperature (K)")
con_ax.set_ylabel("Connectance")
#mut_this_T, comp_this_T, pred_this_T, oneway_pos_this_T, oneway_neg_this_T, none_this_T = 0,0,0,0,0,0
for T in temps:
    interactions_this_T = []
    mean_interactions_this_T = []
    core_int_this_T = []
    mean_core_int_this_T = []
    std_of_ecos_this_T = []
    std_of_cores_this_T = []
    connectance_this_T = []
    core_connectance_this_T = []
    if sort_by_seed:
        int_type_this_T = []
        core_int_type_this_T = []
    else:
        int_type_this_T = np.zeros((6,))
        core_int_type_this_T = np.zeros((6,))
    for seed in seeds: 
        # if there are multiple dates, check appropriate folder for this seed
        if multiple_dates: 
            if seed > seed_threshold:
                date = dates[1]
            else: date = dates[0]
            path_now = outpath+date
        else: path_now = outpath
        # load interactions file
        for f in os.listdir(path_now):
            if f.startswith(f"Js_combined{seed}"):
                break
        J_combo = np.load(path_now+f,allow_pickle=True)
        # load model run and species objects
        modelrun,all_spcs = format_outputs(seed,T,outpath=path_now)
        if len(modelrun) == 0:
            print("File not found: ",seed,T)
            continue
        spcs_at_t = [] # species objects for species that were alive at end
        interactions_at_t = [] # Jij values
        core_int_at_t = [] # J_i(core),j(core) values
        # interaction types
        mut_now, comp_now, pred_now, oneway_pos_now, oneway_neg_now, none_now = 0,0,0,0,0,0
        core_mut_now, core_comp_now, core_pred_now, core_oneway_pos_now, core_oneway_neg_now, core_none_now = 0,0,0,0,0,0
        n_core_spcs = 0 # count core species
        for spc in all_spcs:
            # only count final interactions
            if final_gen in spc.times_alive:
                spcs_at_t.append(spc)
                idx = list(spc.times_alive).index(final_gen)
                if spc.is_core[idx]:
                    n_core_spcs += 1
        #print("seed: ",seed,", T: ",T)
        #print("number of spcs alive at end: ",len(spcs_at_t))
        si = -1
        for spc1 in spcs_at_t:
            si += 1
            sj = -1
            for spc2 in spcs_at_t:
                sj += 1
                # no self-interaction
                if spc1 == spc2:
                    pass
                else:
                    # determine effect of j on i
                    Jij = get_interactions(spc1,spc2,J_combo) #seed,path_now)
                    interactions_at_t.append(Jij)
                    idx = list(spc1.times_alive).index(final_gen)
                    # if spc i is a core species, determine effect of j on i
                    if spc1.is_core[idx]:
                        core_int_at_t.append(Jij)
                        spc1_is_core = True
                    else: spc1_is_core = False
                    # determine mutual interaction type (symmetrical so don't repeat)
                    if sj > si:
                        # impact of i on j
                        Jji = get_interactions(spc2,spc1,J_combo) #seed,outpath=path_now)
                        if spc1_is_core:
                            idx2 = list(spc2.times_alive).index(final_gen)
                            if spc2.is_core[idx2]:
                                corecore = True
                            else: corecore = False
                        else: corecore = False
                        if Jij > 0 and Jji > 0:
                            mut_now += 1
                            if corecore:
                                core_mut_now += 1
                        elif Jij < 0 and Jij < 0:
                            comp_now += 1
                            if corecore:
                                core_comp_now += 1
                        elif Jij > 0 and Jji <0 or Jij < 0 and Jji > 0:
                            pred_now += 1
                            if corecore:
                                core_pred_now += 1
                        elif Jij == 0 and Jji > 0 or Jji == 0 and Jij > 0:
                            oneway_pos_now += 1
                            if corecore:
                                core_oneway_pos_now += 1
                        elif Jij == 0 and Jji < 0 or Jji == 0 and Jij < 0:
                            oneway_neg_now += 1
                            if corecore:
                                core_oneway_neg_now += 1
                        elif Jij == 0 and Jji == 0:
                            none_now += 1
                            if corecore:
                                core_none_now += 1
                        else: 
                            print("interaction not quantified")
                            print(Jij,Jji)
                    #dd

        if sort_by_seed:
            interactions_this_T.append(interactions_at_t)
            core_int_this_T.append(core_int_at_t)
            n_interactions = len(spcs_at_t)**2 - sum(range(1,len(spcs_at_t)))
            int_type_this_T.append(np.array([mut_now,comp_now,pred_now,oneway_pos_now,oneway_neg_now,none_now])/n_interactions)
            n_core_interactions = len(n_core_spcs)**2 - sum(range(1,n_core_spcs))
            core_int_type_this_T.append(np.array([core_mut_now,core_comp_now,core_pred_now,core_oneway_pos_now,core_oneway_neg_now,core_none_now])/n_core_interactions)
            connectance_this_T.append(1-none_now/n_interactions)
            core_connectance_this_T.append(1-none_now/n_core_interactions)
        else:
            interactions_this_T = list(np.concatenate([interactions_this_T,interactions_at_t]).flat)
            core_int_this_T = list(np.concatenate([core_int_this_T,core_int_at_t]).flat)
            mean_interactions_this_T.append(np.mean(interactions_at_t))
            mean_core_int_this_T.append(np.mean(core_int_at_t))
            std_of_ecos_this_T.append(np.std(interactions_at_t))
            std_of_cores_this_T.append(np.std(core_int_at_t))
            int_type_this_T += np.array([mut_now,comp_now,pred_now,oneway_pos_now,oneway_neg_now,none_now])
            core_int_type_this_T += np.array([core_mut_now,core_comp_now,core_pred_now,core_oneway_pos_now,core_oneway_neg_now,core_none_now])
            # number of unique combinitaitons of species pairs
            n_interactions = sum(range(1,len(spcs_at_t)))
            if n_interactions > 0:
                #connectance_this_T.append(1-none_now/n_interactions)
                connectance_this_T.append((n_interactions-none_now)/n_interactions)
            else: connectance_this_T.append(np.nan)
            n_core_interactions = sum(range(1,n_core_spcs))
            if n_core_interactions > 0:
                core_connectance_this_T.append((n_core_interactions-core_none_now)/n_core_interactions)
            else: core_connectance_this_T.append(np.nan)

    # np.save(f"interactions_{T}K.npy",interactions_this_T)
    # np.save(f"core_int_{T}K.npy",core_int_this_T)
    # np.save(f"interaction_types_{T}K.npy",int_type_this_T)
    # np.save(f"core_interaction_types_{T}K.npy",core_int_type_this_T)
    
    connectance_by_T.append(connectance_this_T)
    core_connectance_by_T.append(core_connectance_this_T)
    # scatter connectance
    con_ax.scatter(T,none_now)

    interactions_by_T.append(interactions_this_T)
    core_int_by_T.append(core_int_this_T)
    mean_interactions_by_T.append(mean_interactions_this_T)
    mean_core_int_by_T.append(mean_core_int_this_T)
    std_int_by_T.append(std_of_ecos_this_T)
    std_core_int_by_T.append(std_of_cores_this_T)
    int_type_by_T.append(int_type_this_T)
    core_int_type_by_T.append(core_int_type_this_T)

print("shape of int_type_by_T: ",np.shape(int_type_by_T))
print("shape of core_int_type_by_T: ",np.shape(core_int_type_by_T))


if save_stuff:
    np.save(f"{save_to}interactions_by_T.npy",interactions_by_T)
    np.save(f"{save_to}core_int_by_T.npy",core_int_by_T)
    np.save(f"{save_to}mean_interactions_by_T.npy",mean_interactions_by_T)
    np.save(f"{save_to}mean_core_int_by_T.npy",mean_core_int_by_T)
    np.save(f"{save_to}std_int_by_T.npy",std_int_by_T)
    np.save(f"{save_to}std_core_int_by_T.npy",std_core_int_by_T)
    np.save(f"{save_to}interaction_types_by_T.npy",int_type_by_T)
    np.save(f"{save_to}core_interaction_types_by_T.npy",core_int_type_by_T)

# histograms & boxplots of interactions
fig2,ax2 = plt.subplots(2) #,sharey=True)
ax2[0].boxplot(interactions_by_T,positions = temps,widths=2,showfliers=False)
ax2[1].boxplot(core_int_by_T,positions = temps,widths=2,showfliers=False)
for a in ax2:
    a.set_xlabel("Temperature (K)")
ax2[0].set_ylabel(r"Interactions, $J_\mathrm{ij}$")

fig,ax = plt.subplots(len(temps),2,sharex=True)
a1 = ax[:,0]
a2 = ax[:,1]
#f1,a1 = plt.subplots(len(temps),sharex=True) #,sharey=True) # all interactions
#f2,a2 = plt.subplots(len(temps),sharex=True) #,sharey=True) # core interactions
means_by_T = []
medians_by_T = []
mean_int_strength_by_T = []
median_int_strength_by_T = []
std_by_T = []
skew_by_T = []
core_means_by_T = []
core_medians_by_T = []
core_mean_int_strength_by_T = []
core_median_int_strength_by_T = []
core_std_by_T = []
core_skew_by_T = []
i = -1
for T in temps:
    i += 1
    # all interactions
    a1[i].hist(interactions_by_T[i],bins=50,stacked=True) #,density=True)
    a1[i].set_ylabel("count")
    a1[i].set_title(f"T: {T}")
    if sort_by_seed:
        combo_this_T = list(np.concatenate(interactions_by_T[i]).flat)
    else: combo_this_T = interactions_by_T[i]
    means_by_T.append(np.mean(combo_this_T))
    medians_by_T.append(np.median(combo_this_T))
    mean_int_strength_by_T.append(np.mean(np.abs(combo_this_T)))
    median_int_strength_by_T.append(np.median(np.abs(combo_this_T)))
    std_by_T.append(np.std(combo_this_T))
    skew_by_T.append(stats.skew(combo_this_T))

    # core intearctions
    a2[i].hist(core_int_by_T[i],bins=50,stacked=True) #,density=True)
    a2[i].set_ylabel("count")
    if sort_by_seed:
        combo_this_T = list(np.concatenate(core_int_by_T[i]).flat)
    else: combo_this_T = core_int_by_T[i]
    core_means_by_T.append(np.mean(combo_this_T))
    core_medians_by_T.append(np.median(combo_this_T))
    core_mean_int_strength_by_T.append(np.mean(np.abs(combo_this_T)))
    core_median_int_strength_by_T.append(np.median(np.abs(combo_this_T)))
    core_std_by_T.append(np.std(combo_this_T))
    core_skew_by_T.append(stats.skew(combo_this_T))

a1[-1].set_xlabel(r"J$_\mathrm{ij}$")
#if save_stuff:
#    f.savefig(f"Figures/interactions_by_T_{len(seeds)}seeds.pdf")

a2[-1].set_xlabel(r"core J$_\mathrm{ij}$")
if save_stuff:
    fig.savefig(f"figures/interactions_by_T_{len(seeds)}seeds.pdf")

fig,ax = plt.subplots(3,2,sharex=True,sharey='row')
# all
ax[0,0].set_title("all")
ax[0,0].scatter(temps,means_by_T)
ax[1,0].scatter(temps,std_by_T)
ax[2,0].scatter(temps,skew_by_T)
# core
ax[0,1].set_title("core")
ax[0,1].scatter(temps,core_means_by_T)
ax[1,1].scatter(temps,core_std_by_T)
ax[2,1].scatter(temps,core_skew_by_T)

for a in ax[2,:]:
    a.set_xlabel("Temperature")
ax[0,0].set_ylabel(r"mean J$_{ij}$")
ax[1,0].set_ylabel(r"std J$_{ij}$")
ax[2,0].set_ylabel(r"skew J$_{ij}$")


# scatter mean & median Jij, abs(Jij), and connectance
fig,ax = plt.subplots(3,2,sharex=True,sharey='row')
np.save(f"{save_to}connectance_by_T.npy",connectance_by_T)
temps_all = (np.ones(np.shape(connectance_by_T)).T*temps).T
# all
ax[0,0].set_title("all")
ax[0,0].scatter(temps,means_by_T,label="mean")
ax[0,0].scatter(temps,medians_by_T,label="median")
ax[0,0].legend()
ax[1,0].scatter(temps,mean_int_strength_by_T)
ax[1,0].scatter(temps,median_int_strength_by_T)
i = -1
for T in temps:
    i+= 1
    connectance_this_T = connectance_by_T[i]
    ax[2,0].scatter(T*np.ones(len(connectance_this_T)),connectance_this_T,c="k",s=5)
# core
ax[0,1].set_title("core")
ax[0,1].scatter(temps,core_means_by_T,label="mean")
ax[0,1].scatter(temps,core_medians_by_T,label="median")
ax[1,1].scatter(temps,core_mean_int_strength_by_T)
ax[1,1].scatter(temps,core_median_int_strength_by_T)
i = -1
for T in temps:
    i+= 1
    core_connectance_this_T = core_connectance_by_T[i]
    ax[2,1].scatter(T*np.ones(len(core_connectance_this_T)),core_connectance_this_T,c="k",s=5)

for a in ax[2,:]:
    a.set_xlabel("Temperature")
ax[0,0].set_ylabel(r"Interaction, J$_{ij}$")
ax[1,0].set_ylabel(r"Interaction strength, |J$_{ij}|$")
ax[2,0].set_ylabel(r"Connectance")
ax[2,0].set_ylim(0,1)

ax[0,0].set_title("All")
ax[0,1].set_title("Core")

if save_stuff:
    fig.savefig(f"figures/better_unique_interactions_by_T_{len(seeds)}seeds.pdf")

plt.show()

# interaction types
f,a = plt.subplots(len(temps),2,sharex=True)
ax = a[:,0] # all 
cax = a[:,1] # core
#cfig,cax = plt.subplots(len(temps))
#plt.bar(range(6),np.array([mut_now,comp_now,pred_now,oneway_pos_now,oneway_neg_now,none_now])/(len(spcs_at_t)**2-len(spcs_at_t)),tick_label=["mut","comp","pred/par","1-way +","1-way -","none"])
for row in range(len(temps)):
    T = temps[row]
    int_type_this_T = int_type_by_T[row]
    core_int_type_this_T = core_int_type_by_T[row]
    for i in range(len(seeds)):
        if sort_by_seed and i > 0:
            ax[row].bar(range(6),int_type_this_T[i],bottom=int_type_this_T[i-1],tick_label=["mut","comp","pred/par","1-way +","1-way -","none"])
            cax[row].bar(range(6),core_int_type_this_T[i],bottom=core_int_type_this_T[i-1],tick_label=["mut","comp","pred/par","1-way +","1-way -","none"])
        elif sort_by_seed:
            ax[row].bar(range(6),int_type_this_T[i],tick_label=["mut","comp","pred/par","1-way +","1-way -","none"])
            cax[row].bar(range(6),core_int_type_this_T[i],tick_label=["mut","comp","pred/par","1-way +","1-way -","none"])
        else:
            norm = sum(int_type_this_T)
            if norm > 0:
                ax[row].bar(range(6),int_type_this_T/norm,tick_label=["mut","comp","pred/par","1-way +","1-way -","none"])
            norm = sum(core_int_type_this_T)
            if norm > 0:
                cax[row].bar(range(6),core_int_type_this_T/norm,tick_label=["mut","comp","pred/par","1-way +","1-way -","none"])
    ax[row].set_ylabel(f"T: {T}") # ( "cumulative fraction")

    #cax[row].set_ylabel("cumulative fraction")
    
ax[0].set_title("All")
cax[0].set_title("Core")

if save_stuff:
    f.savefig(f"figures/int_type_by_T_{len(seeds)}seeds.pdf")
#fig.savefig(f"Figures/core_int_type_by_T_{len(seeds)}seeds.pdf")

if not sort_by_seed:
    # same as previous but without subplots
    fig,ax = plt.subplots(2,sharex=True)
    center = len(temps)/2
    offset = .06 #.1
    width = .06 #.1
    cmap = plt.get_cmap('inferno')
    colors = cmap(np.linspace(0,.9,len(temps)))
    for i in range(len(temps)):
        norm = sum(int_type_by_T[i])
        if norm > 0:
            ax[0].bar(np.arange(6)-(center-i)*offset, int_type_by_T[i]/norm, width=width, tick_label=["mut","comp","pred/par","1-way +","1-way -","none"], color=colors[i],label=temps[i])
        norm = sum(core_int_type_by_T[i])
        if norm > 0:
            ax[1].bar(np.arange(6)-(center-i)*offset, core_int_type_by_T[i]/norm, width=width, tick_label=["mut","comp","pred/par","1-way +","1-way -","none"], color=colors[i],label=temps[i])
    
    ax[0].set_ylabel("Fraction (all)")
    ax[1].set_ylabel("Fraction (core)")

    # adjust axes right to make space for legend
    plt.subplots_adjust(right=.8)
    ax[0].legend(bbox_to_anchor=[1,1])

plt.show()

                
