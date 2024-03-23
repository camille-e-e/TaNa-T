import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import sys
sys.path.append('/home/cfebvre/repos/the_model_y3/plotting')
sys.path.append('/home/cfebvre/repos/the_model_y3/src/geoTNM')
from test_classes import Species,State
from MTE_TPC_combo import poff_T,pdeath
"""
The purpose of this script is to produce a plot of interactions in ecosystems and cores, in addition to interaction type fractions, to be used in the TTNM manuscript.

This script uses npy files output from plot_interactions.py.  plot_interactions.py must have been run for both single_TRC and var_TRC experiments.  Write the locations of the npy_files in the first lines of code below.  

NOTE that I have modified the script to combine the outputs from two separate runs of single and var-TRC experiments, in order to increase the samples size.  Consequently, there are single_int_types1 and single_int_types2, etc, with 1 and 2 for all the first and second batch of outputs.  These are then combined into single_int_types (etc) below.

Created: Fall 2023 by Camille Febvre
Last modified: Feb 05, 2024 by Camille Febvre
"""

# plotting options
combine_inputs = False # combine inputs from different files
boxplots = False #True # boxplots of interactions? (otherwise plot mean, 10th and 90th percentiles)
plot_prediction = True #False # predict that interactions are related to poff/pdeath?

# figure sizes
sin_width,doub_width = 3.54,7.48 # standard fig sizes in JTB

# single TRC
single_int_types1 = np.load("model_outputs/interaction_types_by_T_single_Tref303.npy",allow_pickle=True)
single_int_types = np.load("model_outputs/single_TRC/interaction_types_by_T.npy",allow_pickle=True)
single_core_int_types1 = np.load("model_outputs/core_interaction_types_by_T_single_Tref303.npy",allow_pickle=True)
single_core_int_types = np.load("model_outputs/single_TRC/core_interaction_types_by_T.npy",allow_pickle=True)
single_interactions_by_T1 = np.load("model_outputs/interactions_by_T_single_Tref303.npy",allow_pickle=True)
single_interactions_by_T = np.load("model_outputs/single_TRC/interactions_by_T.npy",allow_pickle=True)
single_core_int_by_T1 = np.load("model_outputs/core_int_by_T_single_Tref303.npy",allow_pickle=True)
single_core_int_by_T = np.load("model_outputs/single_TRC/core_int_by_T.npy",allow_pickle=True)

# var-TRC
var_int_types1 = np.load("model_outputs/interaction_types_by_T_var.npy",allow_pickle=True)
var_int_types = np.load("model_outputs/var_TRC/interaction_types_by_T.npy",allow_pickle=True)
var_core_int_types1 = np.load("model_outputs/core_interaction_types_by_T_var.npy",allow_pickle=True)
var_core_int_types = np.load("model_outputs/var_TRC/core_interaction_types_by_T.npy",allow_pickle=True)
var_interactions_by_T1 = np.load("model_outputs/interactions_by_T_var.npy",allow_pickle=True)
var_interactions_by_T = np.load("model_outputs/var_TRC/interactions_by_T.npy",allow_pickle=True)
var_core_int_by_T1 = np.load("model_outputs/core_int_by_T_var.npy",allow_pickle=True)
var_core_int_by_T = np.load("model_outputs/var_TRC/core_int_by_T.npy",allow_pickle=True)

temps = np.r_[274:320:3]
int_names = ["Mut","Comp","Pred","1way+","1way-","None"]

if combine_inputs:
    # combine outputs from different dates
    single_int_types,var_int_types = [],[]
    single_core_int_types,var_core_int_types = [],[]
    single_interactions_by_T,var_interactions_by_T = [],[]
    single_core_int_by_T,var_core_int_by_T = [],[]
    for i in range(len(temps)):
        # total number of interactions of each category
        single_int_types.append(single_int_types1[i]+single_int_types2[i])
        var_int_types.append(var_int_types1[i]+var_int_types2[i])
        single_core_int_types.append(single_core_int_types1[i]+single_core_int_types2[i])
        var_core_int_types.append(var_core_int_types1[i]+var_core_int_types2[i])
        # lists of interactions
        single_interactions_by_T.append(np.hstack([single_interactions_by_T1[i],single_interactions_by_T2[i]]))
        single_core_int_by_T.append(np.hstack([single_core_int_by_T1[i],single_core_int_by_T2[i]]))
        var_interactions_by_T.append(np.hstack([var_interactions_by_T1[i],var_interactions_by_T2[i]]))
        var_core_int_by_T.append(np.hstack([var_core_int_by_T1[i],var_core_int_by_T2[i]]))


# color maps
cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0,.9,6))

# make plot with mean interaction, mean interaction strength, and interaction types
fig,ax = plt.subplots(2,4,figsize=(1.25*doub_width,.6*doub_width),sharex=True,sharey='row')
T_range = np.r_[274:320]

for a in ax[0,:]:
    a.plot(T_range-1,np.zeros((len(T_range),)),"k",linewidth=1)

if plot_prediction:
    prediction = "death" # "po/pd"
    if prediction == "po/pd":
        # predict how poff varies with temperature in var-TRC
        poff_var = np.ones(len(T_range)) #linspace(.1,1,len(T_range))
        # or prescribe a po/pd ratio
        popd_ratio_var = np.linspace(1,5,len(T_range))
    i = -1
    for aa in [ax[0,1],ax[0,3]]:
        a = aa.twinx()
        if aa == ax[0,3]:
            a.set_ylabel("Fitness threshold",color="r")
        a.tick_params(axis='y', colors='red')
        i+= 1
        if prediction == "po/pd": #00:
            if i == 0: # single TRC
                a.plot(T_range,-np.log((poff_T(T_range)/pdeath(T_range))-1),color="r")
            elif i == 1: # var-TRC
                #a.plot(T_range,-np.log(poff_var/pdeath(T_range)-1),color="r")
                a.plot(T_range,-np.log(popd_ratio_var-1),color="r")
        else: # prediction = death
            aa.plot(T_range,.5/pdeath(T_range),color="r")
        #a.set_ylim(-30,100)
#ax[0,4].remove()
#ax[1,4].remove()
ax[0,0].set_title("Single TRC, all")
ax[0,1].set_title("Single TRC, core")
ax[0,2].set_title("Various TRC, all")
ax[0,3].set_title("Various TRC, core")

ax[0,0].set_ylabel(r"Interaction, J$_\mathrm{ij}$")
#ax[1,0].set_ylabel(r"Interaction strength, |J$_\mathrm{ij}$|")
ax[-1,0].set_ylabel("Fraction")
for a in ax[-1,:4]:
    a.set_xlabel("Temperature, T (K)")

fig2,ax2 = plt.subplots(1,4)
for a in ax2:
    a.set_xlim(-200,200)
    a.set_ylabel("PDF")
for a in ax2[0:2]:
    a.set_title("Single TRC")
for a in ax2[2:]:
    a.set_title("Various TRC")
for a in [ax2[0],ax2[2]]:
    a.set_xlabel("Interactions")
for a in [ax2[1],ax2[3]]:
    a.set_xlabel("Core interactions")
cmap = plt.get_cmap("inferno")
colors2 = cmap(np.linspace(0,.9,len(temps)))

fig3,ax3 = plt.subplots(2,2)
for a in ax3[1,:]:
    a.set_xlabel("Temperature, 1/kT")
ax3[0,0].set_ylabel("Ecosystem")
ax3[1,0].set_ylabel("Core")
ax3[0,0].set_title("Single TRC")
ax3[0,1].set_title("Various TRC")
fig3.suptitle(r"log($J_\mathrm{ij}$)")

# single TRC and various TRC
for exp in range(2):
    if exp == 0: # single TRC
        int_types = np.array(single_int_types)
        core_int_types = np.array(single_core_int_types)
        interactions_by_T = np.array(single_interactions_by_T)
        core_int_by_T = np.array(single_core_int_by_T)
    else: # various TRC
        int_types = np.array(var_int_types)
        core_int_types = np.array(var_core_int_types)
        interactions_by_T = np.array(var_interactions_by_T)
        core_int_by_T = np.array(var_core_int_by_T)
    # calculate mean and median interactions
    means_by_T = []
    medians_by_T = []
    q1_by_T = []
    q3_by_T = []
    mean_int_strength_by_T = []
    median_int_strength_by_T = []
    std_by_T = []
    skew_by_T = []
    core_means_by_T = []
    core_medians_by_T = []
    core_q1_by_T = []
    core_q3_by_T = []
    core_mean_int_strength_by_T = []
    core_median_int_strength_by_T = []
    core_std_by_T = []
    core_skew_by_T = []
    # filter out too many zeros
    filtered_ints = []
    filtered_ints_core = []

    i = -1
    for T in temps:
        i += 1
        # all interactions
        combo_this_T = interactions_by_T[i]
        if boxplots:
            # filter out the interactions too close to zero (too many of them)
            ints_now = np.array(interactions_by_T[i])
            idx1 = ints_now > 1
            idx2 = ints_now < -1
            idx = idx1+idx2
            filtered_ints.append(ints_now[idx])
            # Core
            ints_now = np.array(core_int_by_T[i])
            idx1 = ints_now > 1
            idx2 = ints_now < -1
            idx = idx1+idx2
            filtered_ints_core.append(ints_now[idx])
        ax2[2*exp].hist(combo_this_T,50,histtype='step',color=colors2[i],alpha=.5,density=True) #,bins=bins)
        means_by_T.append(np.mean(combo_this_T))
        medians_by_T.append(np.median(combo_this_T))
        q1_by_T.append(np.nanquantile(combo_this_T,.1))
        q3_by_T.append(np.nanquantile(combo_this_T,.9))
        mean_int_strength_by_T.append(np.mean(np.abs(combo_this_T)))
        median_int_strength_by_T.append(np.median(np.abs(combo_this_T)))
        std_by_T.append(np.std(combo_this_T))
        skew_by_T.append(stats.skew(combo_this_T))

        # core intearctions
        combo_this_T = core_int_by_T[i]
        ax2[2*exp+1].hist(combo_this_T,50,histtype='step',color=colors2[i],alpha=.5,density=True) #,bins=bins)
#        ax2[2*exp+1].plot(np.mean(combo_this_T)*np.ones((2,)),[0,.1],linestyle="--",color=colors2[i])
        core_means_by_T.append(np.mean(combo_this_T))
        core_medians_by_T.append(np.median(combo_this_T))
        core_q1_by_T.append(np.nanquantile(combo_this_T,.1))
        core_q3_by_T.append(np.nanquantile(combo_this_T,.9))
        core_mean_int_strength_by_T.append(np.mean(np.abs(combo_this_T)))
        core_median_int_strength_by_T.append(np.median(np.abs(combo_this_T)))
        core_std_by_T.append(np.std(combo_this_T))
        core_skew_by_T.append(stats.skew(combo_this_T))

    print("*****************\nq1_by_T: ",q1_by_T)
    print("*****************\nq3_by_T: ",q3_by_T)
    means_by_T = np.array(means_by_T)
    core_means_by_T = np.array(core_means_by_T)
    q1_by_T = np.array(q1_by_T)
    q3_by_T = np.array(q3_by_T)
    core_q1_by_T = np.array(core_q1_by_T)
    core_q3_by_T = np.array(core_q3_by_T)

    n_interactions,core_n_interactions = [],[]

    # find total number of interactions at each temperature
    for i in range(16):
        n_interactions.append(sum(int_types[i]))
        core_n_interactions.append(sum(core_int_types[i]))

    col = 2*exp
    ## interaction mean and interquartile range (median always zero)
    ## all
    #ax[0,col].fill_between(temps,q1_by_T,q3_by_T,alpha=.5,label="quartiles")
    #plt.figure()
    #plt.fill_between(temps,q1_by_T,q3_by_T,label="quartiles")
    #plt.plot(temps,q1_by_T)
    #plt.plot(temps,q3_by_T)
    #plt.plot(temps,means_by_T,":")

    #ax[0,col].scatter(temps,means_by_T,marker="x",label="mean")
    #ax[0,col].errorbar(temps,means_by_T,yerr=np.array([[means_by_T-q1_by_T],[means_by_T+q3_by_T]]).T) #np.array(std_by_T)/2)

    # Arrhenius plots
    k = 8.6e-5 #eV
    # plot metabolic rate
    def expectation(x_range=np.r_[-36,42],m=0.49,b=10):
        return b + m*x_range
    x_range = np.r_[36,43]
    for a in ax3.flatten():
        if a in ax3[1,:]:
            line = expectation(x_range,m=0.49,b=-17.5) #-21
        else:
            line = expectation(x_range,m=0.49,b=-20) #-21
        a.plot(x_range,line,"r--")
        #a.set_ylim(min(line),max(line))

    # convert to logs
    def convert_to_logs(interactions):
        idx_pos = interactions > 0
        idx_neg = interactions < 0
        log_pos_int = np.log(interactions[idx_pos])
        log_neg_int = -np.log(-interactions[idx_neg])
        log_interactions = np.zeros((len(interactions),))
        log_interactions[idx_pos] = log_pos_int
        log_interactions[idx_neg] = log_neg_int
        return log_interactions

    ax3[0,exp].plot(1/k/temps, np.log(means_by_T),'k')
    ax3[0,exp].fill_between(1/k/temps,convert_to_logs(q1_by_T),convert_to_logs(q3_by_T),alpha=.25) #np.array(std_by_T)/2)
    ax3[1,exp].plot(1/k/temps, np.log(core_means_by_T),'k')
    ax3[1,exp].fill_between(1/k/temps,convert_to_logs(core_q1_by_T),convert_to_logs(core_q3_by_T),alpha=.25) #np.array(std_by_T)/2)

    # Normal plots
    if boxplots:
        ax[0,col].boxplot(filtered_ints,positions=temps,showmeans=True,meanline=True,showfliers=False,widths=2)
        ax[0,col+1].boxplot(filtered_ints_core,positions=temps,showmeans=True,meanline=True,showfliers=False,widths=2)
        #ax[0,col+1].get_xaxis().set_visible(False)
        #ax[0,col].get_xaxis().set_visible(False)
        ax[1,col].tick_params(axis='x', labelrotation=90)
        ax[1,col+1].tick_params(axis='x', labelrotation=90)
    else:
        ax[0,col].vlines(temps,q1_by_T,q3_by_T,"k") #np.array(std_by_T)/2)
        ax[0,col].plot(temps,means_by_T,'ko')
        ax[0,col].plot(temps,np.zeros((len(temps),)),"k",linewidth=.2)

    #ax[0,col].plot(temps,q1_by_T,":")
    #ax[0,col].plot(temps,q3_by_T,":")
    #ax[0,col].scatter(temps,medians_by_T,marker="+",label="median")
    #ax[1,col].scatter(temps,mean_int_strength_by_T,marker="x")#std_by_T)
    #ax[1,col].scatter(temps,median_int_strength_by_T,marker="+")#std_by_T)
    #ax[2,0].scatter(temps,skew_by_T)
    
    ## core
    #ax[0,col+1].fill_between(temps,core_q1_by_T,core_q3_by_T,alpha=.5,label="interquartile")
    #ax[0,col+1].scatter(temps,core_means_by_T,marker="x",label="mean")
    #ax[0,col+1].errorbar(temps,core_means_by_T,yerr=([[core_means_by_T-core_q1_by_T],[core_means_by_T+core_q3_by_T]]).T) #np.array(core_std_by_T)/2)
        ax[0,col+1].vlines(temps,core_q1_by_T,core_q3_by_T,"k") #np.array(core_std_by_T)/2)
        ax[0,col+1].plot(temps,core_means_by_T,'ko')
        ax[0,col+1].plot(temps,np.zeros((len(temps),)),"k",linewidth=.2)
    #ax[0,col+1].plot(temps,core_q1_by_T,":")
    #ax[0,col+1].plot(temps,core_q3_by_T,":")
    #ax[0,col+1].scatter(temps,core_medians_by_T,marker="+",label="median")
    # ax[0,0].legend() #bbox_to_anchor=(1,1))
    
    ## interaction strength
    #ax[1,col+1].scatter(temps,core_mean_int_strength_by_T,marker="x",label="mean") #core_std_by_T)
    #ax[1,col+1].scatter(temps,core_median_int_strength_by_T,marker="+",label="median") #core_std_by_T)
    #ax[1,0].legend() #bbox_to_anchor=(1.2,1))
    #ax[2,1].scatter(temps,core_skew_by_T)

    ## for each interaction type, plot its temperature dependence
    #fig,ax = plt.subplots(1,2,figsize=(doub_width,sin_width),sharey=True)
    for i in range(6):
        if i%3 == 0:
            linestyle="--"
        elif i%3 == 1:
            linestyle="-."
        else: linestyle=":"
        ax[-1,col].plot(temps,int_types.T[i]/n_interactions,color=colors[i],label=int_names[i],linestyle=linestyle)
        ax[-1,col+1].plot(temps,core_int_types.T[i]/core_n_interactions,color=colors[i],label=int_names[i],linestyle=linestyle)

    ax[-1,-1].legend(bbox_to_anchor=(1,1),fontsize=9) #bbox_to_anchor=(1,1),borderaxespad=0.)

    plt.tight_layout()

    letters = iter(["a)","b)","c)","d)","e)","f)","g)","h)","i)","j)","k)","l)"])
    for a in ax.flatten():
        letter = next(letters)
        a.text(0.05,0.9,letter,transform=a.transAxes,horizontalalignment="left",verticalalignment="center")

fig.savefig("figures/combined_interaction_plots.pdf")

plt.show()




