"""
Purpose: 
- re-make interaction plots to see if I get similar answers
to the predictions from plot_Topt_poff_fi_etc.py.
- plot distribution of ecosystem mean interactions
- plot distribution of stds

Made Feb 1, 2024 by Camille Febvre
Last modified Feb 6, 2024 by Camille Febvre
"""

import numpy as np
import matplotlib.pyplot as plt

experiments = ["single_TRC/constpmut","var_TRC/constpmut","single_TRC","var_TRC"]

temps = np.r_[274:320:3]
k = 8.6e-5 #eV

# figure sizes for JTB
sin_width = 3.54 # single column
doub_width = 7.48 # double column

fig,ax = plt.subplots(2,len(experiments),sharey=True,sharex=True)
#ax[0,0].set_title("Single TRC")
#ax[0,1].set_title("Various TRC")
for i in range(len(experiments)):
    ax[0,i].set_title(experiments[i])
ax[0,0].set_ylabel("All interactions")
ax[1,0].set_ylabel("Core interactions")
for a in ax[1,:]:
    a.set_xlabel("Temperature (K)")
    a.tick_params(axis='x', labelrotation=90)
for a in ax[:].flatten():
    a.plot(temps,np.zeros((len(temps),)),"k",linewidth=1)

fig2,ax2 = plt.subplots(2,len(experiments),sharey=True,sharex=True)
#ax2[0,0].set_title("Single TRC")
#ax2[0,1].set_title("Various TRC")
for i in range(len(experiments)):
    ax2[0,i].set_title(experiments[i])
ax2[0,0].set_ylabel("Mean interactions")
ax2[1,0].set_ylabel("Mean core interactions")
for a in ax2[1,:]:
    a.set_xlabel("Temperature (K)")
    a.tick_params(axis='x', labelrotation=90)
for a in ax2[:].flatten():
    a.plot(temps,np.zeros((len(temps),)),"k",linewidth=1)

# standard deviations
fig3,ax3 = plt.subplots(2,len(experiments),sharey=True,sharex=True)
#ax2[0,0].set_title("Single TRC")
#ax2[0,1].set_title("Various TRC")
for i in range(len(experiments)):
    ax3[0,i].set_title(experiments[i])
ax3[0,0].set_ylabel("Std of mean interactions")
ax3[1,0].set_ylabel("Std of mean core interactions")
for a in ax3[1,:]:
    a.set_xlabel("Temperature (K)")
    a.tick_params(axis='x', labelrotation=90)
for a in ax3[:].flatten():
    a.plot(temps,np.zeros((len(temps),)),"k",linewidth=1)

# scatter mean vs std for each T
fig4,ax4 = plt.subplots(4,4,sharey=True,sharex=True)
i = -1
for a in ax4.flatten():
    i += 1
    a.set_title(f"{temps[i]}K")
for a in ax4[-1,:]:
    a.set_xlabel("Mean interaction")
for a in ax4[:,0]:
    a.set_ylabel("Std.")

# figure for MS: dist. of mean ints w/ & w/o const pmut + Arrhenius plots w/ & w/o
fig5,ax5 = plt.subplots(2,4,sharey='row',sharex='row',figsize=(doub_width,.6*doub_width))
ax5[0,0].set_title("Single-TRC; all")
ax5[0,1].set_title("Single-TRC; core")
ax5[0,2].set_title("Various-TRC; all")
ax5[0,3].set_title("Various-TRC; core")
ax5[0,0].set_ylabel(r"Mean $J_\mathrm{ij}$") 
#ax5[1,0].set_ylabel(r"Mean $J_\mathrm{ij}$, $p_\mathrm{mut}$=.01")
ax5[1,0].set_ylabel(r"log(mean $J_\mathrm{ij}$)")
for a in ax5[0,:]:
    a.set_xlabel("Temperature (K)")
for a in ax5[-1,:]:
    a.set_xlabel("Temperature, 1/kT")
for a in ax5[0,:]: #.flatten():
    a.plot(temps-1,np.zeros((len(temps),)),"k",linewidth=.5)
def expectation(x_range=np.r_[-36,42],m=0.49,b=10):
    return b + m*x_range
x_range = np.r_[36,43]
for a in ax5[-1,:]: #.flatten():
    line = expectation(x_range,m=0.49,b=-17.5) #-21
    a.plot(x_range,line,"b--")
    line = expectation(x_range,m=0.49,b=-20) #-21
    a.plot(x_range,line,"b--")

#T_labels = []
#i = 0
#for T in temps:
#    i = not i
#    if i:
#        T_labels.append(str(T))
#for a in ax[1,:]:
#    a.set_xticklabels(T_labels)

col = -1
for experiment in experiments:
    not_yet_labeled=True
    print("Experiment ****** ",experiment)
    col += 1
    # full distributions
    int_by_T = np.load(f"model_outputs/{experiment}/interactions_by_T.npy",allow_pickle=True)
    core_int_by_T = np.load(f"model_outputs/{experiment}/core_int_by_T.npy",allow_pickle=True)
    # distribution of means
    try:
        mean_int_by_T = np.load(f"model_outputs/{experiment}/mean_interactions_by_T.npy",allow_pickle=True)
        mean_core_int_by_T = np.load(f"model_outputs/{experiment}/mean_core_int_by_T.npy",allow_pickle=True)
        std_of_ecos_by_T = np.load(f"model_outputs/{experiment}/std_mean_int_by_T.npy",allow_pickle=True)
        std_of_cores_by_T = np.load(f"model_outputs/{experiment}/std_mean_core_int_by_T.npy",allow_pickle=True)
        print("Shape of std files: ",np.shape(std_of_ecos_by_T))
        plot_means = True
    except: 
        plot_means = False
    
    n_seeds = len(mean_int_by_T[0])
    # filter out too many zeros
    filtered_ints = []
    filtered_ints_core = []
    filtered_mean_ints = [] #np.zeros((len(temps),n_seeds))
    filtered_mean_core_ints = [] # np.zeros((len(temps),n_seeds))

    for i in range(len(temps)):
        # Ecosystem
        ints_now = np.array(int_by_T[i])
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
        if plot_means:
            print("Shape of mean_ints: ",np.shape(mean_int_by_T))
            # Ecosystem mean
            ints_now = np.array(mean_int_by_T[i])
            idx1 = ints_now > 1
            idx2 = ints_now < -1
            idx = idx1+idx2
            filtered_mean_ints.append(ints_now[idx])
            # Core
            ints_now = np.array(mean_core_int_by_T[i])
            idx1 = ints_now > 1
            idx2 = ints_now < -1
            idx = idx1+idx2
            filtered_mean_core_ints.append(ints_now[idx])
        
    print("shape and type of filtered_mean_imts: ",type(filtered_mean_ints),np.shape(filtered_mean_ints))

    ax[0,col].boxplot(filtered_ints,positions=temps,widths=2,showmeans=True,showfliers=False)
    ax[1,col].boxplot(filtered_ints_core,positions=temps,widths=2,showmeans=True,showfliers=False)
    if plot_means: 
        print("Plotting means and std")
        # distribution of means
        print("Shape of filtered_mean_ints: ",np.shape(filtered_mean_ints))
        ax2[0,col].boxplot(filtered_mean_ints,positions=temps,widths=2,showmeans=True,showfliers=False)
        ax2[1,col].boxplot(filtered_mean_core_ints,positions=temps,widths=2,showmeans=True,showfliers=False)
 
        mean_mean_ints = []
        mean_core_mean_ints = []
        q1_mean_ints, q3_mean_ints = [],[]
        q1_core_mean_ints, q3_core_mean_ints = [],[]
        std_mean_ints, std_core_mean_ints = [],[]
        if 1 ==1:
            for i in range(len(temps)):

                if "constpmut" in experiment: # == "single_TRC": # in experiment: 
                    r = 0
                    color = "r"
                    label = r"$p_\mathrm{mut}$=0.01"
                else: 
                    r = 0
                    color = "k"
                    label = r"$p_\mathrm{mut}$(T)"
                if "single_TRC" in experiment:
                    c = 0
                else: c = 2
                if not_yet_labeled: # and col < 2:
                    ax5[r,c].scatter(temps[i], np.mean(filtered_mean_ints[i]),color=color,label=label)
                    not_yet_labeled=False
                else:
                    ax5[r,c].scatter(temps[i], np.mean(filtered_mean_ints[i]),color=color)
                ax5[r,c].vlines(temps[i], np.nanquantile(filtered_mean_ints[i],.1), np.nanquantile(filtered_mean_ints[i],.9) ,color=color)
                ax5[r,c+1].scatter(temps[i], np.mean(filtered_mean_core_ints[i]),color=color)
                ax5[r,c+1].vlines(temps[i], np.nanquantile(filtered_mean_core_ints[i],.1), np.nanquantile(filtered_mean_core_ints[i],.9) ,color=color)

                # Assemble lists of logs of means and quantils
                # Need to deal with negative numbers separately 
                # ECOSYSTEMS
                std_not_quantiles = True
                mean_this_T = np.nanmean(mean_int_by_T[i])
                std_mean_ints.append(np.log(np.nanstd(mean_int_by_T[i])))
                if mean_this_T > 0:
                    mean_mean_ints.append(np.log(mean_this_T)) #np.nanmean(mean_int_by_T[i])))
                else:
                    mean_mean_ints.append(-np.log(-mean_this_T)) #np.nanmean(mean_int_by_T[i])))
                if std_not_quantiles:
                    q1_this_T = mean_this_T - np.nanstd(mean_int_by_T[i])
                else:
                    q1_this_T = np.nanquantile(mean_int_by_T[i],.1)
                if q1_this_T > 0:
                    q1_mean_ints.append(np.log(q1_this_T)) #np.nanquantile(mean_int_by_T[i],.1)))
                else:
                    q1_mean_ints.append(-np.log(-q1_this_T)) #np.nanquantile(mean_int_by_T[i],.1)))
                if std_not_quantiles:
                    q3_this_T = mean_this_T + np.nanstd(mean_int_by_T[i])
                else:
                    q3_this_T = np.nanquantile(mean_int_by_T[i],.9)
                if q3_this_T > 0:
                    q3_mean_ints.append(np.log(q3_this_T)) #np.nanquantile(mean_int_by_T[i],.1)))
                else:
                    q3_mean_ints.append(-np.log(-q3_this_T)) #np.nanquantile(mean_int_by_T[i],.1)))
                # CORES    
                mean_this_T = np.nanmean(mean_core_int_by_T[i])
                std_core_mean_ints.append(np.log(np.nanstd(mean_core_int_by_T[i])))
                if mean_this_T > 0:
                    mean_core_mean_ints.append(np.log(mean_this_T)) #np.nanmean(mean_int_by_T[i])))
                else:
                    mean_core_mean_ints.append(-np.log(-mean_this_T)) #np.nanmean(mean_int_by_T[i])))
                if std_not_quantiles:
                    q1_this_T = mean_this_T - np.nanstd(mean_core_int_by_T[i])
                else:
                    q1_this_T = np.nanquantile(mean_core_int_by_T[i],.1)
                if q1_this_T > 0:
                    q1_core_mean_ints.append(np.log(q1_this_T)) #np.nanquantile(mean_int_by_T[i],.1)))
                else:
                    q1_core_mean_ints.append(-np.log(-q1_this_T)) #np.nanquantile(mean_int_by_T[i],.1)))
                if std_not_quantiles:
                    q3_this_T = mean_this_T + np.nanstd(mean_core_int_by_T[i])
                    print("T ***** ", temps[i])
                    print(f"Q1: {q1_this_T}, Mean: {mean_this_T}, Q3: {q3_this_T}")
                    print(q3_this_T > mean_this_T and mean_this_T > q1_this_T)
                else:
                    q3_this_T = np.nanquantile(mean_core_int_by_T[i],.9)
                if q3_this_T > 0:
                    q3_core_mean_ints.append(np.log(q3_this_T)) #np.nanquantile(mean_int_by_T[i],.1)))
                else:
                    q3_core_mean_ints.append(-np.log(-q3_this_T)) #np.nanquantile(mean_int_by_T[i],.1)))
                print(q3_core_mean_ints[-1] > mean_core_mean_ints[-1] and mean_core_mean_ints[-1] > q1_core_mean_ints[-1])
                
                    

                # find non-nan indices for stds
                T = temps[i]
                valid_idx = 0
                idx = ~np.isnan(std_of_ecos_by_T[i])
                if sum(idx) != 0:
                    #print("indices: ",idx)
                    std_ints_now = np.array(std_of_ecos_by_T[i])[idx]
                    valid_idx += 1
                idx = ~np.isnan(std_of_cores_by_T[i])
                if sum(idx) != 0:
                    std_core_ints_now = np.array(std_of_cores_by_T[i])[idx]
                    valid_idx += 1
                if valid_idx == 2:
                    ax3[0,col].boxplot(std_ints_now,positions=[T],widths=2,showmeans=True,showfliers=False)
                    ax3[1,col].boxplot(std_core_ints_now,positions=[T],widths=2,showmeans=True,showfliers=False)
        #except: pass
        
        # Arrhenius plots
        if "single_TRC" in experiment: 
            c = 0
            label = r"p_\mathrm{mut}(T)"
        else: 
            c = 2
            label = r"p_\mathrm{mut}=0.01"
        if "constpmut" in experiment: 
            color = "r"
        else: color = "k"
        ax5[-1,c].plot(1/k/temps, mean_mean_ints,color=color,label=label)
        ax5[-1,c+1].plot(1/k/temps, mean_core_mean_ints,color=color)
        if std_not_quantiles:
            ax5[-1,c].fill_between(1/k/temps, np.array(mean_mean_ints)-np.array(std_mean_ints), np.array(mean_mean_ints)+np.array(std_mean_ints),color=color,alpha=.25 ) #q1_mean_ints, q3_mean_ints ,color=color,alpha=.25)
            ax5[-1,c+1].fill_between(1/k/temps, np.array(mean_core_mean_ints)-np.array(std_core_mean_ints), np.array(mean_core_mean_ints)+np.array(std_core_mean_ints), color=color, alpha=.25 ) # q1_core_mean_ints, q3_core_mean_ints ,color=color, alpha=.25)
        else:
            ax5[-1,c].fill_between(1/k/temps, np.array(q1_mean_ints), np.array(q3_mean_ints) ,color=color,alpha=.25)
            ax5[-1,c+1].fill_between(1/k/temps, np.array( q1_core_mean_ints), np.array(q3_core_mean_ints) ,color=color, alpha=.25)
        

        # scatter means and stds
        r,c = 0,-1
        i = -1
        for T in temps:
            i += 1
            c += 1
            if c == 4:
                c = 0
                r += 1
            ax4[r,c].scatter(mean_core_int_by_T[i],std_of_cores_by_T[i],s=5)


    else: print("Skipping means and stds: ",experiment)

letters = iter(["a)","b)","c)","d)","e)","f)","g)","h)","i)","j)","k)","l)"])
for a in ax5.flatten():
    letter = next(letters)
    a.text(0.05,0.9,letter,transform=a.transAxes,horizontalalignment="left",verticalalignment="center")

ax5[0,0].legend(loc="center left")
fig5.tight_layout()

plt.tight_layout()
fig.savefig("figures/interaction_boxplots.pdf")
fig5.savefig("figures/interactions_w_wo_pmut.pdf")
plt.show()
