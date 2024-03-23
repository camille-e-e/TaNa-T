"""
Purpose: plot distributions of Topt for species in TTNM experiments.  Based on
Topt, calculate poff,T and plot that.  Divide poff,T by pdeath to find approximate
poff,i.  
Plot distributions of species fi.  Then calculate average Jij from fi+muN.  This
can be compared to the plots from make_final_interactions.

Inputs: 
- species_object_.....npy from TTNM model run.
- modelrun_.....npy from TTNM model run.

Created Jan 31,2024 by Camille Febvre
Last modified Jan 31, 2024 by Camille Febvre
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/home/cfebvre/repos/the_model_y3/src/geoTNM')
from MTE_TPC_combo import pdeath, poff_T

seeds = np.r_[1000:1250]
dates = ['Apr_13_23','Dec_14_23','Jun_21_23'] # Apr_13_23
experiment = "single-TRC" # 'var-TRC'
extra_folder = "varypmut-Tref303" #"varypmut-Tref303" # 'varypmut'

T_range = np.r_[274:320:3]
maxgens=9999
mu = 0.1

Topt_by_T = []
weighted_Topt_by_T = []
core_Topt_by_T = []

poff_by_T = []
core_poff_by_T = []

fi_by_T = []
core_fi_by_T = []

# set up figures
fig4,ax4 = plt.subplots(3,sharex=True) # fitness from spc_objects files
fig5,ax5 = plt.subplots(2,sharex=True) # fitness from spc_objects files
for a in ax5[:].flatten():
    a.plot(T_range,np.zeros((len(T_range),)),"k",linewidth=1)
if experiment != "single-TRC":
    fig,ax = plt.subplots(3,sharex=True) # Topt
    fig2,ax2 = plt.subplots(3,sharex=True) # poff,T(T,Topt)
    fig3,ax3 = plt.subplots(3,sharex=True) # poff,T/pdeath ~ poff,i
    for a in ax:
        a.plot(T_range,T_range,"b--")
    ax[-1].set_xlabel("Temperature (K)")
    ax[0].set_title(r"T_\mathrm{opt}")
    ax[0].set_ylabel(r"of species")
    ax[1].set_ylabel(r"of individuals")
    ax[2].set_ylabel(r"of core species")
    ax2[-1].set_xlabel("Temperature (K)")
    ax2[0].set_title(r"$p_\mathrm{off,T,T_\mathrm{opt}}$")
    ax2[0].set_ylabel(r"of species")
    ax2[1].set_ylabel(r"of individuals")
    ax2[2].set_ylabel(r"of core species")
    ax3[-1].set_xlabel("Temperature (K)")
    ax3[0].set_title(r"$\frac{p_\mathrm{off,T,T_\mathrm{opt}}}{p_\mathrm{death}}$")
    ax3[0].set_ylabel(r"of species")
    ax3[1].set_ylabel(r"of individuals")
    ax3[2].set_ylabel(r"of core species")
else:
    fig,ax = plt.subplots(2,sharex=True)
    ax[-1].set_xlabel("Temperature (K)")
    ax[0].set_ylabel(r"$p_\mathrm{off,T}$")
    ax[1].set_ylabel(r"$\frac{p_\mathrm{off,T}}{p_\mathrm{death}}$")

    ax[0].plot(T_range,poff_T(T_range))
    ax[1].plot(T_range,poff_T(T_range)/pdeath(T_range))
ax4[-1].set_xlabel("Temperature (K)")
ax4[0].set_title("Fitness")
ax4[0].set_ylabel(r"of species")
ax4[1].set_ylabel(r"of individuals")
ax4[2].set_ylabel(r"of core species")
# calculate average Jij from fi and N
ax5[-1].set_xlabel("Temperature (K)")
ax5[0].set_ylabel("Ecosystem")
ax5[1].set_ylabel("Core")
ax5[0].set_title(r"Expected mean $J_\mathrm{ij}$")

for T in T_range:
    print("T: ",T)
    Topt_this_T = []
    core_Topt_this_T = []
    weighted_Topt_this_T = []
    poff_this_T = []
    core_poff_this_T = []
    weighted_poff_this_T = []
    fi_this_T = []
    weighted_fi_this_T = []
    core_fi_this_T = []
    Jij_avg_this_T = []
    core_Jij_avg_this_T = []

    for seed in seeds:
        # find file with this seed and temperature
        for date in dates:
            location = f"/home/cfebvre/out_venus/TTNM_outputs/SteadyT/{experiment}/{extra_folder}/{date}/" 
            spc_file = location + f"species_objects_seed{seed}_{T}K.npy"
            model_file = location + f"modelrun_seed{seed}_{T}K.npy"
            if os.path.exists(spc_file):
                # when file is found, stop checking dates
                break
        # if file is not found, skip it and move on
        if not os.path.exists(spc_file): 
            print("File not found for seed ",seed)
            print(location)
            print(spc_file)
            continue
        # if file is found, collect Topt of each species
        else:
            spc_obj = np.load(spc_file,allow_pickle=True)
            modelrun = np.load(model_file,allow_pickle=True)[0]
            #print("modelrun file exists: ",os.path.exists(model_file))
            #print(type(modelrun))
            if len(modelrun.N_timeseries) >= maxgens - 8000:
                N = modelrun.N_timeseries[maxgens-8001]
                for spc in spc_obj:
                    # check if this species was alive at the end
                    if maxgens in spc.times_alive:
                        idx = list(spc.times_alive).index(maxgens)
                        Topt_this_T.append(spc.Topt)
                        poff_this_T.append(poff_T(T,[spc.Topt,11,-3]))
                        fi_this_T.append(spc.f_timeseries[idx])
                        Jij_avg_this_T.append(spc.f_timeseries[idx] + mu*N)
                        weighted_Topt_this_T.extend([spc.Topt] * spc.populations[idx])
                        weighted_poff_this_T.extend([poff_this_T[-1]] * spc.populations[idx])
                        weighted_fi_this_T.extend([fi_this_T[-1]]*spc.populations[idx])
                        if spc.is_core[idx]:
                            core_Topt_this_T.append(spc.Topt)
                            core_poff_this_T.append(poff_this_T[-1])
                            core_fi_this_T.append(spc.f_timeseries[idx])
                            core_Jij_avg_this_T.append(spc.f_timeseries[idx] + mu*N)
            else:
                print("length of timeseries too short: ",len(modelrun.N_timeseries))
    # plot distribution of Topts this T
    print("Plotting for T: ",T)
    if experiment != "single-TRC":
        ax[0].boxplot(Topt_this_T, positions = [T],widths=2,sym='')
        ax[1].boxplot(weighted_Topt_this_T, positions = [T],widths=2,sym='')
        ax[2].boxplot(core_Topt_this_T, positions = [T],widths=2,sym='')

        ax2[0].boxplot(poff_this_T, positions = [T],widths=2,sym='')
        ax2[1].boxplot(weighted_poff_this_T, positions = [T],widths=2,sym='')
        ax2[2].boxplot(core_poff_this_T, positions = [T],widths=2,sym='')

        ax3[0].boxplot(poff_this_T/pdeath(T), positions = [T],widths=2,sym='')
        ax3[1].boxplot(weighted_poff_this_T/pdeath(T), positions = [T],widths=2,sym='')
        ax3[2].boxplot(core_poff_this_T/pdeath(T), positions = [T],widths=2,sym='')

    ax4[0].boxplot(fi_this_T, positions = [T],widths=2,sym='')
    ax4[1].boxplot(weighted_fi_this_T, positions = [T],widths=2,sym='')
    ax4[2].boxplot(core_fi_this_T, positions = [T],widths=2,sym='')

    ax5[0].boxplot(Jij_avg_this_T, positions = [T],widths=2,sym='')
    ax5[1].boxplot(core_Jij_avg_this_T, positions = [T],widths=2,sym='')

    # append list of Topts this T to matrix of Topt_by_T, etc
#    Topt_by_T.append(Topt_this_T)
#    weighted_Topt_by_T.append(weighted_Topt_this_T)
#    core_Topt_by_T.append(core_Topt_this_T)

#    poff_by_T.append(poff_this_T)
#    core_poff_by_T.append(core_poff_this_T)


plt.show()


