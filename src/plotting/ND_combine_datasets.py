"""
Purpose: load N and D vs. time and temperature matrix npy files produced 
by by_spc_plots and combine them from multiple dates.  
Quickly produce a multi-panel plot of final N and D vs. 
T for ecosystems and cores in single and var-TRC experiments.

Created: Jan 11, 2024 by Camille Febvre from copy of final_ND_all_experiments.py
Last modified: Jan 11, 2024 by Camille Febvre
"""
import numpy as np
import matplotlib.pyplot as plt

# DICTIONARIES OF FINAL STATS
# produce these files from class_plots.py
single_path1 = "npy_files/final_stats_single-TRC_Jun_21_23.npy" #Mar_30_23.npy"
single_path2 = "npy_files/final_stats_single-TRC_Dec_14_23.npy" #Mar_30_23.npy"
single_samplesizes = [50,250]
var_path1 = "npy_files/final_stats_var-TRC_Apr_13_23.npy"
var_path2 = "npy_files/final_stats_var-TRC_Dec_14_23.npy"
var_samplesizes = [50,250]

# TIMESERIES 
# produce these files from by_spc_plots.py
paths = {}
paths['singleN_path1'] = "npy_files/N_by_T_single-TRC_Jun_21_23.npy"
paths['singlecoreN_path1'] = "npy_files/core_N_by_T_single-TRC_Jun_21_23.npy" 
paths['singleD_path1'] = "npy_files/D_by_T_single-TRC_Jun_21_23.npy"
paths['singlecoreD_path1'] = "npy_files/core_D_by_T_single-TRC_Jun_21_23.npy"
paths['singleN_path2'] = "npy_files/N_by_T_single-TRC_Dec_14_23.npy"
paths['singlecoreN_path2'] = "npy_files/core_N_by_T_single-TRC_Dec_14_23.npy"
paths['singleD_path2'] = "npy_files/D_by_T_single-TRC_Dec_14_23.npy" 
paths['singlecoreD_path2'] = "npy_files/core_D_by_T_single-TRC_Dec_14_23.npy" 
paths['varN_path1'] = "npy_files/N_by_T_var-TRC_Apr_13_23.npy"
paths['varcoreN_path1'] = "npy_files/core_N_by_T_var-TRC_Apr_13_23.npy"
paths['varD_path1'] = "npy_files/D_by_T_var-TRC_Apr_13_23.npy"
paths['varcoreD_path1'] = "npy_files/core_D_by_T_var-TRC_Apr_13_23.npy"
paths['varN_path2'] = "npy_files/N_by_T_var-TRC_Dec_14_23.npy"
paths['varD_path2'] = "npy_files/D_by_T_var-TRC_Dec_14_23.npy"
paths['varcoreN_path2'] = "npy_files/core_N_by_T_var-TRC_Dec_14_23.npy"
paths['varcoreD_path2'] = "npy_files/core_D_by_T_var-TRC_Dec_14_23.npy"


# figure sizes for JTB
sin_width = 3.54 # single column
doub_width = 7.48 # double column

class Experiment:
    def __init__(self,experiment,out_dict):
        self.experiment = experiment
        self.temps = np.array(out_dict['temps'])
        try:
            self.refN = np.array(out_dict['refN'])
            self.refD = np.array(out_dict['refD'])
        except: pass

        self.med_N_by_T = np.array(out_dict['med_N'])
        self.N_q1_by_T = np.array(out_dict['q1_N'])
        self.N_q3_by_T = np.array(out_dict['q3_N'])

        self.med_D_by_T = np.array(out_dict['med_D'])
        self.D_q1_by_T = np.array(out_dict['q1_D'])
        self.D_q3_by_T = np.array(out_dict['q3_D'])

        self.coreN_by_T = np.array(out_dict['med_coreN'])
        self.coreN_q1_by_T = np.array(out_dict['q1_coreN'])
        self.coreN_q3_by_T = np.array(out_dict['q3_coreN'])

        self.coreD_by_T = np.array(out_dict['med_coreD'])
        self.coreD_q1_by_T = np.array(out_dict['q1_coreD'])
        self.coreD_q3_by_T = np.array(out_dict['q3_coreD'])

def combine_datasets(loc1,loc2):
    dat1 = np.load(loc1,allow_pickle=True)
    dat2 = np.load(loc2,allow_pickle=True)
    return np.hstack([dat1,dat2])


# Final stats dictionaries (unnecessary?)    
single_TRC_1 = np.load(single_path1,allow_pickle=True).item()
single_TRC_2 = np.load(single_path2,allow_pickle=True).item()
single_experiment1 = Experiment("single TRC",single_TRC_1)
single_experiment2 = Experiment("single TRC",single_TRC_2)
single_experiments = [single_experiment1,single_experiment2]

var_TRC_1 = np.load(var_path1,allow_pickle=True).item()
var_TRC_2 = np.load(var_path2,allow_pickle=True).item()
var_experiment1 = Experiment("various TRC",var_TRC_1)
var_experiment2 = Experiment("various TRC",var_TRC_2)
var_experiments = [var_experiment1,var_experiment2]

fig,ax = plt.subplots(4,2,sharex=True,sharey="row",figsize=(doub_width,1.3*doub_width))

temps = np.r_[274:320:3]
col = -1
for experiment in ["Single TRC","Various TRC"]:
    col += 1
    ax[0,col].set_title(experiment)

    if experiment == "Single TRC":
        exp = "single"
    else: exp = "var"

    outputs = {}
    for output_type in ["N","coreN","D","coreD"]:
        path1 = f"{exp}{output_type}_path1"
        path2 = f"{exp}{output_type}_path2"
        data = combine_datasets(paths[path1],paths[path2])
        np.save(f"npy_files/{exp}_combined_{output_type}_vs_t_T.npy",data)
        outputs[output_type] = data

    # Find medians and quartiles
    med_N_by_T = np.nanmedian(outputs["N"][:,:,-1],axis=1)
    q1_N_by_T = np.nanquantile(outputs["N"][:,:,-1],.25,axis=1)
    q3_N_by_T = np.nanquantile(outputs["N"][:,:,-1],.75,axis=1)

    med_D_by_T = np.nanmedian(outputs["D"][:,:,-1],axis=1)
    q1_D_by_T = np.nanquantile(outputs["D"][:,:,-1],.25,axis=1)
    q3_D_by_T = np.nanquantile(outputs["D"][:,:,-1],.75,axis=1)

    med_coreN_by_T = np.nanmedian(outputs["coreN"][:,:,-1],axis=1)
    q1_coreN_by_T = np.nanquantile(outputs["coreN"][:,:,-1],.25,axis=1)
    q3_coreN_by_T = np.nanquantile(outputs["coreN"][:,:,-1],.75,axis=1)

    med_coreD_by_T = np.nanmedian(outputs["coreD"][:,:,-1],axis=1)
    q1_coreD_by_T = np.nanquantile(outputs["coreD"][:,:,-1],.25,axis=1)
    q3_coreD_by_T = np.nanquantile(outputs["coreD"][:,:,-1],.75,axis=1)

    # eliminate outliers (median diversity not greater than zero)
    idx = med_D_by_T > 0

    ax[0,col].plot(temps[idx],med_N_by_T[idx])
    ax[0,col].fill_between(temps[idx],q1_N_by_T[idx],q3_N_by_T[idx],alpha=.5)

    ax[1,col].plot(temps[idx],med_D_by_T[idx])
    ax[1,col].fill_between(temps[idx],q1_D_by_T[idx],q3_D_by_T[idx],alpha=.5)

    ax[2,col].plot(temps[idx],med_coreN_by_T[idx])
    ax[2,col].fill_between(temps[idx],q1_coreN_by_T[idx],q3_coreN_by_T[idx],alpha=.5)

    ax[3,col].plot(temps[idx],med_coreD_by_T[idx])
    ax[3,col].fill_between(temps[idx],q1_coreD_by_T[idx],q3_coreD_by_T[idx],alpha=.5)

    print(experiment," final N and D: ",med_N_by_T[-1],med_D_by_T[-1])

ax[0,0].set_ylabel("Abundance, N")
ax[1,0].set_ylabel("Species richness, D")
ax[2,0].set_ylabel("Core abundance")
ax[3,0].set_ylabel("Core species richness")

for a in ax[3,:]:
    a.set_xlabel("Temperature, T (K)")

plot_ref = 1
for exp in np.hstack([single_experiments,var_experiments]):
    if plot_ref ==1:
        print("exp: .....",exp)
        #exp = Experiment("single TRC",single_TRC_1)
        try: 
            for a in ax[0,:]:
                a.plot(exp.temps,exp.refN,"r--",label="Reference")
            for a in ax[1,:]:
                a.plot(exp.temps,exp.refD,"r--")
            ax[0,1].legend(loc="upper right")
            plot_ref = 0
        except: pass

plt.savefig("figures/final_stats_all_together.pdf")

plt.show()
