"""
Purpose: load final stats npy files produced by class_plots in order to 
quickly produce a multi-panel plot of final N and D vs. T for ecosystems
and cores in single and var-TRC experiments.

Created: 2023 by Camille Febvre
Last modified: Jan 9, 2024 by Camille Febvre
"""
import numpy as np
import matplotlib.pyplot as plt

# produce these files from class_plots.py
single_path = "npy_files/final_stats_single-TRC_Jun_21_23.npy" #Mar_30_23.npy"
single250_path = "npy_files/final_stats_single-TRC_Dec_14_23.npy" #Mar_30_23.npy"
var_path = "npy_files/final_stats_var-TRC_Apr_13_23.npy"
var250_path = "npy_files/final_stats_var-TRC_Dec_14_23.npy"

n_rows = 6 # 6 if you want to plot Ni too, otherwise 4
color="k"

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

        try:
            self.med_Ni_by_T = np.array(out_dict['med_Ni'])
            self.Ni_q1_by_T = np.array(out_dict['q1_Ni'])
            self.Ni_q3_by_T = np.array(out_dict['q3_Ni'])

            self.coreNi_by_T = np.array(out_dict['med_coreNi'])
            self.coreNi_q1_by_T = np.array(out_dict['q1_coreNi'])
            self.coreNi_q3_by_T = np.array(out_dict['q3_coreNi'])
        except: pass
    
single_TRC = np.load(single_path,allow_pickle=True).item()
single_TRC_250 = np.load(single250_path,allow_pickle=True).item()
single_experiment = Experiment("single TRC",single_TRC)
single250_experiment = Experiment("single TRC",single_TRC_250)

var_TRC = np.load(var_path,allow_pickle=True).item()
var_TRC_250 = np.load(var250_path,allow_pickle=True).item()
var_experiment = Experiment("various TRC",var_TRC)
var250_experiment = Experiment("various TRC",var_TRC_250)

fig,ax = plt.subplots(4,2,sharex=True,sharey="row",figsize=(doub_width,1.3*doub_width))
# for arrhenius plots
f2,a2 = plt.subplots(n_rows,2,sharex=True,sharey="row",figsize=(doub_width,1.5*doub_width))
k = 8.6e-5 #eV
# Arrhenius lines
def expectation(x_range=np.r_[-36,42],m=-0.49,b=10):
    return b + m*x_range
x_range = np.r_[36,43]
line = expectation(x_range,b=26) #29
a2[0,0].plot(x_range,line,"b--")
a2[0,1].plot(x_range,line,"b--")
a2[0,0].set_ylim(min(line),max(line))
line = expectation(x_range,b=22.5) #27
a2[1,0].plot(x_range,line,"b--")
a2[1,1].plot(x_range,line,"b--")
a2[1,0].set_ylim(min(line),max(line))
if n_rows == 6:
    line = expectation(x_range,m=0.49,b=-16.3) #-21
    a2[2,0].plot(x_range,line,"b--")
    a2[2,1].plot(x_range,line,"b--")
    a2[2,0].set_ylim(min(line),max(line))
    line = expectation(x_range,b=25.8) #29
    a2[3,0].plot(x_range,line,"b--")
    a2[3,1].plot(x_range,line,"b--")
    a2[3,0].set_ylim(min(line),max(line))
    line = expectation(x_range,b=21) #25
    a2[4,0].plot(x_range,line,"b--")
    a2[4,1].plot(x_range,line,"b--")
    a2[4,0].set_ylim(min(line),max(line))
    line = expectation(x_range,m=0.49,b=-14) # -19
    a2[5,0].plot(x_range,line,"b--")
    a2[5,1].plot(x_range,line,"b--")
    a2[5,0].set_ylim(min(line),max(line))

    ylabels = iter(["Abundance, N","Species richness, D",r"Population, N$_i$","Core abundance","Core richness","Core population"])
else:
    line = expectation(x_range,b=29)
    a2[2,0].plot(x_range,line,"r--")
    a2[2,1].plot(x_range,line,"r--")
    a2[2,0].set_ylim(min(line),max(line))
    line = expectation(x_range,b=25)
    a2[3,0].plot(x_range,line,"r--")
    a2[3,1].plot(x_range,line,"r--")
    a2[3,0].set_ylim(min(line),max(line))

    ylabels = iter(["Abundance, N","Species richness, D","Core abundance","Core richness"])

for a in a2[:,1]:
    y1 = a.get_ylim()
    a.set_yticks(np.linspace(y1[0],y1[1],5))
    a3 = a.twinx()
    ylims = np.exp(y1) #a.get_ylim())
    a3.set_ylim(ylims[0],ylims[1]) #(min(exp.med_N_by_T[idx]),max(exp.med_N_by_T[idx]))
    ticklocs = np.linspace(y1[0],y1[1],5)
    new_ticklabels = []
    for tick in ticklocs:
        new_ticklabels.append(int(np.exp(tick)))
    # ticklabels = a.get_yticks() #np.linspace(min(np.floor(exp.med_N_by_T[idx])),np.floor(max(exp.med_N_by_T[idx])),10)
    #new_ticklabels=[]
    #for tick in ticklabels:
    #    new_ticklabels.append(int(np.exp(float(tick))))
    ytick_locs = []
    for tick in np.linspace(np.exp(y1[0]),np.exp(y1[1]),5):
        ytick_locs.append(int(tick))
    a3.set_yticks(ytick_locs) #np.linspace(np.exp(y1[0]),np.exp(y1[1]),5)) #ticklocs))
    a3.set_yticklabels(new_ticklabels) #,color="r")
    a3.set_ylabel(next(ylabels)) #,color="r")

T_range = np.r_[274:320]
for a in a2[-1,:]:
    a.set_xlabel(r"1/kT (eV$^{-1}$)")
    a.set_xlim(1/k/T_range[-1],1/k/T_range[0])
for a3 in a2[0,:]:
    a4 = a3.twiny()
    a4.set_xlabel("Temperature, T (K)") #,color="r")
    #a4.set_xlim(1/k/T_range[-1],1/k/T_range[0])
    a4.set_xlim(T_range[0],T_range[-1])
    bottom_tick_locs = np.linspace(1/k/T_range[-1], 1/k/T_range[0], 5)
    ticklabels = []
    for tick in bottom_tick_locs:
        ticklabels.append(int(np.round(1/k/tick)))
    top_tick_locs = np.linspace(T_range[0],T_range[-1],5)
    a4.set_xticks(top_tick_locs)
    a4.set_xticklabels(ticklabels)

letters = iter(["a)","b)","c)","d)","e)","f)","g)","h)","i)","j)","k)","l)"])
for a in a2.flatten():
    letter = next(letters)
    a.text(0.08,0.9,letter,transform=a.transAxes,horizontalalignment="left",verticalalignment="center")

letters = iter(["a)","b)","c)","d)","e)","f)","g)","h)","i)","j)","k)","l)"])
for a in ax.flatten():
    letter = next(letters)
    a.text(0.05,0.9,letter,transform=a.transAxes,horizontalalignment="left",verticalalignment="center")

# average multiple outputs?
averaging = False
weights = [.2,.8]
# single TRC
col = -1
for exp in [single_experiment,var_experiment]:
    col += 1
    if exp.experiment == "single TRC":
        ax[0,col].set_title("Single TRC")
        if averaging:
            exp2 = single250_experiment
    elif exp.experiment == "various TRC":
        ax[0,col].set_title("Various TRC")
        if averaging:
            exp2 = var250_experiment
    else:
        ax[0,col].set_title(exp.experiment)

    idx = exp.med_D_by_T > 0
    arr_T = 1/k/exp.temps[idx]

    # abundance
    if averaging: 
        new_med_N_by_T = weights[0]*exp.med_N_by_T[idx] + weights[1]*exp2.med_N_by_T[idx]
        new_q1_N_by_T = weights[0]*exp.N_q1_by_T[idx] + weights[1]*exp2.N_q1_by_T[idx]
        new_q3_N_by_T = weights[0]*exp.N_q3_by_T[idx] + weights[1]*exp2.N_q3_by_T[idx]
        ax[0,col].plot(exp.temps[idx],new_med_N_by_T,color=color)
        ax[0,col].fill_between(exp.temps[idx],new_q1_N_by_T,new_q3_N_by_T,alpha=.25,color=color)
        a2[0,col].plot(arr_T,np.log(new_med_N_by_T),color=color)
        a2[0,col].fill_between(arr_T,np.log(new_q1_N_by_T),np.log(new_q3_N_by_T),alpha=.25,color=color)
    else:
        ax[0,col].plot(exp.temps[idx],exp.med_N_by_T[idx],color=color)
        ax[0,col].fill_between(exp.temps[idx],exp.N_q1_by_T[idx],exp.N_q3_by_T[idx],alpha=.25,color=color)
        a2[0,col].plot(arr_T,np.log(exp.med_N_by_T[idx]),color=color)
        a2[0,col].fill_between(arr_T,np.log(exp.N_q1_by_T[idx]),np.log(exp.N_q3_by_T[idx]),alpha=.25,color=color)

        """
        a3 = a2[0,1].twinx()
        ylims = a2[0,1].get_ylim()
        a3.set_ylim(np.exp(ylims[0]),np.exp(ylims[1])) #(min(exp.med_N_by_T[idx]),max(exp.med_N_by_T[idx]))
        ticklabels = a2[0,0].get_yticks() #np.linspace(min(np.floor(exp.med_N_by_T[idx])),np.floor(max(exp.med_N_by_T[idx])),10)
        new_ticklabels=[]
        for tick in ticklabels:
            new_ticklabels.append(int(np.exp(float(tick))))
        a3.set_yticklabels(new_ticklabels) #,color="r")
        a3.set_ylabel("Abundance, N") #,color="r")
        

        if col == 1:
            a3 = a2[1,col].twinx()
            a3.set_ylim(min(exp.med_D_by_T[idx]),max(exp.med_D_by_T[idx]))
            ticklabels = a2[1,0].get_yticks() #np.linspace(min(np.floor(exp.med_N_by_T[idx])),np.floor(max(exp.med_N_by_T[idx])),10)
            new_ticklabels = []
            for tick in ticklabels:
                print(tick)
                new_ticklabels.append(int(np.exp(float(tick))))
            #tick_loc = a2[1,0].get_yticklocations()
            #a3.set_yticks(tick_loc)
            a3.set_yticklabels(new_ticklabels) #,color="r")
            a3.set_ylabel("Species richness, D") #,color="r")
            """

    # species richness
    ax[1,col].plot(exp.temps[idx],exp.med_D_by_T[idx],color=color)
    ax[1,col].fill_between(exp.temps[idx],exp.D_q1_by_T[idx],exp.D_q3_by_T[idx],alpha=.25,color=color)
    a2[1,col].plot(arr_T,np.log(exp.med_D_by_T[idx]),color=color)
    a2[1,col].fill_between(arr_T,np.log(exp.D_q1_by_T[idx]),np.log(exp.D_q3_by_T[idx]),alpha=.25,color=color)

    # population
    if n_rows == 6:
        try:
            a2[2,col].plot(arr_T,np.log(exp.med_Ni_by_T[idx]),color=color)
            a2[2,col].fill_between(arr_T,np.log(exp.Ni_q1_by_T[idx]),np.log(exp.Ni_q3_by_T[idx]),alpha=.25,color=color)
        except: pass
        row = 3
    else: row =2

    # core abundance
    ax[2,col].plot(exp.temps[idx],exp.coreN_by_T[idx],color=color)
    ax[2,col].fill_between(exp.temps[idx],exp.coreN_q1_by_T[idx],exp.coreN_q3_by_T[idx],alpha=.25,color=color)
    a2[row,col].plot(arr_T,np.log(exp.coreN_by_T[idx]),color=color)
    a2[row,col].fill_between(arr_T,np.log(exp.coreN_q1_by_T[idx]),np.log(exp.coreN_q3_by_T[idx]),alpha=.25,color=color)

    # core richness
    row += 1
    ax[3,col].plot(exp.temps[idx],exp.coreD_by_T[idx],color=color)
    ax[3,col].fill_between(exp.temps[idx],exp.coreD_q1_by_T[idx],exp.coreD_q3_by_T[idx],alpha=.25,color=color)
    a2[row,col].plot(arr_T,np.log(exp.coreD_by_T[idx]),color=color)
    a2[row,col].fill_between(arr_T,np.log(exp.coreD_q1_by_T[idx]),np.log(exp.coreD_q3_by_T[idx]),alpha=.25,color=color)

    # core populations
    if n_rows == 6:
        row+=1
        try:
            print("Plotting core populations")
            a2[row,col].plot(arr_T,np.log(exp.coreNi_by_T[idx]),color=color)
            a2[row,col].fill_between(arr_T,np.log(exp.coreNi_q1_by_T[idx]),np.log(exp.coreNi_q3_by_T[idx]),alpha=.25,color=color)
            #fdummy,adummy = plt.subplots()
            #adummy.plot(arr_T,np.log(exp.coreNi_by_T[idx]))
        except: pass

    print(exp.experiment," final N and D: ",exp.med_N_by_T[-1],exp.med_D_by_T[-1])
row = 0
ax[0,0].set_ylabel("Abundance, N")
a2[row,0].set_ylabel("log(N)")
row += 1
ax[1,0].set_ylabel("Species richness, D")
a2[row,0].set_ylabel("log(D)")
if n_rows == 6:
    row+=1
    a2[row,0].set_ylabel(r"log(N$_i$)")
    #a2[2,0].set_ylabel(r"Species populations, $N_i$")
row += 1
ax[2,0].set_ylabel("Core abundance")
a2[row,0].set_ylabel("log(core N)")
row+=1
ax[3,0].set_ylabel("Core species richness")
a2[row,0].set_ylabel(r"log(core D)")
if n_rows == 6:
    row += 1
    a2[row,0].set_ylabel(r"log(core N$_i$)")

a2[0,0].set_title("Single TRC")
a2[0,1].set_title("Various TRC")

#f2.tight_layout()

for a in ax[3,:]:
    a.set_xlabel("Temperature, T (K)")

try:
    for a in ax[0,:]:
        a.plot(exp.temps,exp.refN,"r--",label="Reference")
    for a in ax[1,:]:
        a.plot(exp.temps,exp.refD,"r--")
    ax[0,1].legend(loc="upper right")
except: pass

fig.savefig("figures/final_stats_all_together.pdf")
f2.savefig("figures/Arrh_plots_all_together.pdf")

plt.show()
