import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# MTE_TPC_combo imported from output path

find_best_fit = False
plot_TRC = True
module_path = "/home/cfebvre/repos/the_model_y3/src/geoTNM"
if plot_TRC: 
    sys.path.append(module_path)
    import MTE_TPC_combo as Teq
x_variables = ["poff","pdeath","pmut","L","mu","C","theta"] # "pmut" # "T" "mu" "C" "theta" "L"
#x_variables = ["pmut","L","mu","C","theta"] # "pmut" # "T" "mu" "C" "theta" "L"
#seed = np.r_[1000:1050] #[200:700] #[1000:2000] #:13009]
#T = np.r_[274:320:3] # False
dates = ['Feb_09','Mar_09_23','Mar_10_23','Feb_28','Feb_21','Feb_23','Jun_21_23'] #['Feb_24'] #,'Feb_23'] 
sample_size = 50
num_tries=40
experiment = 'prob_test/' # 'prob_test/' #'UnknownExperiment/' #'SteadyT/' #'SpunupT/' #'BasicTNM/' # 'SpunupT/'
#mu_range = [.03,.04,.06,.08,.3] # np.r_[0.05:0.3:.05]
poff_range = np.r_[.1:1:.1]
pdeath_range = np.r_[.1:1:.1]
mu_range = [.02,.03,.04,.05,.06,.08,.1,.15,.2,.25,.3] # np.r_[0.05:0.3:.05]
pmut_range = [0,.004,.007,.01,.02,.03]
T_range = np.r_[274:320:3]
C_range = [60,80,100,120,140,200,300]
theta_range = [.15,.2,.25,.3,.35]
L_range = [10,13,14,16,17,20]
poff_extra_folder = ['/D_init_60']
pdeath_extra_folder = ['/D_init_60']
mu_extra_folder = ["/mu"+ str(i) for i in mu_range] 
pmut_extra_folder = ["/pmut"+ str(i) for i in pmut_range] 
#extra_folder = "/MTE-env-1.6x/" #"/var-TRC-scaledup/"
C_extra_folder = [f"C{i}/" for i in C_range] 
C_extra_folder[2] = 'control/'
theta_extra_folder = [f"theta{i:.2f}/" for i in theta_range] 
theta_extra_folder[2] = 'control/'
L_extra_folder = [f"L{i}/" for i in L_range] 
L_extra_folder[-1] = 'control/'

ctrl = {}
ctrl['poff'] = .9
ctrl['pdeath'] = .2
ctrl['pmut'] = .01
ctrl['L'] = 20 # value of x_variable in standard TNM
ctrl['mu'] = .1
ctrl['C'] = 100
ctrl['theta'] = .25

all_ranges = {}
all_ranges['poff'] = poff_range
all_ranges['pdeath'] = pdeath_range
all_ranges['mu'] = mu_range
all_ranges['pmut'] = pmut_range
all_ranges['L'] = L_range
all_ranges['mu'] = mu_range
all_ranges['C'] = C_range
all_ranges['theta'] = theta_range

ctrl_locations = {}
for var in x_variables:
    if ctrl[var] in all_ranges[var]:
        ctrl_locations[var] = list(all_ranges[var]).index(ctrl[var])
    else: 
        print("Error: control not in range: ",ctrl[var],all_ranges[var])
        exit()

def probstr(poff=.9,pdeath=.2):
    return f'poff{poff:.2f}_pdeath{pdeath:.2f}' # np.r_[274:320:3] #[274:320:3] #[274:320:4] #[283:317:3] #[288:310:5] #[283:317:3] #False #288

home_path = "/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/"
base_path = "/net/venus/kenes/user/cfebvre/out_venus/TTNM_outputs/"
maxgens=10000

#if x_variable == "T":
#    writefreq = 4000
#    maxgens = int(maxgens - writefreq*int(maxgens/writefreq))
#else:
writefreq = False# 4000

# Set path to files
locat = {}
x_order = {}
# sort x variables
sort_idx = {}
x_sorted = {}
folder_order = {}
if "poff" in x_variables: 
    all_folders = [poff_extra_folder,pdeath_extra_folder,pmut_extra_folder,L_extra_folder,mu_extra_folder,C_extra_folder,theta_extra_folder]
else: all_folders = [pmut_extra_folder,L_extra_folder,mu_extra_folder,C_extra_folder,theta_extra_folder]
if len(all_folders) != len(x_variables):
    print("Length of all folders and x_variables not the same: ",x_variables,all_folders)

i = -1
for extra_folder in all_folders: 
    i += 1
    x_var = x_variables[i]
    locat[x_var] = []
    x_order[x_var] = []
    for x_folder in extra_folder:
        print("x_var: ",x_var,"... treating ",x_folder,"...")
        for d in dates:
            if x_var in ["poff","pdeath","mu","pmut"]:
                # model outputs on anahim
                temp_path = home_path+experiment+d+x_folder
            else: # model outputs on venus
                temp_path = base_path+experiment+x_folder+d
            #print(temp_path)
            if os.path.exists(temp_path):
                locat[x_var].append(temp_path)
                if x_var == "poff":
                    x_order[x_var] = poff_range
                elif x_var == "pdeath":
                    x_order[x_var] = pdeath_range
                else:
                    if "control" in x_folder:
                        x_now = ctrl[x_var]
                    elif x_folder[-1] == "/":
                        x_now = float(x_folder[len(x_var):-1])
                    elif x_folder[0] == "/":
                        x_now = float(x_folder[len(x_var)+1:])
                    x_order[x_var].append(x_now)
                    #print(locat[-1],os.path.exists(locat[-1]))
            else: print(temp_path," not found")

        # sort x variables
        sort_idx[x_var] = np.argsort(x_order[x_var])
        x_sorted[x_var] = np.sort(x_order[x_var])
        folder_order[x_var] = sort_idx[x_var]
    
#output_folder = base_path+experiment+dates[0]
output_folder = base_path+experiment+f"Figures_vary_params"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
print("File locations: ", locat,"\n****")
print("x_order: ",x_order)

# make ending for all output pdf files
fig_ending = f"_{experiment[:-1]}_all1D_params.pdf"


# %% Collect all output files to plot
n_vars = {}
n_folders = {}
if "poff" in x_variables:
    n_vars['poff'] = len(poff_range)
    n_folders['poff'] = 1 # len(poff_range)
    n_vars['pdeath'] = len(pdeath_range)
    n_folders['pdeath'] = 1 # len(pdeath_range)
n_vars['mu'] = len(mu_range)
n_folders['mu'] = len(mu_range)
n_vars['pmut'] = len(pmut_range)
n_folders['pmut'] = len(pmut_range)
n_vars['C'] = len(C_range)
n_folders['C'] = len(C_range)
n_vars['theta'] = len(theta_range)
n_folders['theta'] = len(theta_range)
n_vars['L'] = len(L_range)
n_folders['L'] = len(L_range)

# For each variable, make a row.
# The columns will be survival, abundance, diversity, core diversity
fig,ax = plt.subplots(4,len(x_variables),sharex="col",sharey='row',figsize=(12,10))
pmut_fig,pmut_ax = plt.subplots()
pred_fig,pred_ax = plt.subplots()

col = -1
for x_var in x_variables: # cycle thru poff, pdeath, mu, pmut etc...
    print(x_var)
    col += 1

    # cycle through all seeds in each x_folder
    survival_by_x = np.zeros(n_vars[x_var])
    Ntot_by_x = np.zeros(n_vars[x_var])
    Dtot_by_x = np.zeros(n_vars[x_var])
    N_by_x = [] # np.zeros((n_vars,)) # []
    D_by_x = [] # np.zeros((n_vars,)) # []
    coreD_by_x = []# np.zeros((n_vars,)) # []

    for _ in range(n_vars[x_var]):
        N_by_x.append([])
        D_by_x.append([])
        coreD_by_x.append([])

    if x_var in ["poff","pdeath"]: # only one extra_folder
        if x_var == "poff": 
            x_range = poff_range
        elif x_var == "pdeath":
            x_range = pdeath_range
            
    # cycle through folders and x_variables
    print("Cycling through ",len(folder_order[x_var])," folders")
    print("Order: ",folder_order[x_var])
    ii = -1
    if x_var in ["poff","pdeath"]: # only one extra_folder
        cycle = range(len(x_range))
    else: cycle = folder_order[x_var]
    for i in cycle: # folder_order[x_var]: #range(n_folders):
        # i values not necessarily in order
        ii += 1 # integers in order
        if x_var == "poff":
            pstr = probstr(poff=poff_range[ii])
        elif x_var == "pdeath":
            pstr = probstr(pdeath=pdeath_range[ii])
        elif x_var == "C" and x_order[x_var][ii] in [200,300]:
            pstr = probstr(pdeath=.2,poff=1)
        else: pstr = False
        # j is 0 for poff,pdeath, otherwise i
        if pstr: 
            print("Prob str: ", pstr)
            j = 0 # only one location for all poff and pdeath files
        else: j = i
        print("x_var ",x_var,"i: ",i," x: ",x_order[x_var][ii])
        #mu = mu_range[i]
        #print("x_order[x_var]: ",x_order[x_var])
        #print("iith value: ",ii)
        #print(x_order[x_var][ii])
        x = x_order[x_var][ii]
        x_folder = locat[x_var][j]
        #if x_var in ["poff","pdeath"]:
        #    x_folder = locat[x_var][i]
        #else: x_folder = locat[x_var][0]
        divfiles = []
        ecosystfiles = []
        for filename in os.listdir(locat[x_var][j]):
            found = 0
            if filename.endswith(".dat"):
                if filename.startswith("div"):
                    if not pstr:
                        divfiles.append(filename) 
                        found += 1
                    elif pstr in filename:
                        divfiles.append(filename) 
                        found += 1
                elif filename.startswith("pypy"): 
                    if not pstr:
                        ecosystfiles.append(filename)
                        found += 1
                    elif pstr in filename:
                        ecosystfiles.append(filename)
                        found += 1
        print("Files found: ",found)    
        if found == 0: 
            print("Searched: ",locat[x_var][j])
            #print(os.listdir(locat[x_var][j]))
                        
        if x_var in ["poff","pdeath","mu","pmut","C","theta","L"]: # type(mu_range) != bool:
            for pyfile in ecosystfiles:
                py_fullpath = locat[x_var][j]+'/'+pyfile
                #print(py_fullpath,os.path.exists(py_fullpath))
                if not os.path.exists(py_fullpath): continue
                with open(py_fullpath,'r') as p:
                    output = np.genfromtxt(p, delimiter=' ',skip_footer=1) # make maxgens x 9 matrix of output for current model run
                    if len(np.shape(output)) > 1:
                        gen = output[0:,][0:,0]
                        popu = output[0:,][0:,1]
                        div = output[0:,][0:,2]
                        enc = output[0:,][0:,3]
                        core_size = output[0:,][0:,4]
                        core_div = output[0:,][0:,5]
                        Jtot = output[0:,][0:,7]

                        if gen[-1] >= maxgens-10:
                            survival_by_x[ii] += 1
                            N_by_x[ii].append(np.mean(popu[-100:]))
                            D_by_x[ii].append(np.mean(div[-100:]))
                            coreD_by_x[ii].append(np.mean(core_div[-100:]))
                            Ntot_by_x[ii] += np.mean(popu[-100:])
                            Dtot_by_x[ii] += np.mean(div[-100:])

    Nmed_by_x, Dmed_by_x, coreDmed_by_x, Nstd_by_x, Dstd_by_x, coreDstd_by_x = [],[],[],[],[],[]
    Nmean_by_x, Dmean_by_x,coreDmean_by_x, Ncov_by_x, Dcov_by_x, coreDcov_by_x = [],[],[],[],[],[]
    Nq1_by_x, Nq3_by_x, Dq1_by_x, coreDq1_by_x, Dq3_by_x, coreDq3_by_x = [],[],[],[],[],[]
    for i in range(n_vars[x_var]):
        Nmed_by_x.append(np.median(N_by_x[i]))
        Dmed_by_x.append(np.median(D_by_x[i]))
        coreDmed_by_x.append(np.median(coreD_by_x[i]))
        Nstd_by_x.append(np.std(N_by_x[i]))
        Dstd_by_x.append(np.std(D_by_x[i]))
        coreDstd_by_x.append(np.std(coreD_by_x[i]))

        Nmean_by_x.append(np.mean(N_by_x[i]))
        Dmean_by_x.append(np.mean(D_by_x[i]))
        coreDmean_by_x.append(np.mean(coreD_by_x[i]))
        Ncov_by_x.append(np.mean(N_by_x[i])/np.std(N_by_x[i]))
        Dcov_by_x.append(np.mean(D_by_x[i])/np.std(D_by_x[i]))
        coreDcov_by_x.append(np.mean(coreD_by_x[i])/np.std(coreD_by_x[i]))

        Nq1_by_x.append(np.nanquantile(N_by_x[i],.25))
        Nq3_by_x.append(np.nanquantile(N_by_x[i],.75))
        Dq1_by_x.append(np.nanquantile(D_by_x[i],.25))
        Dq3_by_x.append(np.nanquantile(D_by_x[i],.75))
        coreDq1_by_x.append(np.nanquantile(coreD_by_x[i],.25))
        coreDq3_by_x.append(np.nanquantile(coreD_by_x[i],.75))

    if x_var == "mu":
        x_lab = r"1/$\mu$"
        x_vals = 1/np.array(x_sorted[x_var])
    elif x_var == "pmut":
        x_lab = r"Mutation probability"
    elif x_var == "C":
        x_lab = "Interaction scale (C)"
    elif x_var == "theta":
        x_lab = r"Connectivity ($\theta$)"
    elif x_var == "L":
        x_lab = "Genome lenght (L)"
    elif x_var == "poff":
        x_lab = r"p$_\mathrm{off,scaler}$"
    elif x_var == "pdeath":
        x_lab = r"p$_\mathrm{death}$"
    if x_var != "mu":
        x_vals = x_sorted[x_var]

    ctrl_idx = ctrl_locations[x_var]
    cmap = plt.get_cmap("Greys")
    colors = cmap(np.linspace(0,1,10))
    color = colors[3]

    # Plot survival
    print("x_sorted: ",x_sorted[x_var])
    print(x_var,survival_by_x)
    print("len(survival_by_x): ",len(survival_by_x))
    ax[0,col].scatter(x_vals, survival_by_x/sample_size,c="k")
    ax[0,col].scatter(x_vals[ctrl_idx],survival_by_x[ctrl_idx]/sample_size,c="red") #,marker="+")
    #ax[0,col].set_ylim(.5,1)
    #ax[0,col].set_xlabel(x_lab)

    # Plot average abundance
    ax[1,col].fill_between(x_vals, Nq1_by_x, Nq3_by_x,label="interquartile range",color=color)
    ax[1,col].scatter(x_vals,Nmed_by_x,c="k") #,yerr=Nstd_by_x,fmt='o',label=f"median +- std")
    ax[1,col].scatter(x_vals[ctrl_idx],Nmed_by_x[ctrl_idx],c="red") #"white",marker="+")
    #ax.scatter(x_order, Nmed_by_x) #X_avg)
    #ax[1,col].set_xlabel(x_lab)
#    ax[1,col].set_ylabel("Avg abundance of surviving ecosystems")
    #ax[1,col].legend()

    # Plot average diversity
    ax[2,col].fill_between(x_vals, Dq1_by_x, Dq3_by_x,label="intequartile range",color=color)
    ax[2,col].scatter(x_vals,Dmed_by_x,c="k") #,yerr=Dstd_by_x,fmt='o',label=f"median +- std")
    ax[2,col].scatter(x_vals[ctrl_idx],Dmed_by_x[ctrl_idx],c="r") #"white",marker="+")
    #ax[2,col].set_xlabel(x_lab)
#    ax[2,col].set_ylabel("Avg diversity of surviving ecosystems")
    #ax.legend()

    # Plot average core diversity
    ax[3,col].fill_between(x_vals, coreDq1_by_x, coreDq3_by_x,label="intequartile range",color=color)
    ax[3,col].scatter(x_vals,coreDmed_by_x,c="k") #,yerr=coreDstd_by_x,fmt='o',label=f"median +- std")
    ax[3,col].scatter(x_vals[ctrl_idx],coreDmed_by_x[ctrl_idx],c="r") #"white",marker="+")
    ax[3,col].set_xlabel(x_lab)
#    ax[3,col].set_ylabel("Avg core diversity of surviving ecosystems")
    #ax.legend()

    ax[0,0].set_ylabel(f"Survival fraction")
    
    if x_var == "pmut":
        print("Pmut values: ",x_vals)
        print("Diversity: ",Dmed_by_x)
        # Plot average diversity at different pmut on its own plot
        pmut_ax.fill_between(x_vals, Dq1_by_x, Dq3_by_x,label="intequartile range",color=color)
        pmut_ax.scatter(x_vals,Dmed_by_x,c="k") #,yerr=Dstd_by_x,fmt='o',label=f"median +- std")
        pmut_ax.scatter(x_vals[ctrl_idx],Dmed_by_x[ctrl_idx],c="r") #"white",marker="+")
        pmut_ax.set_ylabel("Avg. final diversiyt")
        pmut_ax.set_xlabel(r"$p_\mathrm{mut}$")

        if plot_TRC: 
            experiment_temps = np.r_[274:320:3]
            #pmut_T = Teq.pmut(experiment_temps)
            #pmut_ax.plot(pmut_T,50*np.ones(len(pmut_T)),"ro",label="TRC")
            i = -1
            for Ti in experiment_temps:
                #print("Ti: ",Ti)
                #print("pmut(Ti): ",Teq.pmut(Ti))
                i += 1
                x = Teq.pmut(Ti)
                #y = i/len(x_vals)*175
                if i%2 > 0:
                    y = 125
                else: y = 150
                pmut_ax.plot([x,x],[0,175],'b')
                pmut_ax.text(x,y,Ti,rotation=90,color='k')

        # find line of best fit
        if find_best_fit:
            from fit_curve import best_fit_slope_intercept as best_line, best_fit_log as best_log, log_func
            from scipy.optimize import curve_fit
            # best fit is a line?
            m,b = best_line(x_vals,Dmed_by_x)
            def regression_line(x_in,m,b):
                return m*x_in + b
            pmut_ax.plot(x_vals,regression_line(x_vals,m,b),color="r",label="best linear fit")
            T_vals = np.r_[274:320]
            pmut_at_T = Teq.pmut(T_vals)
            predicted_div = regression_line(pmut_at_T,m,b)
            pred_ax.plot(T_vals,predicted_div,label="div as linear pmut")

            # best fit is logarithmic?
            # Logarithmic fit
            def log_func(x,a,b,c):
                return a*np.log(b*x) + c
            def root_func(x,a,b,c):
                return a*np.sqrt(b*x) + c
            popt,pcov = curve_fit(root_func,x_vals,Dmed_by_x,p0=[400,4,-10])
            a,b,c = popt
            pmut_ax.plot(x_vals,root_func(np.array(x_vals),a,b,c),"--",label="best root fit")
            pmut_ax.legend()
            predicted_div2 = root_func(pmut_at_T,a,b,c)
            pred_ax.plot(T_vals,predicted_div2,"--",label="div as root pmut")
            pred_ax.legend()

            pred_ax.set_xlabel("Temperature (K)")
            pred_ax.set_ylabel("Expected Diversity")
        else: # use interpolation
            from scipy import interp
            # interpolate line between points of div vs. pmut
            T_vals = np.r_[274:320:3]
            pmut_at_T = Teq.pmut(T_vals)
            div_at_T = interp(pmut_at_T,x_vals,Dmed_by_x)
            # plot predicted div(T) on pred_ax
            pred_ax.plot(T_vals,div_at_T,"o-")
            pred_ax.set_xlabel("Temperature (K)")
            pred_ax.set_ylabel("Expected Diversity")

            np.save(output_folder+"/predicted_div.npy",div_at_T)
            print("Saved to ",output_folder+"predicted_div.npy")

ax[1,0].set_ylabel(f"Abundance")
ax[2,0].set_ylabel(f"Diversity")
ax[3,0].set_ylabel(f"Core div.")
ax[3,0].set_ylim(0)

plt.tight_layout()

fig.savefig(output_folder+"/all_params"+fig_ending)
fig.savefig("figures/all_params"+fig_ending)
pmut_fig.savefig(output_folder+"/D_vs_pmut"+fig_ending)
    
plt.show()

