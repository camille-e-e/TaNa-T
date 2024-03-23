import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# MTE_TPC_combo imported from output path

x_variable = "L" # "pmut" # "T" "mu" "C" "theta" "L"
#seed = np.r_[1000:1050] #[200:700] #[1000:2000] #:13009]
#T = np.r_[274:320:3] # False
probs = 'poff1.00_pdeath0.20' # np.r_[274:320:3] #[274:320:3] #[274:320:4] #[283:317:3] #[288:310:5] #[283:317:3] #False #288
dates = ['Mar_09_23','Mar_10_23'] #['Feb_24'] #,'Feb_23'] 
sample_size = 50
num_tries=40
experiment = 'prob_test/' # 'prob_test/' #'UnknownExperiment/' #'SteadyT/' #'SpunupT/' #'BasicTNM/' # 'SpunupT/'
#mu_range = [.03,.04,.06,.08,.3] # np.r_[0.05:0.3:.05]
mu_range = False #[.02,.03,.04,.05,.06,.08,.1,.15,.2,.25,.3] # np.r_[0.05:0.3:.05]
pmut_range = [0,.004,.007,.01,.02,.03]
T_range = np.r_[274:320:3]
C_range = [60,80,'control',120,140]
theta_range = [.15,.2,.3,.35]
L_range = [10,13,14,16,17,'control']
#extra_folder = ["/mu"+ str(i) for i in mu_range] 
#extra_folder = ["/pmut"+ str(i) for i in pmut_range] 
#extra_folder = "/MTE-env-1.6x/" #"/var-TRC-scaledup/"
#extra_folder = [f"C{i}/" for i in C_range] 
#extra_folder = [f"theta{i:.2f}/" for i in theta_range] 
#theta_range.insert(2,'control')
#extra_folder.insert(2,'control/')
extra_folder = [f"L{i}/" for i in L_range] 
ctrl = 20 # value of x_variable in standard TNM
if x_variable+"control/" in extra_folder:
    idx = list(extra_folder).index(x_variable+"control/")
    extra_folder[idx] = "control/"
print("extra_folder: ",extra_folder)
if x_variable != "L":
    L = 20 #20
#base_path = "/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/"
base_path = "/net/venus/kenes/user/cfebvre/out_venus/TTNM_outputs/"
maxgens=10000

if x_variable == "T":
    writefreq = 4000
    maxgens = int(maxgens - writefreq*int(maxgens/writefreq))
else:
    writefreq = False# 4000

# Set path to files
if np.shape(dates) == () and np.shape(extra_folder) == ():
    x_order = T_range
    if "home" in base_path:
        locat = [base_path+experiment+dates[0]+extra_folder]
    else:
        locat = [base_path+experiment+extra_folder+dates[0]]
    output_folder = locat
    #print("-----",locat,os.path.exists(locat))
else:
    #locat1 = base_path+experiment+dates[0]+extra_folder[0]
    #output_folder = locat1
    #locat = [locat1]
    locat = []
    x_order = []
    for d in dates:
        if np.shape(extra_folder) == ():
            #if x_variable in ["mu", "pmut","C","theta","L"]: #type(mu_range) != bool:
            if "home" in base_path:
                locat.append(base_path+experiment+d+extra_folder)
                x_order.append(float(extra_folder[len(x_variable)+1:]))
            else:
                locat.append(base_path+experiment+extra_folder+d)
                if extra_folder != "control/":
                    x_order.append(float(extra_folder[len(x_variable):-1]))
                else: 
                    x_order.append(ctrl)
            #print(locat[-1],os.path.exists(locat[-1]))

        else: # locat.append(base_path+experiment+dates[i]+extra_folder[i])
            for x_folder in extra_folder:
                print("treating ",x_folder,"...")
                if "home" in base_path:
                    temp_path = base_path+experiment+d+x_folder
                else: 
                    temp_path = base_path+experiment+x_folder+d
                #print(temp_path)
                if os.path.exists(temp_path):
                    locat.append(temp_path)
                    if "control" in x_folder:
                        x_now = ctrl
                    elif x_folder[-1] == "/":
                        x_now = float(x_folder[len(x_variable):-1])
                    elif x_folder[0] == "/":
                        x_now = float(x_folder[len(x_variable)+1:])
                    x_order.append(x_now)
                    #print(locat[-1],os.path.exists(locat[-1]))
                else: print(temp_path," not found")
    
    #output_folder = base_path+experiment+dates[0]
    output_folder = base_path+experiment+f"Figures_vary_{x_variable}"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
print("File locations: ", locat,"\n****")
print("x_order: ",x_order)

# Import TRC
if x_variable == "T":
    sys.path.append(locat[0])
    print("path final entry: ",sys.path[-1])
    print("MTE_TPC_combo.py" in os.listdir(locat[0]))
    from MTE_TPC_combo import poff_T

# make ending for all output pdf files
if np.shape(dates) == () and np.shape(extra_folder) == ():
    fig_ending = f"_{experiment[:-1]}{dates}{extra_folder[1:-1]}.pdf"
else:
    if np.shape(extra_folder) == ():
        fig_ending = f"_{experiment[:-1]}{dates[0]}{extra_folder[:-1]}.pdf"
    else:
        fig_ending = f"_{experiment[:-1]}{dates[0]}{extra_folder[0][:-1]}.pdf"


# %% Collect all output files to plot
if x_variable == "mu": #type(mu_range)!= bool:
    n_vars = len(mu_range)
    n_folders = len(mu_range)
elif x_variable == "T": 
    n_vars = len(T_range)
elif x_variable == "pmut": 
    n_vars = len(pmut_range)
    n_folders = len(pmut_range)
elif x_variable == "C": 
    n_vars = len(C_range)
    n_folders = len(C_range)
elif x_variable == "theta": 
    n_vars = len(theta_range)
    n_folders = len(theta_range)
elif x_variable == "L": 
    n_vars = len(L_range)
    n_folders = len(L_range)
else: n_vars = 1

# sort x variables
sort_idx = np.argsort(x_order)
x_sorted = np.sort(x_order)
folder_order = sort_idx

# cycle through all seeds in each x_folder
survival_by_x = np.zeros(n_vars)
Ntot_by_x = np.zeros(n_vars)
Dtot_by_x = np.zeros(n_vars)
N_by_x = [] # np.zeros((n_vars,)) # []
D_by_x = [] # np.zeros((n_vars,)) # []
coreD_by_x = []# np.zeros((n_vars,)) # []

for _ in range(n_vars):
    N_by_x.append([])
    D_by_x.append([])
    coreD_by_x.append([])

if x_variable == "T": 
    n_folders = 1
    folder_order = [0]
    divfiles_by_T = []
    ecosystfiles_by_T = []
    for _ in T_range:
        divfiles_by_T.append([])
        ecosystfiles_by_T.append([])

# cycle through folders and x_variables
print("Cycling through ",len(folder_order)," folders")
print("Order: ",folder_order)
ii = -1
for i in folder_order: #range(n_folders):
    ii += 1
    print("i: ",i," x: ",x_order[i])
    #mu = mu_range[i]
    if x_variable != "T": #type(mu_range) != bool:
        #N_by_x.append([])
        #D_by_x.append([])
        #coreD_by_x.append([])
        x = x_order[i]
        #x_folder = extra_folder[i]
        x_folder = locat[i]
        divfiles = []
        ecosystfiles = []
    for filename in os.listdir(locat[i]):
        if filename.endswith(".dat"):
            if filename.startswith("div"):
                divfiles.append(filename) 
            elif filename.startswith("pypy"): ecosystfiles.append(filename)
        elif filename.startswith("species_objects"):
            # *** RIGHT NOW SPECIES_OBJECTS WORKS FOR T AND PYPYFILES WORK FOR MU AND PMUT
            for i in range(len(T_range)):
                T = T_range[i]
                if filename.endswith(f"{T}K.npy"):
                    divfiles_by_T[i].append(filename)
                    break
        elif filename.startswith("modelrun"):
            for i in range(len(T_range)):
                T = T_range[i]
                if filename.endswith(f"{T}K.npy"):
                    ecosystfiles_by_T[i].append(filename)
                    break

                    
    if x_variable in ["mu","pmut","C","theta","L"]: # type(mu_range) != bool:
        for pyfile in ecosystfiles:
            py_fullpath = locat[i]+'/'+pyfile
            #print(py_fullpath,os.path.exists(py_fullpath))
            if not os.path.exists(py_fullpath): continue
            with open(py_fullpath,'r') as p:
                output = np.genfromtxt(p, delimiter=' ',skip_footer=1) # make maxgens x 9 matrix of output for current model run
                gen = output[0:,][0:,0]
                popu = output[0:,][0:,1]
                div = output[0:,][0:,2]
                enc = output[0:,][0:,3]
                core_size = output[0:,][0:,4]
                core_div = output[0:,][0:,5]
                Jtot = output[0:,][0:,7]

                if gen[-1] >= maxgens-10:
                    survival_by_x[i] += 1
                    N_by_x[ii].append(np.mean(popu[-100:]))
                    D_by_x[ii].append(np.mean(div[-100:]))
                    coreD_by_x[ii].append(np.mean(core_div[-100:]))
                    Ntot_by_x[ii] += np.mean(popu[-100:])
                    Dtot_by_x[ii] += np.mean(div[-100:])
    else:
        if type(locat) != str:
            sys.path.append(locat[0])
        else: sys.path.append(locat)
        from test_classes import Species,State
        for i in range(len(T_range)):
            T = T_range[i]
            files_this_T = ecosystfiles_by_T[i]
            for mfile in files_this_T:
                #print("mfile: ",mfile)
                m = np.load(locat[0]+mfile,allow_pickle=True)
                if len(m[0].N_timeseries) >= maxgens:
                    survival_by_x[i] += 1
                    N_by_x[i].append(np.mean(m[0].N_timeseries[-100:]))
                    D_by_x[i].append(np.mean(m[0].D_timeseries[-100:]))
                    coreD_by_x[i].append(np.mean(m[0].coreD_timeseries[-100:]))
                else: 
                    print(len(m[0].N_timeseries))


#print("N_by_x: ", N_by_x)
#print("D_by_x: ", D_by_x)
Nmed_by_x, Dmed_by_x, coreDmed_by_x, Nstd_by_x, Dstd_by_x, coreDstd_by_x = [],[],[],[],[],[]
Nmean_by_x, Dmean_by_x,coreDmean_by_x, Ncov_by_x, Dcov_by_x, coreDcov_by_x = [],[],[],[],[],[]
Nq1_by_x, Nq3_by_x, Dq1_by_x, coreDq1_by_x, Dq3_by_x, coreDq3_by_x = [],[],[],[],[],[]
for i in range(n_vars):
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

if x_variable == "mu":
    x_lab = r"$\mu$"
elif x_variable == "pmut":
    x_lab = r"Mutation probability"
elif x_variable == "C":
    x_lab = "Interaction scale (C)"
elif x_variable == "theta":
    x_lab = r"Connectivity ($\theta$)"
elif x_variable == "L":
    x_lab = "Genome lenght (L)"
elif x_variable == "T":
    x_lab = "Temperature (K)"
    x_order = T_range
if len(x_order) == 1:
    fig,ax = plt.subplots()
    ax.hist(N_by_x)
    ax.set_xlabel("Final abundance")
    ax.set_ylabel("Count")
    fig.text(.6,.8,f"survival: {int(survival_by_x[0])}/{sample_size}")

    fig,ax = plt.subplots()
    ax.hist(D_by_x)
    ax.set_xlabel("Final diversity")
    ax.set_ylabel("Count")
    fig.text(.6,.8,f"survival: {int(survival_by_x[0])}/{sample_size}")
else:
    # Plot survival
    fig,ax = plt.subplots()
    ax.scatter(x_sorted, survival_by_x/sample_size)
    ax.set_xlabel(x_lab)
    if x_variable == "T":
        temps = np.r_[T_range[0]:T_range[-1]:.5]
        TRC = poff_T(temps)
        ax.plot(temps,TRC,"r",label="TRC")
        ax.legend()
    ax.set_ylabel(f"survival fraction (n = {sample_size})")

    plt.savefig(output_folder+f"/survival_varying_{x_variable}"+fig_ending)

    # Plot average survival
    fig,ax = plt.subplots()
    ax.errorbar(x_sorted,Nmed_by_x,yerr=Nstd_by_x,fmt='o',label=f"median +- std")
    #ax.scatter(x_order, Nmed_by_x) #X_avg)
    ax.fill_between(x_sorted, Nq1_by_x, Nq3_by_x, alpha=.3,label="interquartile range")
    ax.set_xlabel(x_lab)
    ax.set_ylabel("Avg abundance of surviving ecosystems")
    ax.legend()

    plt.savefig(output_folder+f"/final_N_varying_{x_variable}"+fig_ending)

    # Plot average diversity
    fig,ax = plt.subplots()
    ax.errorbar(x_sorted,Dmed_by_x,yerr=Dstd_by_x,fmt='o',label=f"median +- std")
    ax.fill_between(x_sorted, Dq1_by_x, Dq3_by_x, alpha=.3,label="intequartile range")
    ax.set_xlabel(x_lab)
    ax.set_ylabel("Avg diversity of surviving ecosystems")
    ax.legend()

    plt.savefig(output_folder+f"/final_D_varying_{x_variable}"+fig_ending)

    # Plot average core diversity
    fig,ax = plt.subplots()
    ax.errorbar(x_sorted,coreDmed_by_x,yerr=coreDstd_by_x,fmt='o',label=f"median +- std")
    ax.fill_between(x_sorted, coreDq1_by_x, coreDq3_by_x, alpha=.3,label="intequartile range")
    ax.set_xlabel(x_lab)
    ax.set_ylabel("Avg core diversity of surviving ecosystems")
    ax.legend()
    
    plt.savefig(output_folder+f"/final_coreD_varying_{x_variable}"+fig_ending)

#plt.figure()
#plt.plot(x_sorted,label="x_sorted")
#plt.plot(x_order,'--',label="x_order")
#plt.legend)
    
plt.show()

