"""
Created Tuesday, July 26, 2022

This script is based on avg_timeseries.py.  
It produces plots that show the average of ecosystem
measures (such as population, diversity, interactions, etc.)
over time.  

The distinction between this script and avg_timeseries
is that averages are taken only over extant ecosystems 
(so exinct ecosystems don't change the mean population
of extant ecosystems).

"""
# Import modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import date,datetime,timedelta
import time
import string

# Define which experiment to analyze
# -----------------------------------
folder = "SteadyT/" # "SpunupT" # "SteadyT/" # ends in /
experiment = "Sep_29/" # "Feb_08/" # "Jan_09/" #,"Oct_11/"] #["Oct_11/","Nov_03/"] # "Aug_10/" # "Jun_20/" # "Jul_21/" # "Sep_09/" # "Jul_21/" # ends in /
extra_folder = "MTE_TPC_combo/" # "MTE-env-scaledup/" #,"MTE-env/"] # ["single-TPC/","one-TRC/"] # "poff_is_roff/" # "Variable_Tresponse/" # "MTE_TPC_combo/" # "Variable_Tresponse/" #"poff_is_roff/" # ends in /
num_tries = 40 #number of other dates to try for file name #other_dates = [experiment[:4] + i for i in list(np.array(np.r_[12:30],dtype=str))] # if filename dates don't match folder date, list other filename dates
sample_size = 100 # or False
base_path = "/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/"
maxgens = 10_000 # lenght of experiment (in generations)
spinup = 1000 #
multi_temp = True # one temperature or multiple
show_plots = True # if False, this code can be run without X11 forwarding
# ------------------------------------
# End of user-defined inputs
draw_MTE = False
if folder == "SpunupT/":
    maxgens += spinup

def MTE(T):
    B0 = 0.19*1.9e10 #
    Ea = 0.6 # eV
    k = 8.6E-5 # ev/K Boltzmann's constant
    B = B0 * np.exp(-Ea/k/T)
    return B

def pdeath(T):
    Tr = 294
    dTr = 0.05
    Ad = 6000
    TD = 1/Tr - 1/T
    return dTr*np.exp(Ad*TD)

# Set path to files
if np.shape(experiment) == () and np.shape(extra_folder) == ():
    locat = base_path+folder+experiment+extra_folder
    output_folder = locat
else:
    locat1 = base_path+folder+experiment[0]+extra_folder[0]
    output_folder = locat1
    locat = [locat1]
    for i in range(len(experiment)):
        if np.shape(extra_folder) == ():
            locat.append(base_path+folder+experiment[0]+extra_folder)
        else: locat.append(base_path+folder+experiment[i]+extra_folder[i])

# make ending for all pdf files
if np.shape(experiment) == () and np.shape(extra_folder) == ():
    fig_ending = f"{folder[:-1]}{experiment[:-1]}{extra_folder[:-1]}.pdf"
else:
    if np.shape(extra_folder) == ():
        fig_ending = f"{folder[:-1]}{experiment[0][:-1]}{extra_folder[:-1]}.pdf"
    else:
        fig_ending = f"{folder[:-1]}{experiment[0][:-1]}{extra_folder[0][:-1]}.pdf"

# figure out which computer we're on
filepath = os.getcwd()
if filepath[0] == 'C': # windows
    system = 'windows'
else:
    system = "linux"

# Make directory for all plots
if system == 'linux':
    figure_folder = output_folder+"Figures_"+folder
    if not folder.endswith('/'):
        figure_folder += '/'
elif system == 'windows':
    if output_folder == './':
        output_folder = os.getcwd()
    figure_folder = output_folder + "Figures_"+folder
    if not folder.endsiwth('\\'):
        figure_folder += '\\'
else: Exception("plotting directory not found: ",figure_folder)


if os.path.isdir(figure_folder) != True:
    os.mkdir(figure_folder)
    print("Making new folder")
print("Figure folder: ",figure_folder,"exists: ",os.path.exists(figure_folder))

# Collect all output files to plot
divfiles = []
ecosystfiles = []
if np.shape(locat) == ():
    for filename in os.listdir(locat):
        if filename.endswith(".dat"):
            if filename.startswith("div"):
                divfiles.append(filename) 
            elif filename.startswith("pypy"): ecosystfiles.append(filename)
else: 
    for location in locat:
        for filename in os.listdir(location):
            if filename.endswith(".dat"):
                if filename.startswith("div"):
                    divfiles.append(filename) 
                elif filename.startswith("pypy"): ecosystfiles.append(filename)

print(ecosystfiles)

seeds = set()
if multi_temp:
    print("Multi temp")
    temperatures = set()

# track order of seeds and Ts
order = []

# loop through all files in ecosystfiles and figure out seeds and temperatures
for file in ecosystfiles:
    if np.shape(locat) == ():
        filename = locat + file
    else: 
        for location in locat:
            filename = location+file
            if os.path.isfile(filename):
                break
    if os.path.isfile(filename) != True:
        print("error, file doesn't exist. ",filename)
        continue
    # Determine the seed
    loc1 = filename.index('seed')
    try: loc2 = filename[loc1:].index(experiment[:-1]) # find date in filename
    except: # if experiment took longer than a day to run, filename may not match folder date
        marker = filename[loc1:].index('_')
        date_now = filename[loc1+marker-3:loc1+marker+3]
        loc2 = marker-3
#        for d in np.r_[0:num_tries]: #len(other_dates)]:
#            date1 = datetime.strptime(experiment[:-1],"%b_%d")
#            adjusted_date = date1 + timedelta(days=int(d))
#            date2 = adjusted_date.strftime("%b_%d")
#            try: 
#                loc2 = filename[loc1:].index(date2) #other_dates[d])
#                break
#            except: continue
    seed = filename[loc1+4:loc1+loc2] # loc1+4 begins after "seed", loc2 is end of seed
    seeds.add(seed)
    
    # find temperature
    if multi_temp:
        loc3 = filename[loc1+loc2:].index('K')
        #temperature = filename[loc1+loc2+len(experiment):loc1+loc2+loc3]
        temperature = filename[loc1+loc2+loc3-3:loc1+loc2+loc3]
        temperatures.add(int(float(temperature)))
        order.append([seed,str(int(float(temperature)))])
    else: order.append(seed)

# if only a certain sample size is desired, toss any files after sample_size of that T is acheived
if sample_size and multi_temp:
    seeds_each_T = np.zeros(len(temperatures))
    seeds_sampled_by_T = np.zeros(len(temperatures))
    temp_list = np.sort(list(temperatures))
    order2 = []
    i = -1
    for pair in order:
        i += 1
        seed,T = pair
        T_idx = list(temp_list).index(float(T))
        seeds_each_T[T_idx] += 1
        if seeds_each_T[T_idx] <= sample_size:
            order2.append(pair)
            seeds_sampled_by_T[T_idx] += 1
        else: 
            ecosystfiles.pop(i)
            i -= 1
    print("Seeds each T: ")
    print(seeds_each_T)
    print(seeds_sampled_by_T)
    print("----------------------------")
    order = order2

# Pre-allocate space
# -----------------
# Matrix of all populations (for all T and seeds)
popu_all = np.zeros((maxgens,len(ecosystfiles))) # time increases down rows, experiments sorted into columns
if multi_temp:
    temperatures = np.sort(list(temperatures)) # list of ints, sorted
    popu_sum_by_T = {}
    div_sum_by_T = {}
    enc_sum_by_T = {}
    coresize_sum_by_T = {}
    corediv_sum_by_T = {}
    Jtot_sum_by_T = {}
    quakes_by_T = {}
    quakecount_by_T = {}
    living_at_t_by_T = np.zeros((maxgens,len(temperatures)))
else:
    popu_sum = np.zeros((maxgens,len(ecosystfiles)))
    div_sum = np.zeros((maxgens,len(ecosystfiles)))
    enc_sum = np.zeros((maxgens,len(ecosystfiles)))
    coresize_sum = np.zeros((maxgens,len(ecosystfiles)))
    corediv_sum = np.zeros((maxgens,len(ecosystfiles)))
    Jtot_sum = np.zeros((maxgens,len(ecosystfiles)))

order_copy = order.copy()
order = iter(order)

# Check for files that got split up over multiple days into multiple files
duplicates = [pair for pair in order_copy if order_copy.count(pair)>1]
# non-repeating list of duplicates:
unique_dups = []
for dup in duplicates:
    if dup not in unique_dups:
        unique_dups.append(dup)
# find locatio nin list of filenames of two files with same seed and T
dup_IDs = []
for pair in unique_dups:
    ID1 = order_copy.index(pair)
    ID2 = ID1+1+order_copy[ID1+1:].index(pair)
    if order_copy[ID1] != order_copy[ID2]:
        print("OH NO! duplicate indexing not working!")
    dup_IDs.append([ID1,ID2])

# ------------------------
# Define colormap
cmap = plt.get_cmap('hot')
if multi_temp:
    colors = cmap(np.linspace(0,.8,len(temperatures)))
else: 
    colors = cmap(np.linspace(0,.8,len(ecosystfiles)))

# -------------------------
# Cycle through files to collect model outputs

# keep track of abandoned IDs of duplicate files
abandonedIDs = []
# keep track of files with weird shapes
unusual_files = []
# number of experiments collected (excludes unfound files etc)
count = 0
# order in loop
n = -1

# start loop 
for file in ecosystfiles:
    count +=1
    n += 1
    split_file = False
    if np.shape(locat) == ():
        filename = locat + file
    else: 
        for location in locat:
            filename = location+file
            if os.path.isfile(filename):
                break
    seed,temperature = next(order)
    #print(seed,temperature)
    if [seed,temperature] in unique_dups:
        # THIS IS A SPLIT FILE
        # If this is the first file, we will grab outputs from 
        # both files.  If it's the second file, we'll ignore it.
        for pair in dup_IDs: # pair = [ID1, ID2]
            if n in pair: # is n ID1 or ID2?
                split_file = pair.index(n) + 1
                firstID = min(pair)
                secondID = max(pair)
                filename2 = ecosystfiles[secondID]
                break
        if split_file == False: # something went wrong
            print("OH NO! split file didn't find out if this is first or second chunk, n: ",n)
        elif split_file == 2:
            print("Abandoning ID: ",pair[1])
            abandonedIDs.append(n)
            count -= 1
            continue
    
    # Obtain model output
    # -------------------
    try: 
        output = np.genfromtxt(filename, delimiter=' ') # make maxgens x 9 matrix of output for current model run
        if split_file:
            output2 = np.genfromtxt(locat+filename2,delimiter=' ')
            output = np.vstack([output,output2])
            if len(output) > maxgens:
                print("Wrong output size for files ",pair,". Shape of output: ",np.shape(output))
    except:
        print("wrong size of output file, seed ",seed)
        unusual_files.append(filename)
        count-=1
        continue

    # shape of output should be (9,maxgens); otherwise extinction may have occured
    if type(output) == int:
        print("Filename: ",filename," is just ",output)
        unusual_files.append(filename)
        count-=1
        continue
    elif len(np.shape(output)) < 2: 
        if np.shape(output)[0] >= 9: # (9,):
            # mass extinction occured after first generation
            # fill entire matrix with zeros
            tmp = np.zeros((maxgens,np.shape(output)[0]))
            tmp[0] = output
            output = tmp.copy()
        else:
            print("No model output detected or taken into consideration.")
            print(np.shape(output))
            unusual_files.append(filename)
            count -= 1
            continue
    elif len(output) > maxgens:
        unusual_files.append(filename)
        print("output being truncated at maxgens: ",filename)
        output = output[:,][:maxgens]

    # Interpret model output in columns
    gen = output[0:,][0:,0]
    popu = output[0:,][0:,1]
    div = output[0:,][0:,2]
    enc = output[0:,][0:,3]
    core_size = output[0:,][0:,4]
    core_div = output[0:,][0:,5]
    Jtot = output[0:,][0:,7]
    
    #print(f"shape of output arrays: ",np.shape(popu),type(popu))

    # how long is this ecosystem alive?
    if len(popu) > 0:
        T_col = list(temperatures).index(int(temperature))
        living_at_t_by_T[:len(popu),T_col] += np.array(popu)>0

    # approximate time in steady state (based on core diversity) (T/F)
    method = 1
    core_2to5 = [0]
    for t in range(len(core_div)): # cycle through all times in core_div array
        if core_div[t]<6 and core_div[t]>=2:
            if method == 1: core_2to5.append(1)
            elif method == 2:
                if core_div[t-1] == core_div[t]:
                    core_2to5.append(1)
                else: core_2to5.append(0)
        else: core_2to5.append(0)
    # smooth results by only counting steady states longer than 20 gen
    smoothed = [0]
    SS_starts = []
    quakes = []
    quake_count = 0
    for u in range(20,len(div)):
        if sum(core_2to5[u-20:u]) == 20: # past 20 gen all in SS
            smoothed.append(1)
            if smoothed[-2] == 0: # SS began 20 gen ago
                SS_starts.append(u-20)
        else: 
            smoothed.append(0)
            if smoothed[-2] == 1: # quake just occured
                quakes.append(u-20)
                quake_count+=1

    # add this vector to the list of vectors
    popu_all[:len(popu),count-1] = popu
    if multi_temp:
        # get ready to account for things at this T if this is the first output of this T
        if str(temperature) not in quakes_by_T.keys():
            print("Adding new key: ",str(temperature))
            quakes_by_T[str(temperature)] = []
            quakecount_by_T[str(temperature)] = []
            popu_sum_by_T[str(temperature)] = np.zeros((maxgens))
            div_sum_by_T[str(temperature)] = np.zeros((maxgens))
            enc_sum_by_T[str(temperature)] = np.zeros((maxgens))
            coresize_sum_by_T[str(temperature)] = np.zeros((maxgens))
            corediv_sum_by_T[str(temperature)] = np.zeros((maxgens))
            Jtot_sum_by_T[str(temperature)] = np.zeros((maxgens))

        popu_sum_by_T[str(temperature)][:len(popu)] += popu
        div_sum_by_T[str(temperature)][:len(div)] += div
        enc_sum_by_T[str(temperature)][:len(div)] += enc
        coresize_sum_by_T[str(temperature)][:len(div)] += core_size
        corediv_sum_by_T[str(temperature)][:len(div)] += core_div
        Jtot_sum_by_T[str(temperature)][:len(div)] += Jtot
        quakes_by_T[str(temperature)].append(quakes)
        quakecount_by_T[str(temperature)].append(quake_count)

    else: #not multi_temp
        popu_sum[:len(popu)] += popu
        div_sum[:len(div)] += div
        enc_sum[:len(div)] += div
        coresize_sum[:len(div)] += div
        corediv_sum[:len(div)] += div
        Jtot_sum[:len(div)] += div
# End output-collection loop, now we have info for all files 

# Process quakes
# HAVEN'T SET THIS UP HERE *** !!!

# Average output
if multi_temp: 
    # get TPC curve from input
    #import geoTNM.temperature_effects as Teq
    if np.shape(locat) == ():
        sys.path.append(locat)
    else: sys.path.append(locat[0])
    if not "MTE_TPC_combo.py" in os.listdir(sys.path[-1]):
        print("Error: MTE_TPC_combo not in locat: ",locat)
    import MTE_TPC_combo as Teq
    TPC = Teq.poff_T(temperatures) - Teq.pdeath(temperatures) 
    temp_smooth = np.linspace(temperatures[0],temperatures[-1],100)
    TPC_smooth = 3.3*Teq.poff_T(temp_smooth) - Teq.pdeath(temp_smooth) 

    # pre-allocate space for averages sorted by T
    popu_avg_by_T,div_avg_by_T,enc_avg_by_T,coresize_avg_by_T,corediv_avg_by_T,Jtot_avg_by_T = [],[],[],[],[],[]


    # PLOT 1: number of ecosystems living by time
    plt.figure()
    # loop through temperatures and divide sum of populations by number of living ecosystems
    for T in temperatures:
        T_col = list(temperatures).index(T)
        # for this temperature, get time series of # living ecosystems
        living_at_t = living_at_t_by_T[:,T_col]
        print(f"shape of living_at_t at {T}K: ",np.shape(living_at_t))

        # if at some point in time, all experiments went extinct, find out when
        if 0 in living_at_t:
            first0 = list(living_at_t).index(0)
        # otherwise there were some living experiments for all time
        else: first0 = len(living_at_t)

        # use number of living experiments at each time to find averages of other timeseries
        print("Keys of popu_sum_by_T: ",popu_sum_by_T.keys())
        popu_avg = popu_sum_by_T[str(T)][:first0]/living_at_t[:first0]
        popu_avg_by_T.append(np.array(popu_avg))
        div_avg = div_sum_by_T[str(T)][:first0]/living_at_t[:first0]
        div_avg_by_T.append(np.array(div_avg))
        coresize_avg = coresize_sum_by_T[str(T)][:first0]/living_at_t[:first0]
        coresize_avg_by_T.append(np.array(coresize_avg))
        corediv_avg = corediv_sum_by_T[str(T)][:first0]/living_at_t[:first0]
        corediv_avg_by_T.append(np.array(corediv_avg))
        Jtot_avg = Jtot_sum_by_T[str(T)][:first0]/living_at_t[:first0]
        Jtot_avg_by_T.append(np.array(Jtot_avg))

        if show_plots:
            # for each temperature, plot living_at_t
            plt.plot(living_at_t,label=f"T={T}K",color=colors[T_col])
    if show_plots:
        # after loop through T's has ended, label axes etc.
        plt.legend(bbox_to_anchor=(1,1))
        plt.title("Number of ecosystems alive by time and temp.")
        plt.xlabel("Time,t (gen.)")
        plt.xscale("log")
        plt.ylabel("Number of living ecosystems")
        plt.tight_layout()
        #if np.shape(experiment) == () and np.shape(extra_folder) == ():
        plt.savefig(figure_folder+f"n_living_ecos_"+fig_ending) #{folder[:-1]}{experiment[:-1]}{extra_folder[:-1]}.pdf")
        #else:
        #    if np.shape(extra_folder) == ():
        #        plt.savefig(figure_folder+f"n_living_ecos_{folder[:-1]}{experiment[0][:-1]}{extra_folder[:-1]}.pdf")
        #    else:
        #        plt.savefig(figure_folder+f"n_living_ecos_{folder[:-1]}{experiment[0][:-1]}{extra_folder[0][:-1]}.pdf")
        #plt.show()

    # PLOT 2: timeseries of average population of living ecosystems
    plt.figure()
    for T_col in range(len(temperatures)):
        plt.plot(popu_avg_by_T[T_col],color=colors[T_col],label=f"T={temperatures[T_col]}")
    plt.title(f"Average population of living ecosystems, {len(seeds)} seeds, {folder[:-1]}")
    plt.xlabel("Time,t (gen.)")
    plt.ylabel("Average population,<N> (indiv.)")
    plt.xscale("log")
    plt.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig(figure_folder+f"avg_living_popu_"+fig_ending) #{folder[:-1]}{experiment[:-1]}{extra_folder[:-1]}.pdf")
    #plt.show()

    # PLOT 3: timeseries of popu_avg INCLUDING extincitons
    plt.figure()
    popu_avg_all_by_T = []
    for T_col in range(len(temperatures)):
        T = str(temperatures[T_col])
        popu_avg = popu_sum_by_T[T]/len(seeds)
        popu_avg_all_by_T.append(popu_avg)
        
        plt.plot(popu_avg,color=colors[T_col],label=f"T={T}")
    plt.title(f"Average of all ecosystems (including extinctions)\n{len(seeds)} seeds, {folder[:-1]}")
    plt.xlabel("Time,t (gen.)")
    plt.xscale("log")
    plt.ylabel("Average population,<N> (indiv.)")
    plt.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig(figure_folder+f"avg_all_popu_"+fig_ending) #{folder[:-1]}{experiment[:-1]}{extra_folder[:-1]}.pdf")

    # PLOT 4: scatter plot of final number of surviving ecosystems
    fig,ax = plt.subplots() #1,2,sharey=True)
    #plt.suptitle(f"Number of surviving ecosystems (out of {len(seeds)} at each T)")
    #for T_col in range(len(temperatures)):
    #    ax[0].plot(living_at_t_by_T[:,T_col],color=colors[T_col],label=temperatures[T_col])
    #ax[0].set_ylabel("Number of ecosystems alive",color="b")
    #ax[0].set_xlabel("Time,t (gen.)")
    #ax[0].set_xscale("log")
    #ax[0].set_title("Survival over time")
    #ax[0].legend(loc="upper left")

    #ax[1].set_title(f"Survival after {maxgens} gen.")
    ax.plot(temperatures,living_at_t_by_T[-1,:]/seeds_sampled_by_T,'*-',label="survival")
    ax.fill_between(temperatures,np.zeros((len(temperatures),)),living_at_t_by_T[-1,:]/seeds_sampled_by_T,alpha=.2)
    ax.text(0.1,0.7,f"{int(sum(living_at_t_by_T[-1,:]))/int(sum(seeds_sampled_by_T))*100:.1f}% survival overall",color="blue",transform=ax.transAxes,horizontalalignment="left",verticalalignment="center")
    # ax.set_ylabel("Proportion of ecosystems alive",color="b")
    ax.set_ylabel("Fraction") #,color="b")
    ax2 = ax #plt.twinx(ax)
    ax2.plot(temp_smooth,pdeath(temp_smooth),":",color='red',label=r"$p_{death}$")
    if np.shape(extra_folder) == ():
        if 'MTE' in extra_folder:
            draw_MTE = True
    elif 'MTE' in extra_folder[0]:
        draw_MTE = True
    if draw_MTE: #'MTE' in extra_folder:
        ax2.plot(temp_smooth,MTE(temp_smooth),"--",color="black",label="MTE")
    #    ax2.legend(loc="upper right")
        ax.set_ylabel("Fraction",color="k")
        #ax2.set_ylabel("MTE envelope")
    elif 'var' not in extra_folder:
        ax2.plot(temp_smooth,TPC_smooth,"r",label=r"$r_{max}$")
        ax.set_ylabel("Fraction",color="k")
    ax2.legend(loc="upper right")
    #ax2.set_ylabel(r"Input TPC,$r_{max}$")
        #ax2.set_ylabel(r"$r_{max}$ ($\frac{\Delta N}{gen.}$)",color="r")
    print("Living at end (by T):")
    print(living_at_t_by_T[-1,:])
    print("Temperatures:")
    print(temperatures)
    ax.set_xlabel("Temperature,T (K)")
    ax.set_ylim(0,1)
    ax2.set_ylim(0,1)
    plt.tight_layout()
    plt.savefig(figure_folder+f"survival_by_T_"+fig_ending) #{folder[:-1]}{experiment[:-1]}{extra_folder[:-1]}.pdf")

    # PLOT 5: timeseries of diversity of living ecosystems
    plt.figure()
    for T_col in range(len(temperatures)):
        plt.plot(div_avg_by_T[T_col],color=colors[T_col],label=f"T={temperatures[T_col]}")
    plt.title(f"Average diversity of living ecosystems, {len(seeds)} seeds, {folder[:-1]}")
    plt.xlabel("Time,t (gen.)")
    plt.xscale("log")
    plt.ylabel("Average diversity (species)")
    plt.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig(figure_folder+f"avg_living_div_"+fig_ending) #{folder[:-1]}{experiment[:-1]}{extra_folder[:-1]}.pdf")

    # PLOT 6: timeseries of corepop
    plt.figure()
    for T_col in range(len(temperatures)):
        plt.plot(coresize_avg_by_T[T_col],color=colors[T_col],label=f"T={temperatures[T_col]}")
    plt.title(f"Average population of core in living ecosystems, {len(seeds)} seeds, {folder[:-1]}")
    plt.xlabel("Time,t (gen.)")
    plt.xscale("log")
    plt.ylabel("Average population of core (indiv.)")
    plt.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig(figure_folder+f"avg_living_coresize_"+fig_ending) #{folder[:-1]}{experiment[:-1]}{extra_folder[:-1]}.pdf")

    # PLOT 7: timeseries of corediv
    plt.figure()
    for T_col in range(len(temperatures)):
        plt.plot(corediv_avg_by_T[T_col],color=colors[T_col],label=f"T={temperatures[T_col]}")
    plt.title(f"Average diversity of core in living ecosystems, {len(seeds)} seeds, {folder[:-1]}")
    plt.xlabel("Time,t (gen.)")
    plt.xscale("log")
    plt.ylabel("Average diversity of core (species)")
    plt.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig(figure_folder+f"avg_living_corediv_"+fig_ending) #{folder[:-1]}{experiment[:-1]}{extra_folder[:-1]}.pdf")

    # PLOT 8: timeseries of Jtot
    plt.figure()
    for T_col in range(len(temperatures)):
        plt.plot(Jtot_avg_by_T[T_col],color=colors[T_col],label=f"T={temperatures[T_col]}")
    plt.title(f"Avg interactions of core species in live ecosystems, {len(seeds)} seeds, {folder[:-1]}")
    plt.xlabel("Time,t (gen.)")
    plt.xscale("log")
    plt.ylabel("Average interactions of core (unitless)")
    plt.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig(figure_folder+f"avg_living_Jtot_"+fig_ending) #{folder[:-1]}{experiment[:-1]}{extra_folder[:-1]}.pdf")


    # PLOT 9: put all timeseries into one figure
    fig, ax = plt.subplots(2,2,sharex=True)
    #plt.suptitle("Avg. characteristics of living ecosystems overtime")
    for T_col in range(len(temperatures)):
        ax[0,0].set_title("Core avg.")
        ax[0,1].set_title("Ecosyst. avg.")

        ax[0,0].plot(coresize_avg_by_T[T_col],color=colors[T_col],label=temperatures[T_col])
        ax[0,0].set_ylabel("Abundance")
        ax[0,0].set_xscale("log")
        ax[0,1].plot(popu_avg_by_T[T_col],color=colors[T_col],label=temperatures[T_col])
        ax[0,1].set_xscale("log")
        
        ax[1,0].plot(corediv_avg_by_T[T_col],color=colors[T_col],label=temperatures[T_col])
        ax[1,0].set_ylabel("Diversity")
        ax[1,0].set_xscale("log")
        ax[1,1].plot(div_avg_by_T[T_col],color=colors[T_col],label=temperatures[T_col])
        ax[1,1].set_xscale("log")

        #ax[2,0].plot(Jtot_avg_by_T[T_col],color=colors[T_col],label=temperatures[T_col])
        #ax[2,0].set_ylabel("Interactions")
        #ax[2,0].set_xscale("log")
        #ax[2,1].plot(popu_avg_by_T[T_col],color=colors[T_col],label=f"temperatures[T_col]}")

    ax[1,0].set_xlabel("Time,t (gen.)")
    ax[1,1].set_xlabel("Time,t (gen.)")

    popu_ylim = ax[0,1].get_ylim()
    ax[0,0].set_ylim(popu_ylim)
    #ax[0,0].legend(bbox_to_anchor=(0,1.02,2.2,.102),loc=3,ncol=6,mode="expand",borderaxespad=0)
    fig.subplots_adjust(right=0.85, wspace=.25)
    ax[0,1].legend(bbox_to_anchor = (1,1))

    #Get the lengend handles and labels 
    #h1, l1 = ax[0,1].get_legend_handles_labels() 
    #h2, l2 = ax[1,1].get_legend_handles_labels()

    #Shrink the subplots to make room for the legend
    #for axi in fig.axes:
    #    box = axi.get_position()
    #    axi.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])

    #Make the legend
    #ax[1,0].legend(h1+h2, l1+l2,  bbox_to_anchor=(0,-.05, 2.2,-0.15), loc=9,ncol=6)

    #lines, labels = [],[]
    #Line, Label = fig.axes[0].get_legend_handles_labels()
    #lines.extend(Line)
    #labels.extend(Label)
    #fig.legend(lines,labels,bbox_to_anchor=(.7,.9))
    
        #ax[0,1].legend(bbox_to_anchor=(1,1))
    
    # label subplots
    ax[0,0].text(0.05,0.9,"A",transform=ax[0,0].transAxes,horizontalalignment="left",verticalalignment="center")
    ax[0,1].text(0.05,0.9,'B',transform=ax[0,1].transAxes,horizontalalignment='left',verticalalignment='center')
    ax[1,0].text(0.05,0.9,'C',transform=ax[1,0].transAxes,horizontalalignment='left',verticalalignment='center')
    ax[1,1].text(0.05,0.9,'D',transform=ax[1,1].transAxes,horizontalalignment='left',verticalalignment='center')

    #axs = ax.flat
    #for n, a in enumerate(axs):
    #    a.text(-0.1, 1.1, string.ascii_uppercase[n], transform=a.transAxes,size=15) #, weight='bold')

    #plt.tight_layout()
    plt.savefig(figure_folder+f"TTNM_timeseries_popdiv_"+fig_ending) #{folder[:-1]}_{experiment[:-1]}{extra_folder[:-1]}.pdf")

    # PREPARE plot 10: get final states
    final_pop_byT, final_div_byT, final_corepop_byT, final_corediv_byT, final_Jtot_byT,final_survival_byT = [],[],[],[],[],[]
    for T_col in range(len(temperatures)):
        popu_avg_this_T = popu_avg_by_T[T_col]
        div_avg_this_T = div_avg_by_T[T_col]
        coresize_avg_this_T = coresize_avg_by_T[T_col]
        corediv_avg_this_T = corediv_avg_by_T[T_col]
        Jtot_avg_this_T = Jtot_avg_by_T[T_col]
        # add zeros to end of timeseries if it doesn't go til maxgens
        if len(popu_avg_this_T) < maxgens:
            extra0s = maxgens - len(popu_avg_this_T)
            popu_avg_this_T = np.append(popu_avg_this_T,np.zeros(extra0s))
            div_avg_this_T = np.append(div_avg_this_T,np.zeros(extra0s))
            coresize_avg_this_T = np.append(coresize_avg_this_T,np.zeros(extra0s))
            corediv_avg_this_T = np.append(corediv_avg_this_T,np.zeros(extra0s))
            Jtot_avg_this_T = np.append(Jtot_avg_this_T,np.zeros(extra0s))

#        print(f"len(popu_avg_by_T[{T_col}]: {len(popu_avg_by_T[T_col])}")
        final_pop_byT.append(np.mean(popu_avg_this_T[-100:]))
        final_div_byT.append(np.mean(div_avg_this_T[-100:]))
        final_corepop_byT.append(np.mean(coresize_avg_this_T[-100:]))
        final_corediv_byT.append(np.mean(corediv_avg_this_T[-100:]))
        final_Jtot_byT.append(np.mean(Jtot_avg_this_T[-100:]))
        final_survival_byT.append(living_at_t_by_T[T_col][-1])
    print(np.shape(final_corepop_byT))
    print(np.shape(coresize_avg_by_T))

    # PLOT 10: final subplots: corepop, corediv, Jtot, input curve
    fig,ax = plt.subplots(2,2,sharex=True)
    # plt.suptitle(f"Avg. core characteristics of living ecosystems at {maxgens} gen \n{folder[:-1]} {experiment[:-1]}")
    # left col is core, right col is ecosystem
    ax[0,0].set_title("Core Avg.")
    ax[0,1].set_title("Ecosystem Avg.")

    # corepop
    #final_corepop_byT = np.array(coresize_avg_by_T)[:,-1]
    ax[0,0].plot(temperatures,final_corepop_byT)
    ax[0,0].set_ylabel("Abundance",color="b")
    #ax2 = plt.twinx(ax[0,0])
    #ax2.plot(temperatures,Teq.poff_T(temperatures) - Teq.pdeath(temperatures),label="TPC",color="red",alpha=.5)
    #ax2.set_ylim(0)
    #ax2.legend()
    
    # ecosyst pop
    ax[0,1].plot(temperatures,final_pop_byT)
 #   ax[0,1].set_ylabel("Population",color="b")
    #ax2 = plt.twinx(ax[0,1])
    #ax2.plot(temperatures,Teq.poff_T(temperatures) - Teq.pdeath(temperatures),label="TPC",color="red",alpha=.5)
    #ax2.set_ylim(0)
    #ax2.legend()

    # corediv
    #final_corediv_byT = np.array(corediv_avg_by_T)[:,-1]
    ax[1,0].plot(temperatures,final_corediv_byT)
    ax[1,0].set_ylabel("Diversity",color="b")
    #ax2 = plt.twinx(ax[1,0])
    #ax2.plot(temperatures,Teq.poff_T(temperatures) - Teq.pdeath(temperatures),label="TPC",color="red",alpha=.5)
    #ax2.set_ylim(0)
    #ax2.legend()
    
    # ecosyst div
    ax[1,1].plot(temperatures, final_div_byT)
#    ax[1,1].set_ylabel("Diversity",color="b")
    #ax2 = plt.twinx(ax[1,1])
    #ax2.plot(temperatures,Teq.poff_T(temperatures) - Teq.pdeath(temperatures),label="TPC",color="red",alpha=.5)
    #ax2.set_ylabel(r"Effect of temperature on growth, $r_{off} - r_{death}$",color="r") #"Expected $r_{max}$ ($\frac{\Delta N}{gen.}$)",color="r")
    #ax2.set_ylim(0)
    #ax2.legend()

    # Jtot
    #final_Jtot_byT = np.array(Jtot_avg_by_T)[:,-1]
    #ax[2,0].plot(temperatures,final_Jtot_byT)
    #ax[2,0].set_ylabel("Core-core interact.",color="b")
    #ax2 = plt.twinx(ax[2,0])
    #ax2.plot(temperatures,Teq.poff_T(temperatures) - Teq.pdeath(temperatures),label="TPC",color="red",alpha=.5)
    #ax2.set_ylim(0)
    #ax2.legend()
    
    # survival, and ylabel for input TPC
    #ax[2,1].plot(temperatures, living_at_t_by_T[-1,:]) # final_survival_byT)
    #ax[2,1].set_ylabel("# surviving ecosyst.s",color='b')
    ax[1,1].set_xlabel("Temperature,T (K)")
    ax[1,0].set_xlabel("Temperature,T (K)")

    #ax2 = plt.twinx(ax[2,1])
    #ax2.plot(temp_smooth,TPC_smooth,color="r",alpha=.5)
    #ax2.set_ylabel(r"$r_{off}-r_{death}$",color="r") #r"Effect of temperature on growth, $r_{off} - r_{death}$",color="r") #"Expected $r_{max}$ ($\frac{\Delta N}{gen.}$)",color="r")
    #ax2.set_ylim(0)
    #except:
    #    print("Couldn't find T curve")
    #fig.text(0.5,0.01,"Temperature,T (K)",ha="center",va="center")

    # label subplots
    axs = ax.flat

    for n, a in enumerate(axs):
        a.text(-0.1, 1.1, string.ascii_uppercase[n], transform=a.transAxes,size=15) #, weight='bold')

    plt.tight_layout()
    plt.savefig(figure_folder+f"TTNM_final_avg_"+fig_ending) #{folder[:-1]}_{experiment[:-1]}{extra_folder[:-1]}.pdf")


    # PLOT 11: Interactions by T
    # Jtot
    fig,ax = plt.subplots()
    #final_Jtot_byT = np.array(Jtot_avg_by_T)[:,-1]
    ax.plot(temperatures,final_Jtot_byT)
    ax.set_ylabel("Core-core interact.",color="b")
    ax.set_xlabel("Temperature,T (K)")
    ax.set_title(f"Avg interactions after {maxgens} gen.")

    plt.savefig(figure_folder+f"final_core_interactions_"+fig_ending) #{folder[:-1]}_{experiment[:-1]}{extra_folder[:-1]}.pdf")

if show_plots:
    plt.show()

