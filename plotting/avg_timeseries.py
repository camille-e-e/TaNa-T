# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:09:13 2022

@author: camil

average population over time

mar 19:
    - fixed steady state lengths plot
    - added output folder (needs to be edited upon running)
    - plot normalized average effect on each other (ouptu array remains the same)

@author: camil


** Should check what measures other people use.
"""
import os
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import statistics as stats
from scipy.signal import argrelextrema
import cProfile, pstats

# EDIT THESE DETAILS
# ------------------
# input: folder of experiments (should end in slash)
# folder = "C_ranJran_output" #'orignial_output\\' #'out_Mar18' ## folder full of experiments
folder = 'SteadyT/' # "BasicTNM/" #"output\\"
experiment = 'Sep_24/' # 'Jun_20/' # 'Jun_17/' #'c_outputMay18\\' #' 'py_outputMay17\\' # (requires slashes after) folder containing "folder" (above) if Palmieri output
extra_folder = 'poff_is_roff/' #'poff_is_roff/' # '' #one_ten_thousand/' # 'skew_norm_more_parameters/' # '' or '..../'
other_dates = ['Sep_' + str(i) for i in np.r_[25:30]] # list of possible dates in filenames other than that of experiment folder
# full path of input folder
# locat = './' + experiment + folder
# locat = 'C:\\Users\\camil\\projects\\TaNa_test\\TTNM_output\\' #'C:\\Users\\camil\\projects\\TaNa_test\\Figures_etc\\' + experiment + folder #May14_C_ranJran\\model_output\\'
locat = "/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/"
# locat = 'C:\\Users\\camil\\projects\\TaNa_test\\Figures_etc\\py_outputMay17\\'
locat = locat+folder+experiment+extra_folder # ends with /

multi_temp = True
want_plots = True # show plots or not (otherwise just save them, x11 not needed)
want2profile = False #True # profile or not
code = 'py' # "C" or "py"
maxgens = 50000 # maximum number of generations in experiments in folder

# STOP EDITTING
# -------------
# output file
output_folder = locat # 'C:\\Users\\camil\\projects\\TaNa_test\\TTNM_output\\' #TN_python\\py_output\\test_output' #'C:\\Users\\camil\\projects\\TaNa_test\\Figures_etc'  # './' if you want to put output folder in current directory 
# C:\Users\camil\projects\TaNa_test\TN_python\py_output\test_output

# figure out if we're on Windows or Linux
filepath = os.getcwd() 
today = date.today()
if filepath[0] == 'C': # windows
    filepath += '\\'
    system = 'windows'
else:
    filepath += '/' # linux
    system = 'linux'

# Make directory for all plots
if system == 'linux':
    figure_folder = output_folder + "Figures_"+folder
    if not folder.endswith('/'):
        figure_folder += '/'
elif system == 'windows':
    if output_folder == './':
        output_folder = os.getcwd()
    figure_folder = output_folder + "Figures_"+folder
    if not folder.endswith('\\'):
        figure_folder+='\\'
else:
    Exception("plotting directory not found: ",figure_folder)

if os.path.isdir(figure_folder) != True:
    os.mkdir(figure_folder)
    print("Making new folder")
print("Figure folder: ",figure_folder,"exists: ",os.path.exists(figure_folder))

if want2profile:
    profiler = cProfile.Profile()
    profiler.enable()

# Collect all output files you want to plot
whofiles = []
ecosystfiles = []
for filename in os.listdir(locat):
    if filename.endswith(".dat"):
        if filename.startswith("div"):
            whofiles.append(filename) # collect names of who data files
        elif filename.startswith("pypy"): ecosystfiles.append(filename) # collect names of files with overall ecosystem statistics


# %% Loop through ecosysyem files

# matrix of populations
popu_all = np.zeros((maxgens,len(ecosystfiles))) #sparse.lil_matrix((maxgens+1,len(ecosystfiles))) # huge matrix... may not be viable

# if not multi_temp, need to deal with these things which aren't currently being used!
# ------------
# vector of STD of population
#popu_STD = np.zeros((maxgens,2))
# sum across multiple runs: generations
#gen_sum = sparse.lil_matrix((maxgens,1))
# sum across multiple runs: populations
#popu_sum = sparse.lil_matrix((maxgens,1)) 

# Sums accross model runs
# gen_sum = sparse.lil_matrix((maxgens,1))
# popu_sum = sparse.lil_matrix((maxgens,1))
#div_sum = sparse.lil_matrix((maxgens,1))
#enc_sum = sparse.lil_matrix((maxgens,1))
#coresize_sum = sparse.lil_matrix((maxgens,1))
#corediv_sum = sparse.lil_matrix((maxgens,1))
#Jtot_sum = sparse.lil_matrix((maxgens,1))
    

seeds = set() # keep track of seeds to use in legend later
if multi_temp:
    print("Yooo it's multi temp")
    temperatures=set()    
    # track order of creation of pop_all etc
    order = []

# loop through all files in ecosystfiles and collect desired outputs
for file in ecosystfiles:
    filename = locat + file
    if os.path.isfile(filename) != True:
        print("Error, file doesn't exist")
        continue
    else: pass #print("File found", file)
    # make a name for plotting purposes
    name = file[:12]
    # Determine the seed used
    loc1 = filename.index('seed')
    try:
        loc2 = filename[loc1:].index(experiment[:-1])
    except:
        # if the experiment took longer than a day to run, the filename date won't match the folder date.  loop through other possible dates in the filename to check for file.
        for i in np.r_[0:len(other_dates)]:
            try:
                loc2 = filename[loc1:].index(other_dates[i])
                break
            except: continue
    seed = filename[loc1+4:loc1+loc2]
    seeds.add(seed)
    
    if multi_temp:
        # find temperature
        loc3 = filename[loc1+loc2:].index('K')
        temperature = filename[loc1+loc2+len(experiment):loc1+loc2+loc3]
        temperatures.add(int(float(temperature))) # set of ints
        # track order of inputs to data
        order.append([seed,temperature])
        
if multi_temp:
    # define arrays for each T
    popu_sum_by_T = {}
    div_sum_by_T = {}
    enc_sum_by_T = {}
    coresize_sum_by_T = {}
    corediv_sum_by_T = {}
    Jtot_sum_by_T = {}
    quakes_by_T = {} # dict by 'T'; each one has lists for all experiments each of quake times for each experiment
    quakecount_by_T = {} # each 'T' has a list of number of quakes in each experiment
    # number of living ecosystems at each t, by T
    living_at_t_by_T = np.zeros((maxgens,len(temperatures)))
    print("shape of living_at_t_by_T: ",np.shape(living_at_t_by_T))
        
    # convert set to sorted array
    temperatures = np.sort(list(temperatures)) # list of ints, sorted
    # convert list to iterable
    order_copy = order.copy()
    order = iter(order)
    
# check if any experiments got split into two files
# [[seed,T],[seed,T] ..... ] each of which has more than one filename associated
duplicates = [pair for pair in order_copy if order_copy.count(pair) > 1]
# make non-repeating list of duplicates
unique_dups = []
for dup in duplicates:
    if dup not in unique_dups:
        unique_dups.append(dup)
    
dup_IDs = []
# find location in list of filenames of two files with same seed and T
for pair in unique_dups:
    ID1 = order_copy.index(pair)
    ID2 = ID1 + 1 + order_copy[ID1+1:].index(pair)
    if order_copy[ID1] != order_copy[ID2]:
        print("OH NO! duplicate indexing not working!")
    dup_IDs.append([ID1,ID2])
    # !!! may want to check there aren't ever THREE files with same seed,T combos
    
# define blue white red colormap for plotting
bwr= plt.get_cmap('hot')
if multi_temp:
    colors = bwr(np.linspace(0,1,len(temperatures)))
else:
    colors = iter(bwr(np.linspace(0,1,len(ecosystfiles))))    
     
# abandon files of duplicates, but keep track of which files they were    
abandonedIDs = []
count = 0
n = -1
# keep track of which files can't be used because shape of data is wrong
unusual_files = []

# start loop again once seeds and temps sets are finished
for file in ecosystfiles:
    count += 1
    n += 1
    split_file = False # if the model output is split between two files
    filename = locat + file
    seed, temperature = next(order)
    if [seed,temperature] in unique_dups:
        # THIS IS A SPLIT UP FILE
        # figure out if this is first or second file, and get filename of other file
        for pair in dup_IDs:
            if n in pair:
                split_file = pair.index(n) + 1 # split_file = 1 if this is first file, or 2 if second
                #print("YAY ID FOUND! n = ",n)
                firstID = min(pair)
                secondID = max(pair)
                filename2 = ecosystfiles[secondID]
                break
                # (added one so it's either 1 or 2, not 0 (which is False))
        if split_file == False:
            print("OH NO! split file didn't find out if this is spinup or second chunk")
            print("n: ",n)
        if split_file == 2:
            # if this is the second file, ignore it; just deal with both files when dealing with the first one.
            print("Abandoning ID : ",pair[1]) 
            abandonedIDs.append(n)
            count -= 1
            continue
            
    # get ready to account for quakes at this T
    if str(temperature) not in quakes_by_T.keys():
        quakes_by_T[str(temperature)] = []
        quakecount_by_T[str(temperature)] = []
        
    # print("Seed: ",seed,", temperature: ",temperature)
    # Obtain model output 
    try:
        data = np.genfromtxt(filename, delimiter = ' ') # make 100,000 x 9 matrix of data for current model run    
        if split_file:
            data2 = np.genfromtxt(locat+filename2,delimiter = ' ')
            data = np.vstack([data,data2])
            if len(data) > maxgens: 
                print("OH NO, wrong data size for files ",pair,". Shape of data: ",np.shape(data))
                plt.plot(data[0:,][0:,0])
                plt.ylabel("self-identified generation")
    except:
        print('wrong size of input file, seed ',seed)
        unusual_files.append(f'Seed {seed}: wrong input file size: {np.shape(data)}, data not used')
        count -= 1
        continue
    # shape of data should be (9,max_gens). if not, extinction occurred.
    if len(np.shape(data)) < 2: # or np.shape(data)[0] != 9: # == (9,):
        if np.shape(data) == (9,):
            # if mass extinction after first generation,
            # fill entire dataset with zeros so this is taken
            # into averages etc
            data_tmp = np.zeros((maxgens,9))
            data_tmp[0] = data
            data = data_tmp.copy()
        else:                
            # complete extinction before first generation, go to next
            print("Not enough data (complete extinction before first generation)")
            print(np.shape(data))
            unusual_files.append(f"Seed {seed}: output shape: {np.shape(data)}")
            count -= 1
            continue #pass
        
    # Get info from "data" (model output)
    if len(data) > maxgens:
        unusual_files.append(f"Seed {seed}: file length: {len(data)}")
        data = data[:,][:maxgens]
    # The columns in Arthur data are: ['generation number','number of individuals','number of species', 'encountered.size()', 'individuals in core', 'species in core', 'effect on environment', 'effect on each other', 'external resources']
    gen = data[0:,][0:,0] # generations are in the first column
    popu = data[0:,][0:,1] # population in the second column

    # May  need to revise this section because when I merged master and variable_Tresponse, I didn't double check this section for duplicates *** !!!
    if len(popu) < maxgens:
        popu_long = np.zeros(maxgens)
        popu_long[:len(popu)] = popu.copy()
        #print("Length of popu is ",len(popu)," and lenght of popu_long is ",len(popu_long))
        popu_all[:,count-1] = popu_long
    elif len(popu) > maxgens:
        unusual_files.append(f"Seed {seed}: file length: {len(popu)}")
        # # truncate
        data = data[:,][:maxgens]
        gen = data[0:,][0:,0] # generations are in the first column
        popu = data[0:,][0:,1] # population in the second column
        popu_all[:,count-1] = np.reshape(popu,(1,len(popu)))
    else: # normal data size
        popu_all[:,count-1] = np.reshape(popu,(1,len(popu))) # matrix of all populations (for STD later)
    div = data[0:,][0:maxgens,2] # diversity in the third column
    enc = data[0:,][0:maxgens,3] # number of species encountered
    core_size = data[0:,][0:maxgens,4] # number of indiv in core
    core_div = data[0:,][0:maxgens,5] # number of spc in core
    Jtot = data[0:,][0:maxgens,7] # overall effect on each other            
    
    # How long is this ecosystem alive?
    if len(popu) > 0:
        T_column = list(temperatures).index(int(temperature))
        living_at_t_by_T[:len(popu),T_column] += np.array(np.array(popu)>0,dtype=int) #np.ones(len(popu))

    # Find quakes
    method = 1
    core_2to5 = [0]
    for t in range(len(core_div)):
        if core_div[t] < 6 and core_div[t] >= 2:
            if method == 1: core_2to5.append(1)
            elif method == 2:
                if core_div[t-1] == core_div[t]:
                    core_2to5.append(1)
                else: core_2to5.append(0)
        else: core_2to5.append(0)
    #print("total time in SS: ",sum(core_2to5))
    #print(core_2to5)
    smoothed = [0]
    quakes = []
    quake_count = 0
    for u in range(20,len(div)):
        if sum(core_2to5[u-20:u])==20: # past 20 generations all in SS
            smoothed.append(1)
            if smoothed[-2] == 0: # if 20 generations ago was the beginning of an SS, make note
                quake_count+=1
                quakes.append(u-20)
        else: 
            smoothed.append(0)
    #print("quakes detected: ",quake_count)
    # append quake times to quakes_by_T
    quakes_by_T[str(temperature)].append(quakes)
    quakecount_by_T[str(temperature)].append(quake_count)
    
    # put "data" from this file into column of matrix of all "data"
    # if len(div) < maxgens:
    #     difference = maxgens-len(div)
    #     popu = np.array(list(popu) + list(np.zeros(difference,int)))
    #     div = np.array(list(div) + list(np.zeros(difference,int)))
    #     enc = np.array(list(enc) + list(np.zeros(difference,int)))
    #     core_size = np.array(list(core_size) + list(np.zeros(difference,int)))
    #     core_div = np.array(list(core_div) + list(np.zeros(difference,int)))
    #     Jtot = np.array(list(Jtot) + list(np.zeros(difference,int)))

    # put populations from this file into column of matrix of all populations
    popu_all[:len(popu),n] = popu
    # popu_all[:,n] = np.reshape(popu,(1,len(popu))) # matrix of all populations (for STD later)
    
    # plot individual runs on same plot
    if multi_temp:
        c = colors[list(temperatures).index(float(temperature))]
        # print("Plotting temperature ",temperature,c)
    else: 
        c = next(colors)
    plt.plot(popu, color=c)
    # plt.ylim(0,20000)
    # print("plot for T = ",temperature,seed)
    plt.title(f"{extra_folder[:-1]} all runs")
    # plt.title("Seeds %s"%str(seeds))
    # plt.legend(seeds)

    # *** Do I need those other lines of code above? ***
    popu = popu_all[:,n]
    if split_file==2:
        div = np.array(list(np.zeros(begin,int)) + list(div))
        enc = np.array(list(np.zeros(begin,int)) + list(enc))
        core_size = np.array(list(np.zeros(begin,int)) + list(core_size))
        core_div = np.array(list(np.zeros(begin,int)) + list(core_div))
        Jtot = np.array(list(np.zeros(begin,int)) + list(Jtot))                                  
    
    if len(div) < maxgens:
        difference = maxgens-len(div)
        #popu = np.array(list(popu) + list(np.zeros(difference,int)))        
        div = np.array(list(div) + list(np.zeros(difference,int)))    
        enc = np.array(list(enc) + list(np.zeros(difference,int)))        
        core_size = np.array(list(core_size) + list(np.zeros(difference,int)))        
        core_div = np.array(list(core_div) + list(np.zeros(difference,int)))        
        Jtot = np.array(list(Jtot) + list(np.zeros(difference,int)))        
   # print("shapes of outputs:")
   # print(np.shape(popu),np.shape(div),np.shape(enc),np.shape(core_size),np.shape(core_div),np.shape(Jtot))
    
    # Sum the current seed with the rest of the experiments    
    if multi_temp:
        if str(temperature) not in popu_sum_by_T.keys():
            print("new key for ",temperature)
            # if this T is not in the dictionaries yet, set zeros of the right size
            popu_sum_by_T[str(temperature)] = np.zeros((maxgens,1))
            div_sum_by_T[str(temperature)] = np.zeros((maxgens,1))
            enc_sum_by_T[str(temperature)] = np.zeros((maxgens,1))
            coresize_sum_by_T[str(temperature)] = np.zeros((maxgens,1))
            corediv_sum_by_T[str(temperature)] = np.zeros((maxgens,1))
            Jtot_sum_by_T[str(temperature)] = np.zeros((maxgens,1))
        #print("Popu_sum_by_T for ",temperature)
        popu_sum_by_T[str(temperature)] += np.reshape(popu,(len(popu),1)) # add populations from this experiment to total population at each generation
        div_sum_by_T[str(temperature)] += np.reshape(div,(len(div),1)) # add diversity from this experiment at each timepoint
        enc_sum_by_T[str(temperature)] += np.reshape(enc,(len(div),1))
        coresize_sum_by_T[str(temperature)] += np.reshape(core_size,(len(div),1))
        corediv_sum_by_T[str(temperature)] += np.reshape(core_div,(len(div),1))
        Jtot_sum_by_T[str(temperature)] += np.reshape(Jtot,(len(div),1))

        if [np.shape(popu_sum_by_T[str(temperature)]),np.shape(div_sum_by_T[str(temperature)]),np.shape(enc_sum_by_T[str(temperature)]),np.shape(coresize_sum_by_T[str(temperature)]),np.shape(corediv_sum_by_T[str(temperature)]),np.shape(Jtot_sum_by_T[str(temperature)])] != [(10000, 1), (10000, 1), (10000, 1), (10000, 1), (10000, 1), (10000, 1)] : 
            print(np.shape(popu_sum_by_T[str(temperature)]),np.shape(div_sum_by_T[str(temperature)]),np.shape(enc_sum_by_T[str(temperature)]),np.shape(coresize_sum_by_T[str(temperature)]),np.shape(corediv_sum_by_T[str(temperature)]),np.shape(Jtot_sum_by_T[str(temperature)]))

            
    else:
        # i feel like I should not index this with gen, which might not be continuous.  check later *** !!!
        popu_sum[gen-1,:] += np.reshape(popu,(len(popu),1)) # add populations from this experiment to total population at each generation
        div_sum[gen-1,:] += np.reshape(div,(len(div),1)) # add diversity from this experiment at each timepoint
        enc_sum[gen-1,:] += np.reshape(enc,(len(div),1))
        coresize_sum[gen-1,:] += np.reshape(core_size,(len(div),1))
        corediv_sum[gen-1,:] += np.reshape(core_div,(len(div),1))
        Jtot_sum[gen-1,:] += np.reshape(Jtot,(len(div),1))

#  Equilibria statistics
# average standard deviation in sliding window 
ecosystem_population = np.array(popu)

# %% process quake stuff
if multi_temp:
    
    # Quake timing for each experiment
    quacks_by_T = []
    for T in temperatures:
        quacks = []
        for results in quakes_by_T[str(T)]:
            #plt.scatter(results,np.ones(len(results)),color=colors[list(temperatures).index(T)])
            quacks+= list(results)
        quacks_by_T.append(quacks)
    #plt.show()
    plt.figure()

    try:
        plt.violinplot(quacks_by_T,positions=temperatures,widths=3,showmeans=True,showextrema=False)
    except:
        for T in range(len(temperatures)):
            if len(quacks_by_T[T])>1:
                plt.violinplot(quacks_by_T[T],positions=[temperatures[T]],widths=3,showextrema=False,showmeans=True)
    plt.xlabel("Temperature,T (K)")
    plt.ylabel("Quake times (gen.)")
    plt.title(f"Timing of quakes depending on temperature, {folder[:-1]}")
    plt.savefig(figure_folder+f'/quake_times{experiment[:-1]}.pdf')
    
    # Quake counts per experiment
    quake_count_avg = []
    plt.figure()
    for T in temperatures:
        quake_count_avg.append(sum(quakecount_by_T[str(T)]))
    plt.scatter(temperatures,quake_count_avg)
    plt.xlabel("Temperature,T (K)")
    plt.ylabel("Avg. number of quakes per experiment")
    plt.title(f"Avg number of quakes at different temperatures, {folder[:-1]}")
    
    plt.figure()
    for T in temperatures:
        plt.violinplot(quakecount_by_T[str(T)],positions=[T],widths=3,showmeans=True) #showextrema=False,
    plt.ylabel("Number of quakes per experiment")
    plt.xlabel("Temperature,T (K)")
    plt.title(f"Quake Frequency at Different Temperatures, {folder[:-1]}")
    plt.savefig(figure_folder+f'/quake_counts_byT{experiment[:-1]}.pdf')
    
    # Steady State Duration
    SS_by_T = []
    for T in temperatures:
        SS_this_T = []
        for results in quakes_by_T[str(T)]:
            if len(results) > 1:
                SS = []
                for quake in range(1,len(results)):
                    SS.append(results[quake]-results[quake-1])
                SS_this_T+=SS
            else: SS_this_T.append(0)
        SS_by_T.append(SS_this_T) 
        # SS_by_T = [SS_this_T1, SS_this_T2, .... , SS_this_Tn]
                # = [[SS_this_seed1, SS_this_seed2, SS_this_seed3, ... , SS_this_seedn], [SS_this_seed1, SS_this_seed2, ... , SS_this_seedn], ..... , [SS_this_seed1, SS_this_seed2, ... , SS_this_seedn]]
                # = [[[SS1, SS2, SS3, ... , SSn , SS1, SS2, ... , SSn, ..... , SS1, SS2, ... , SSn]],
                    #[[SS1, SS2, SS3, ... , SSn, SS1, SS2, ... , SSn, ..... , SS1, SS2, ... , SSn]], 
                    # ....
                    #[[SS1, SS2, SS3, ... , SSn, SS1, SS2, ... , SSn, ..... , SS1, SS2, ... , SSn]]]
        
    plt.figure()
    for T in range(len(temperatures)):
        plt.violinplot(SS_by_T[T],positions=[temperatures[T]],widths=3,showmeans=True)
    plt.ylabel("SS durations per experiment")
    plt.xlabel("Temperature,T (K)")
    plt.title(f"Steady State Duration at Different Temperatures, {folder[:-1]}")
    plt.savefig(figure_folder+f'/SS_duration{experiment[:-1]}.pdf')


# %% average output
if multi_temp:
    popu_avg_by_T,div_avg_by_T,enc_avg_by_T,coresize_avg_by_T,corediv_avg_by_T,Jtot_avg_by_T = [],[],[],[],[],[]
    # loops through temperatures in ascending order
    plt.figure()
    cmap = plt.get_cmap('hot')
    colors = cmap(np.linspace(1,0,len(temperatures)))
    for T in temperatures:
        T_col = list(temperatures).index(T)
        living_at_t = living_at_t_by_T[:,T_col]
        
        if 0 in living_at_t:
            first0 = list(living_at_t).index(0)
        else: first0 = len(living_at_t)
        """
        popu_avg_by_T.append([])
        # only consider living populations:
        living_at_t = []
        for t in range(maxgens):
            living_at_t.append(len(popu_all[t,:].nonzero()[0]))
            """

        popu_avg_by_T.append(popu_sum_by_T[str(T)][:first0]/living_at_t[:first0]) # len(seeds))
        div_avg_by_T.append(div_sum_by_T[str(T)][:first0]/living_at_t[:first0]) # len(seeds))
        enc_avg_by_T.append(enc_sum_by_T[str(T)][:first0]/living_at_t[:first0]) # len(seeds))
        coresize_avg_by_T.append(coresize_sum_by_T[str(T)][:first0]/living_at_t[:first0]) # len(seeds))
        corediv_avg_by_T.append(corediv_sum_by_T[str(T)][:first0]/living_at_t[:first0]) # len(seeds))
        Jtot_avg_by_T.append(Jtot_sum_by_T[str(T)][:first0]/living_at_t[:first0]) # len(seeds))

        plt.plot(living_at_t,label=f"T={T}",color=colors[T_col])
    plt.legend()
    plt.title("Number of ecosystems alive by time and temp.")
    plt.xlabel("Time,t (gen.)")
    plt.ylabel("Number of living ecosystems")
    plt.savefig(figure_folder+"/living_ecosystems.png")
else:
    popu_avg = np.mean(popu_all,axis=1) #popu_sum.toarray()/count
    popu_med = np.median(popu_all,axis=1)
    print("Count = ",count)
    # upper and lower quartiles of population
    medval = np.zeros(maxgens)
    for g in range(maxgens):
        vals = popu_all[g].copy()
        if sum(vals) == 0:
            break
        non0s = max(np.nonzero(vals)[0]) # highest index with value in it
        # vals = vals[:non0s] 
        vals.sort()
        min_tmp = .25*len(vals)
        if np.floor(min_tmp) != np.ceil(min_tmp):
            minval1 = vals[int(np.floor(min_tmp))]
            minval2 = vals[int(np.ceil(min_tmp))]
            minval = np.mean((minval1,minval2))
        else:
            minval = vals[int(min_tmp)]
        max_tmp = .75*len(vals)
        maxval = vals[int(max_tmp)]
        med_tmp = .5*len(vals)
        medval[g] = vals[int(med_tmp)]
        popu_STD[g] = [minval,maxval]
    #popu_STD = np.array(popu_all).std(0) # std along 1st axis (std of each generation)
    # !!! *** ###
    # div_sum isn't edited after being pre-defined at top, so these things are all empty.
    div_avg = div_sum.toarray()/count
    enc_avg = enc_sum.toarray()/count
    coresize_avg = coresize_sum.toarray()/count
    corediv_avg = corediv_sum.toarray()/count
    Jtot_avg = Jtot_sum.toarray()/count

# %% Plots
  
if not multi_temp:
    # Savre arrays for experiment averages    
    # Population average
    output_file = figure_folder+'popu_av_arr_'+folder[:-1]
    np.save(output_file,popu_avg)
    # Population median
    output_file = figure_folder+'popu_med_arr_'+folder[:-1]
    np.save(output_file,popu_med)
    # Population STD
    output_file = figure_folder+'popu_std_arr_'+folder[:-1]
    np.save(output_file,popu_STD)
    # Diversity average
    output_file = figure_folder+'div_avg_arr_'+folder[:-1]
    np.save(output_file,div_avg)
    # Total interaction strength averaged
    output_file = figure_folder+'Jtot_avg_arr_'+folder[:-1]
    np.save(output_file,Jtot_avg)
    # # Quake times (all)
    # output_file = figure_folder+'quake_times_arr_'+folder[:-1]
    # np.save(output_file,quake_times)
    # # Steady state lengths
    # output_file= figure_folder+'qESS_lengths_arr_'+folder[:-1]
    # np.save(output_file,quake_lengths)

# 1., 2. Average population size and diversity over time
if not multi_temp:
    fig, ax = plt.subplots() 
    ax.set_xscale('log')
    # plt.title('%s: Avg pop\'n and div\'ty of ecosystems over %d experiments'%(folder,count))
    plt.title(f'{folder} {extra_folder}: avg pop\'n and div\'ty of ecosystems over {count} experiments')
    
    # ax.errorbar(range(len(popu_STD)),popu_avg[:-(len(popu_avg)-len(popu_STD))], yerr = popu_STD)
    ax.plot(popu_avg,'black')
    ax.plot(popu_STD[:,0],':b') # lower quartile
    ax.plot(popu_STD[:,1],':g') # upper quartile
    ax.plot(medval,'--r') # median
    ax.legend(['Average Population Size','Lower Quartile','Upper Quartile','Median'])
    # ax.set_ylim(0,4*max(popu_avg))
    # how to get log scale with ax?
    ax.set_xlabel('Log of generations', color = 'r') 
    ax.set_ylabel('Ecosystem Size (individuals)', color = 'black') 
    
    # # using the twinx() for creating another axes object for secondry y-Axis 
    # ax2 = ax.twinx()
    # ax2.plot(div_avg,'r')
    # # secondary y-axis label 
    # ax2.set_ylabel('Diversity (species in ecosystem)', color = 'r')
    plt.figure()
    plt.plot(div_avg)
    plt.title("Average Diversity over %d Experiments"%count)
    plt.tight_layout()
    # save figure in Figure Folder
    figname = figure_folder+'popu_div_avg_'+folder[:-1]+'.pdf'
    plt.savefig(figname,format='pdf')
else: # (multi_temp:)** START HERE CAMILLE MARCH 28 22
    fig, ax = plt.subplots() 
    # ax.set_xscale('log')
    plt.title('Avg popln of ecosystems over time \n(%d seeds, %s, %s)'%(count/len(temperatures),folder[:-1],experiment[:-1])) #extra_folder))
    
    for T in range(len(temperatures)):
        c = colors[list(temperatures).index(float(temperatures[T]))]
        # ax.errorbar(range(len(popu_STD)),popu_avg[:-(len(popu_avg)-len(popu_STD))], yerr = popu_STD)
        ax.plot(popu_avg_by_T[T],label=str(temperatures[T]),color=c)

    ax.legend(bbox_to_anchor=(1,1),loc='upper left') #loc='center')
    # ax.legend(['Average Population Size','Lower Quartile','Upper Quartile','Median'])
    # ax.set_ylim(0,4*max(popu_avg))
    # how to get log scale with ax?
    ax.set_xlabel('Time (generations)', color = 'r') 
    ax.set_ylabel('N. indiv.s in ecosystem (avg w/in T)', color = 'black') 
    plt.tight_layout() 
    plt.savefig(figure_folder+f'/populations_timeseries{experiment[:-1]}.pdf')
    # # using the twinx() for creating another axes object for secondry y-Axis 
    # ax2 = ax.twinx()
    # ax2.plot(div_avg,'r')
    # # secondary y-axis label 
    # ax2.set_ylabel('Diversity (species in ecosystem)', color = 'r')
    plt.figure()
    for T in range(len(temperatures)):
        c = colors[list(temperatures).index(float(temperatures[T]))]
        plt.plot(div_avg_by_T[T],label=str(temperatures[T]),color=c)
    # plt.ylim(1,5000)
    plt.xlabel("Time (generations)")
    plt.ylabel("N. species in ecosystem (avg w/in T)")
    plt.legend(bbox_to_anchor=(1,1),loc='upper left')  #loc='center')
    plt.title("Average diversity over time \n(%d seeds, %s, %s)"%(count/len(temperatures),folder[:-1],experiment[:-1])) #extra_folder))
    plt.tight_layout()
    # save figure in Figure Folder
    #figname = figure_folder+'popu_div_avg_'+experiment[:-1]+'.pdf'
    #plt.savefig(figname,format='pdf')
    plt.savefig(figure_folder+"/diversity_timeseries_"+experiment[:-1]+".pdf")
        
    #plt.figure()
    # plt.ylim(0,2000)
    #i = -1
    #for j in popu_avg_by_T:
    #    i += 1
    #    plt.scatter(i,j[-1])
    ## plt.ylim(0,2000)
    #plt.title(experiment)
    #plt.ylabel(f"Avg popln size after {maxgens} gens")
    #plt.xlabel("Time (generations)")
    #plt.tight_layout()
    #plt.savefig(figure_folder+"diversity_timeseries_"+experiment[:-1]+".pdf")



# 3. Average interaction type
if not multi_temp:
    plt.figure()
    plt.plot(Jtot_avg[:,0]/popu_avg)#[:40000])
    plt.xscale('log')
    plt.xlabel('Log of generations')
    plt.ylabel("Avg effect on eachother (normalized to population size)")
    plt.title('Avg normalized interactions of ecosystems: \n(%d seeds, %s, %s)'%(count,folder[:-1],experiment[:-1]))
    figname = figure_folder+'Jtot_avg_normalized_'+folder[:-1]+'.pdf'
    plt.tight_layout()
    plt.savefig(figname,format='pdf')
    #output_file = './'+figure_folder+'/Jtot_avg_arr'+folder
    #np.save(output_file, Jtot_avg)
else:
    fig, ax = plt.subplots() 
    # ax.set_xscale('log')
    plt.title('Avg normalized interactions of ecosystems: \n(%d seeds, %s, %s)'%(count/len(temperatures),folder[:-1],experiment[:-1]))
    
    for T in range(len(temperatures)):
        c = colors[list(temperatures).index(float(temperatures[T]))]
        # ax.errorbar(range(len(popu_STD)),popu_avg[:-(len(popu_avg)-len(popu_STD))], yerr = popu_STD)
        ax.plot(Jtot_avg_by_T[T],label=str(temperatures[T]),color=c)

    #ax.legend(ncol=3,loc='lower right')
    ax.legend(bbox_to_anchor=(1,1),loc='upper left')
    # ax.legend(['Average Population Size','Lower Quartile','Upper Quartile','Median'])
    # ax.set_ylim(0,4*max(popu_avg))
    # how to get log scale with ax?
    ax.set_xlabel('Time (generations)') 
    ax.set_ylabel('Avg interaction in ecosystem at T', color = 'black') 
    plt.tight_layout()
    plt.savefig(figure_folder+"/interaction_timeseries_"+experiment[:-1]+".pdf")

if want_plots:
    plt.show()

# 3.b. Average final interaction strength distribution
# plt.hist(np.reshape(Jtot,(len(div),1)))
# plt.title("Final interaction strengths after %s generations"%maxgens)
# plt.xlabel("Interaction Strength")
# plt.ylabel("Number of occurances")

# # # 4. Average duration of steady states
# # # times of all steady quakes overlaid on average population
# # plt.figure()
# # jet= plt.get_cmap('jet')
# # colors = iter(jet(np.linspace(0,1,count)))
# # marker_height = int(np.max(popu_avg)/4)
# # for quakes in quake_times:
# #     c = next(colors)
# #     for i in quakes:
# #         plt.plot(i*np.ones(marker_height),range(-marker_height,0),color=c) #,ls='dotted')
# #         # plt.plot(ecosystem_population[:,1])
# # plt.plot(popu_avg)
# # plt.xscale('log')
# # plt.title('Average Ecosystem Size and Quake Times')
# # plt.xlabel("Log of Generations")
# # plt.ylabel("Ecosystem Population")
# # figname = figure_folder+'quakes_avg_'+folder[:-1]+'.pdf'
# # plt.savefig(figname,format='pdf')
# # #output_file = './'+figure_folder+'/quake_times_arr'+folder
# # #np.save(output_file, quake_times)
# # if want_plots:
# #     plt.show()

# # print("There were an average of ",np.mean(quake_counts)," quakes per experiment\nSTD = ",np.std(quake_counts))

# # distribution of quake times
# plt.figure()
# # quake_times_reshaped = []
# # # for i in quake_times:
# #     # quake_times_reshaped.append(i)
# # flatquakes_tmp = np.array(quake_times).flatten() #_reshaped
# # flatquakes = np.array(quake_times).flatten()
# # # flatquakes = []
# # # for i in flatquakes_tmp:
# # #     if i != []:
# # #         for j in i:
# # #             flatquakes.append(j)
# plt.hist(quake_times) #, weights=np.ones(len(quake_times)) / len(quake_times))
# plt.title("Distribution of Quake Times over %d experiments"%(count))
# plt.ylabel("Occurances")
# plt.xlabel("Generation at which Quake Occured")
# # plt.legend(seeds, bbox_to_anchor=(1,1), loc="upper left")
# figname = figure_folder+'quakes_dist_'+folder[:-1]+'.pdf'
# plt.savefig(figname,format='pdf')
# if want_plots:
#     plt.show()

# #quake lengths
# plt.figure()
# jet= plt.get_cmap('jet')
# colors = iter(jet(np.linspace(0,1,count)))
# for i in range(len(quake_times)):
#     times = quake_times[i]
#     if len(times) < len(quake_lengths[i]):
#         times = np.append(times,maxgens)
#     if len(times) != len(quake_lengths[i]):
#         print("Length of times: ",len(times),", Length of quake_lengths: ",len(quake_lengths))
#     plt.scatter(times,quake_lengths[i],color = next(colors))
# plt.title('Steady state lengths')
# plt.xlabel("Generation at which steady state ended")
# plt.ylabel("Number of generations steady state lasted")
# plt.xscale('log')
# # plt.legend(seeds, bbox_to_anchor=(1,1), loc="upper left")
# figname = figure_folder+'steadystates_'+folder[:-1]+'.pdf'
# plt.savefig(figname,format='pdf')
# #output_file = './'+figure_folder+'/qESS_lengths_arr'+folder
# #np.save(output_file, quake_lengths)
# if want_plots:
#     plt.show()

# # print(os.getcwd())

# # 5. Variation within steady states

# # %% Profile 2
# if want2profile:
#     statOUT = figure_folder +'/timer'+folder[:-1]+'.dat'
#     stats = pstats.Stats(profiler).sort_stats('cumtime')
#     #stats.dump_stats(statOUT)
#     profiler.disable()
#     stats.print_stats(statOUT)

# # %% Write output to file
# # variables = [popu_avg, div_avg, enc_avg, coresize_avg, corediv_avg, Jtot_avg, flatquakes]
# # counter = -1
# # for file in files:
# #     counter += 1
# #     with open file as f:
# #         f.save(file, variables[i])
    

# # with open file as f:
# #     f.write(popu_avg \n div_avg \n enc_avg \n Jtot_avg \n coresize_avg \n corediv_avg \n flatquakes)


# print(unusual_files)
