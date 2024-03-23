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
# input: folder of experiments (should end in slash)
# folder = "C_ranJran_output" #'orignial_output\\' #'out_Mar18' ## folder full of experiments
folder = "SpunupT/" #"output\\"
experiment = 'Jul_05/' #'c_outputMay18\\' #' 'py_outputMay17\\' # (requires slashes after) folder containing "folder" (above) if Palmieri output
extra_folder = 'poff_is_roff/' # 'skew_norm_more_parameters/' # '' or '..../'
other_dates = ['Jul_06','Jul_07','Jul_08','Jul_09','Jul_10','Jul_11'] # list of possible dates in filenames other than that of experiment folder
# full path of input folder
# locat = './' + experiment + folder
# locat = 'C:\\Users\\camil\\projects\\TaNa_test\\TTNM_output\\' #'C:\\Users\\camil\\projects\\TaNa_test\\Figures_etc\\' + experiment + folder #May14_C_ranJran\\model_output\\'
locat = "/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/"
# locat = 'C:\\Users\\camil\\projects\\TaNa_test\\Figures_etc\\py_outputMay17\\'
locat = locat+folder+experiment+extra_folder # ends with /

multi_temp = True
want_plots = True
want2profile = False #True
code = 'py' # "C" or "py"
maxgens = 50_000 # maximum number of generations in experiments in folder

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

# # name file today's date and the input folder name
# filename = 'plots_' + today.strftime("%b_%d_") + folder + '.dat'
# files = []
# for i in range(7):
#     file = filepath+filename+str(i)
#     files.append(file)
    
# %% Profile
if want2profile:
#for i in range(1,3):
    profiler = cProfile.Profile()
    profiler.enable()
    # input = seed, path, tag(optional)

# Arthur style output
whofiles = []
ecosystfiles = []
for filename in os.listdir(locat):
    if filename.endswith(".dat"):
        if filename.startswith("who"):
            whofiles.append(filename) # collect names of who data files
        elif filename.startswith("pypy"): ecosystfiles.append(filename) # collect names of files with overall ecosystem statistics


# %% Loop through ecosysyem files

# os.chdir("C:\\Users\\camil\\projects\\TaNa_test\\TN_c\\data_files\\")
# locat = "C:\\Users\\camil\\projects\\TaNa_test\\TN_c\\data_files\\"

# matrix of populations
popu_all = np.zeros((maxgens,len(ecosystfiles))) #sparse.lil_matrix((maxgens+1,len(ecosystfiles))) # huge matrix... may not be viable
# vector of STD of population
popu_STD = np.zeros((maxgens,2))
# sum across multiple runs: generations
gen_sum = sparse.lil_matrix((maxgens,1))
# sum across multiple runs: populations
popu_sum = sparse.lil_matrix((maxgens,1)) 

# Sums accross model runs
# gen_sum = sparse.lil_matrix((maxgens,1))
# popu_sum = sparse.lil_matrix((maxgens,1))
div_sum = sparse.lil_matrix((maxgens,1))
enc_sum = sparse.lil_matrix((maxgens,1))
coresize_sum = sparse.lil_matrix((maxgens,1))
corediv_sum = sparse.lil_matrix((maxgens,1))
Jtot_sum = sparse.lil_matrix((maxgens,1))
    
# quakes
quake_counts = [] # how many quakes occured in each experiment
quake_times = [] # generations at which quake occurred
quake_lengths = [] # duration of steady states
seeds = set() # keep track of seeds to use in legend later
if multi_temp:
    print("Yooo it's multi temp")
    temperatures=set()    
    # track order of creation of pop_all etc
    order = []
    print("Type of temperatures:",type(temperatures))
 
count = 0

# keep track of which files can't be used because shape of data is wrong
unusual_files = []
    
#for seed in [0,1,2,3,4,5,6,7,8,9,10,11,12,20,21,22,23,24,25,26,27,28,29,30,40,41,42,43,44,45,46,47,48,60,61,62,63,64,65,66,67,68,69,80,81,82,83,85,85,86,87,88]: #range(199):
#for seed in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,20,21,22,23,24,25,26,27,28,29,30]:
# for seed in [100,101,201,200,201,202,300,301,302]:
for file in ecosystfiles:
    count += 1
    filename = locat + file
    if os.path.isfile(filename) != True:
        print("Error, file doesn't exist")
        continue
    else: print("File found", file)
    # make a name for plotting purposes
    name = file[:12]
    # Determine the seed used
    loc1 = filename.index('seed')
    try:
        loc2 = filename[loc1:].index(experiment[:-1])
    except:
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
        temperatures.add(int(temperature))
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
        
    # convert set to sorted array
    temperatures = np.sort(list(temperatures))
    # convert list to iterable
    order = iter(order)
    
bwr= plt.get_cmap('bwr')
if multi_temp:
    colors = bwr(np.linspace(0,1,len(temperatures)))
else:
    colors = iter(bwr(np.linspace(0,1,len(ecosystfiles))))    
        
# start loop again once seeds and temps sets are finished
for file in ecosystfiles:
    filename = locat + file
    seed, temperature = next(order)
    # print("Seed: ",seed,", temperature: ",temperature)
    # Obtain model output 
    try:
        data = np.genfromtxt(filename, delimiter = ' ') # make 100,000 x 9 matrix of data for current model run    
    except:
        print('wrong size of input file, seed ',seed)
        unusual_files.append(f'Seed {seed}: wrong input file size: {np.shape(data)}, data not used')
        count -= 1
        continue
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
        
    # The columns in Arthur data are: ['generation number','number of individuals','number of species', 'encountered.size()', 'individuals in core', 'species in core', 'effect on environment', 'effect on each other', 'external resources']
    gen = data[0:,][0:,0] # generations are in the first column
    popu = data[0:,][0:,1] # population in the second column
    if len(popu) < maxgens:
        popu_long = np.zeros(maxgens)
        popu_long[:len(popu)] = popu.copy()
        print("Length of popu is ",len(popu)," and lenght of popu_long is ",len(popu_long))
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
    
    # plot individual runs on same plot
    if multi_temp:
        c = colors[list(temperatures).index(int(temperature))]
        # print("Plotting temperature ",temperature,c)
    else: 
        c = next(colors)
    plt.plot(popu, color=c)
    # plt.ylim(0,20000)
    # print("plot for T = ",temperature,seed)
    plt.title(f"{extra_folder[:-1]} all runs")
    # plt.title("Seeds %s"%str(seeds))
    # plt.legend(seeds)

    if len(popu) < maxgens:
        difference = maxgens-len(popu)
        popu = np.array(list(popu) + list(np.zeros(difference,int)))        
        div = np.array(list(div) + list(np.zeros(difference,int)))    
        enc = np.array(list(enc) + list(np.zeros(difference,int)))        
        core_size = np.array(list(core_size) + list(np.zeros(difference,int)))        
        core_div = np.array(list(core_div) + list(np.zeros(difference,int)))        
        Jtot = np.array(list(Jtot) + list(np.zeros(difference,int)))        

    # Sum the current seed with the rest of the experiments    
    # gen_sum[gen,:] += np.ones((len(gen),1)) # add generations (important if you want results for ecosystems while they're alive only)
    if multi_temp:
        # for o in order:
        #     s,T = o
        #     print(s,T)
            # T_index = temperatures.index(T)
        if str(temperature) not in popu_sum_by_T.keys():
            print("new key for ",temperature)
            # if this T is not in the dictionaries yet, set zeros of the right size
            popu_sum_by_T[str(temperature)] = np.zeros((maxgens,1))
            div_sum_by_T[str(temperature)] = np.zeros((maxgens,1))
            enc_sum_by_T[str(temperature)] = np.zeros((maxgens,1))
            coresize_sum_by_T[str(temperature)] = np.zeros((maxgens,1))
            corediv_sum_by_T[str(temperature)] = np.zeros((maxgens,1))
            Jtot_sum_by_T[str(temperature)] = np.zeros((maxgens,1))
        print("Popu_sum_by_T for ",temperature)
        popu_sum_by_T[str(temperature)] += np.reshape(popu,(len(popu),1)) # add populations from this experiment to total population at each generation
        div_sum_by_T[str(temperature)] += np.reshape(div,(len(div),1)) # add diversity from this experiment at each timepoint
        enc_sum_by_T[str(temperature)] += np.reshape(enc,(len(div),1))
        coresize_sum_by_T[str(temperature)] += np.reshape(core_size,(len(div),1))
        corediv_sum_by_T[str(temperature)] += np.reshape(core_div,(len(div),1))
        Jtot_sum_by_T[str(temperature)] += np.reshape(Jtot,(len(div),1))

            
    else:
        popu_sum[gen-1,:] += np.reshape(popu,(len(popu),1)) # add populations from this experiment to total population at each generation
        div_sum[gen-1,:] += np.reshape(div,(len(div),1)) # add diversity from this experiment at each timepoint
        enc_sum[gen-1,:] += np.reshape(enc,(len(div),1))
        coresize_sum[gen-1,:] += np.reshape(core_size,(len(div),1))
        corediv_sum[gen-1,:] += np.reshape(core_div,(len(div),1))
        Jtot_sum[gen-1,:] += np.reshape(Jtot,(len(div),1))

#  Equilibria statistics
# average standard deviation in sliding window 
ecosystem_population = np.array(popu)
std_N = []
mean_N = []
window = 500
i = 0

# # --------------insert Mar 15 ----------------
#  # coarse smoothing is ok
# for i in range(len(ecosystem_population))[::window]:
#     group = ecosystem_population[i:i+window]
#     group_mean = np.mean(group)
#     mean_N.append(group_mean)
    
# std_of_coarse_mean = []
# little_window = 5
# for i in range(len(mean_N)-little_window):
#     std_group = stats.stdev(mean_N[i:i+little_window])
#     std_of_coarse_mean.append(std_group)  
    
# # Find local maxima in stadard deviation
# maxima_idx = argrelextrema(np.array(std_of_coarse_mean), np.greater)

# # make sure there are maxima
# if len(maxima_idx) > 0:
#     maxima_vals = np.array(std_of_coarse_mean)[maxima_idx]    
    
#     # Only count extrema above threshold
#     # Determine threshold using second box of histogram
#     histplt = plt.hist(std_of_coarse_mean)
#     quakes_idx = np.nonzero(maxima_vals>histplt[1][1]) #(biggest_quake)/10)
# # --------------end insert Mar 15 ----------------


#     # if no quakes occurred, continue
#     if len(quakes_idx) > 0:
        
#         # if len(np.shape(maxima_idx)) > 1:
#         #     maxima_idx = maxima_idx[0]
        
#         # quakes need to be multiplied by the window!
#         quakes = np.array(maxima_idx[0])[quakes_idx]*window            
#         # quakes = quakes[0]

#         if len(quakes) > 0:
#             # first quake length was first quake time - 0
#             quake_lens = [quakes[0]]
#             # each quake lasted from one time to the next
#             for q in range(1,len(quakes)-1):
#                 quake_lens.append(int(quakes[q]-quakes[q-1]))
#             # last steady state lasted from the last quake until the end
#             quake_lens.append(int(maxgens - quakes[-1]))
#         else: 
#             quake_lens = [maxgens]
#         # quake_lens = [quake_lens] # put all entries into
#             # brackets so it can be appended at once
#     else: # if no quakes
#         quakes, quake_lens = [], [maxgens]
# else: quakes, quake_lens = [], [maxgens] # if no maxima detected, just add nothing.
            
# # see plot1runPalmieri for plotting indiv quakes overlaid on model run        
# print("There were ",len(quakes)," quakes in this experiment\nQuakes at generations: ",str(quakes))
# quake_counts.append(len(quakes))
# quake_times.append(quakes)
# quake_lengths.append(quake_lens)

# average output
if multi_temp:
    popu_avg_by_T,div_avg_by_T,enc_avg_by_T,coresize_avg_by_T,corediv_avg_by_T,Jtot_avg_by_T = [],[],[],[],[],[]
    # loops through temperatures in ascending order
    for T in temperatures:
        popu_avg_by_T.append(popu_sum_by_T[str(T)]/len(seeds))
        div_avg_by_T.append(div_sum_by_T[str(T)]/len(seeds))
        enc_avg_by_T.append(enc_sum_by_T[str(T)]/len(seeds))
        coresize_avg_by_T.append(coresize_sum_by_T[str(T)]/len(seeds))
        corediv_avg_by_T.append(corediv_sum_by_T[str(T)]/len(seeds))
        Jtot_avg_by_T.append(Jtot_sum_by_T[str(T)]/len(seeds))
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
    div_avg = div_sum.toarray()/count
    enc_avg = enc_sum.toarray()/count
    coresize_avg = coresize_sum.toarray()/count
    corediv_avg = corediv_sum.toarray()/count
    Jtot_avg = Jtot_sum.toarray()/count

# %% Plots
# Make directory for all plots
if system == 'linux':
    figure_folder = output_folder + "/Figures_"+folder
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
    if want_plots:
        plt.show()
else: # (multi_temp:)** START HERE CAMILLE MARCH 28 22
    fig, ax = plt.subplots() 
    # ax.set_xscale('log')
    plt.title('Avg popln of ecosystems over time \n(%d seeds, %s, %s)'%(count/len(temperatures),folder[:-1],experiment[:-1])) #extra_folder))
    
    for T in range(len(temperatures)):
        c = colors[list(temperatures).index(int(temperatures[T]))]
        # ax.errorbar(range(len(popu_STD)),popu_avg[:-(len(popu_avg)-len(popu_STD))], yerr = popu_STD)
        ax.plot(popu_avg_by_T[T],label=str(temperatures[T]),color=c)

    ax.legend(bbox_to_anchor=(1,1),loc='upper left') #loc='center')
    # ax.legend(['Average Population Size','Lower Quartile','Upper Quartile','Median'])
    # ax.set_ylim(0,4*max(popu_avg))
    # how to get log scale with ax?
    ax.set_xlabel('Time (generations)', color = 'r') 
    ax.set_ylabel('N. indiv.s in ecosystem (avg w/in T)', color = 'black') 
    plt.tight_layout() 
    # # using the twinx() for creating another axes object for secondry y-Axis 
    # ax2 = ax.twinx()
    # ax2.plot(div_avg,'r')
    # # secondary y-axis label 
    # ax2.set_ylabel('Diversity (species in ecosystem)', color = 'r')
    plt.figure()
    for T in range(len(temperatures)):
        c = colors[list(temperatures).index(int(temperatures[T]))]
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
    if want_plots:
        plt.show()
        
    plt.savefig(figure_folder+"population_timeseries_"+experiment[:-1]+".pdf")
        
    plt.figure()
    # plt.ylim(0,2000)
    i = -1
    for j in popu_avg_by_T:
        i += 1
        plt.scatter(i,j[-1])
    # plt.ylim(0,2000)
    plt.title(experiment)
    plt.ylabel(f"Avg popln size after {maxgens} gens")
    plt.xlabel("Time (generations)")
    plt.tight_layout()
    plt.savefig(figure_folder+"diversity_timeseries_"+experiment[:-1]+".pdf")



# 3. Average interaction type
if not multi_temp:
    plt.figure()
    plt.plot(Jtot_avg[:,0]/popu_avg)#[:40000])
    plt.xscale('log')
    plt.xlabel('Log of generations')
    plt.ylabel("Avg effect on eachother (normalized to population size)")
    plt.title('Avg normalized interactions of ecosystems: \n(%d seeds, %s, %s)'%(count,folder[:-1],experiment[:-1]))
    figname = figure_folder+'Jtot_avg_normalized_'+folder[:-1]+'.pdf'
    plt.savefig(figname,format='pdf')
    #output_file = './'+figure_folder+'/Jtot_avg_arr'+folder
    #np.save(output_file, Jtot_avg)
    if want_plots:
        plt.show()
else:
    fig, ax = plt.subplots() 
    # ax.set_xscale('log')
    plt.title('Avg normalized interactions of ecosystems: \n(%d seeds, %s, %s)'%(count/len(temperatures),folder[:-1],experiment[:-1]))
    
    for T in range(len(temperatures)):
        c = colors[list(temperatures).index(int(temperatures[T]))]
        # ax.errorbar(range(len(popu_STD)),popu_avg[:-(len(popu_avg)-len(popu_STD))], yerr = popu_STD)
        ax.plot(Jtot_avg_by_T[T],label=str(temperatures[T]),color=c)

    #ax.legend(ncol=3,loc='lower right')
    ax.legend(bbox_to_anchor=(1,1),loc='upper left')
    # ax.legend(['Average Population Size','Lower Quartile','Upper Quartile','Median'])
    # ax.set_ylim(0,4*max(popu_avg))
    # how to get log scale with ax?
    ax.set_xlabel('Time (generations)') 
    ax.set_ylabel('Avg interaction in ecosystem at T', color = 'black') 
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
