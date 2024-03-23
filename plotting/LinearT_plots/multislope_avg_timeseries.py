# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:09:13 2022

@author: camil

average population over time for different LinearT experiments where temperature varies with different slopes

each output from each different slope should have its own folder, and each experiment needs to be the same length

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
experiment = 'Jun_17/' #'c_outputMay18\\' #' 'py_outputMay17\\' # (requires slashes after) folder containing "folder" (above) if Palmieri output
other_dates = False # ['Jun_10'] # list of possible dates in filenames other than that of experiment folder
folder = "LinearT/"
maxgens = 10000 # maximum number of generations in experiments in folder
multi_slope = True
# for multiple slopes, all must have interval 1 and same length.
# keep in same order: temp_starts, temp_ends, and folders
temp_starts = 298*np.ones(12) # np.r_[292,294,296,290]
temp_ends = np.r_[270:330:5] # [318,316,314,324]
locat = "/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/"
extra_folders = list(np.sort(os.listdir(locat+'LinearT/'+experiment))) # ['slope_292_318/','slope_294_316/','slope_296_314/','slope_290_324/'] # folders of each slope (ending in /)

for i in range(len(extra_folders)):
    if extra_folders[i].endswith('/'):
        continue
    else:
        extra_folders[i] += '/'

if len(extra_folders) > len(temp_starts):
    print("error: too many folders: ",extra_folders)
    print("removing figure folder")
    idx = extra_folders.index("Figures_LinearT/")
    extra_folders.pop(idx)
    if len(extra_folders) > len(temp_starts):
        print("error: too many folders STILL: ", extra_folders)

# full path of input folder
# locat = './' + experiment + folder
# locat = 'C:\\Users\\camil\\projects\\TaNa_test\\TTNM_output\\' #'C:\\Users\\camil\\projects\\TaNa_test\\Figures_etc\\' + experiment + folder #May14_C_ranJran\\model_output\\'
# locat = 'C:\\Users\\camil\\projects\\TaNa_test\\Figures_etc\\py_outputMay17\\'

multi_slope = True
want_plots = True
#want2profile = False #True
#code = 'py' # "C" or "py"

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

    
# Get filenames from locat
def get_filenames(locat):
    whofiles = []
    ecosystfiles = []
    print("LOCAT: \n",locat)
    #print(os.listdir(locat))
    for filename in os.listdir(locat):
        if filename.endswith(".dat"):
            if filename.startswith("diversity"):
                whofiles.append(filename) # collect names of who data files
            elif filename.startswith("pypy"): ecosystfiles.append(filename) # collect names of files with overall ecosystem statistics
    if ecosystfiles == False: # or len(ecosystfiles == 0):
        print(os.listdir(locat))
    else: print(len(ecosystfiles)," files found in ",locat)
    return whofiles, ecosystfiles


# %% Loop through ecosysyem files

# os.chdir("C:\\Users\\camil\\projects\\TaNa_test\\TN_c\\data_files\\")
# locat = "C:\\Users\\camil\\projects\\TaNa_test\\TN_c\\data_files\\"

# matrix of populations
# modification.... Jun 15, 2022 *****
popu_all = np.zeros((maxgens,len(temp_starts))) #sparse.lil_matrix((maxgens+1,len(ecosystfiles))) # huge matrix... may not be viable
# vector of STD of population
print("number of slope folders: ",len(extra_folders))
popu_STD = np.zeros((maxgens,2))

# sum across multiple runs
popu_sum = np.zeros(maxgens)
enc_sum = np.zeros(maxgens)
div_sum = np.zeros(maxgens)
coresize_sum = np.zeros(maxgens)
corediv_sum = np.zeros(maxgens)
Jtot_sum = np.zeros(maxgens)

# quakes
quake_counts = [] # how many quakes occured in each experiment
quake_times = [] # generations at which quake occurred
quake_lengths = [] # duration of steady states
#seeds = set() # keep track of seeds to use in legend later
counts = []
if multi_slope:
    print("Yooo it's multi slope")
    slopes=set()    
    # FIX THIS *** TODay june 17,2022
    locats = []
    ecosystfiles_by_slope = [] # list of lists for each slope
    for extra_folder in extra_folders:
        locat_now = locat+folder+experiment+extra_folder # ends with /
        if not os.path.exists(locat_now): #print("exists")
            print(locat_now,"doesn't exist")
        locats.append(locat_now)
        _,ecosystfiles_this_slope = get_filenames(locat_now) # list of output files for each seed at this slope
        print("number of experiments found in ",extra_folder,": ",len(ecosystfiles_this_slope))
        ecosystfiles_by_slope.append(ecosystfiles_this_slope)
# track order of creation of pop_all etc 

# keep track of which files can't be used because shape of data is wrong
unusual_files = []
    
    
# define arrays for each T
popu_sum_by_T = {}
div_sum_by_T = {}
enc_sum_by_T = {}
coresize_sum_by_T = {}
corediv_sum_by_T = {}
Jtot_sum_by_T = {}
        
    
# Make a colormpa
bwr= plt.get_cmap('bwr')
colors = bwr(np.linspace(0,1,len(extra_folders)))
        
# start loop again once seeds and temps sets are finished
i = -1
fig,ax = plt.subplots()
print("len(temp_starts): ",len(temp_starts),", len(extra_folders): ",len(extra_folders))
for ecosystfiles_this_slope in ecosystfiles_by_slope:
    count = 0
    i += 1
    if i>len(temp_starts):
        print("i larger than expected: ",i,">",len(temp_starts))
    print("THIS SLOPE: ",extra_folders[i])
    for file in ecosystfiles_this_slope:
        count+=1
        locat = locats[i]
        filename = locat + file
        if os.path.exists(filename):
            pass #print(f"file {count} exists")
        else: print(f"file {count} not found... {filename}")

        # Obtain model output 
        try:
            data = np.genfromtxt(filename, delimiter = ' ') # make maxgens x 9 matrix of data for current model run    
        except:
            print('wrong size of input file ',filename)
            unusual_files.append(f'{filename}: wrong input file size') #: {np.shape(data)}, data not used')
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
                unusual_files.append(f"{filename}: output shape: {np.shape(data)}")
                count -= 1
                continue #pass
            
        # The columns in Arthur data are: ['generation number','number of individuals','number of species', 'encountered.size()', 'individuals in core', 'species in core', 'effect on environment', 'effect on each other', 'external resources']
        gen = np.arange(maxgens)
        popu = data[0:,][0:,1] # population in the second column
        # if extinction occurred before maxgens reached, fill in zeros at end
        if len(popu) < maxgens:
            popu_long = np.zeros(maxgens)
            popu_long[:len(popu)] = popu.copy()
            #print("Length of popu is ",len(popu)," and lenght of popu_long is ",len(popu_long))
            popu_all[:,i] = popu_long
        elif len(popu) > maxgens:
            unusual_files.append(f"{filename}: file length: {len(popu)}")
            # # truncate
            data = data[:,][:maxgens]
            #gen = data[0:,][0:,0] # generations are in the first column
            popu = data[0:,][0:,1] # population in the second column
            popu_all[:,i] = np.reshape(popu,(1,len(popu)))
        else: # normal data size
            popu_all[:,i] = np.reshape(popu,(1,len(popu))) # matrix of all populations (for STD later)
        div = data[0:,][0:maxgens,2] # diversity in the third column
        enc = data[0:,][0:maxgens,3] # number of species encountered
        core_size = data[0:,][0:maxgens,4] # number of indiv in core
        core_div = data[0:,][0:maxgens,5] # number of spc in core
        Jtot = data[0:,][0:maxgens,7] # overall effect on each other            
        
        # plot individual runs on same plot
        c = colors[i] #[list(temperatures).index(int(temperature))]
        ax.plot(popu, color=c)
        # plt.ylim(0,20000)

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
        if str(extra_folders[i]) not in popu_sum_by_T.keys():
            print("new key for ",extra_folders[i])
            # if this T is not in the dictionaries yet, set zeros of the right size
            popu_sum_by_T[extra_folders[i]] = np.zeros((maxgens,1))
            div_sum_by_T[extra_folders[i]] = np.zeros((maxgens,1))
            enc_sum_by_T[extra_folders[i]] = np.zeros((maxgens,1))
            coresize_sum_by_T[extra_folders[i]] = np.zeros((maxgens,1))
            corediv_sum_by_T[extra_folders[i]] = np.zeros((maxgens,1))
            Jtot_sum_by_T[extra_folders[i]] = np.zeros((maxgens,1))
        popu_sum_by_T[extra_folders[i]] += np.reshape(popu,(len(popu),1)) # add populations from this experiment to total population at each generation
        div_sum_by_T[extra_folders[i]] += np.reshape(div,(len(div),1)) # add diversity from this experiment at each timepoint
        enc_sum_by_T[extra_folders[i]] += np.reshape(enc,(len(div),1))
        coresize_sum_by_T[extra_folders[i]] += np.reshape(core_size,(len(div),1))
        corediv_sum_by_T[extra_folders[i]] += np.reshape(core_div,(len(div),1))
        Jtot_sum_by_T[extra_folders[i]] += np.reshape(Jtot,(len(div),1))
    
    # number of seeds in that slope
    counts.append(count)


# add temperature to plot of all runs
ax2 = plt.twinx(ax)
for i in np.arange(len(temp_starts)):
    if temp_starts[i] < temp_ends[i]:
        temps = np.r_[temp_starts[i]:temp_ends[i]]
    else:
        temps = np.r_[temp_starts[i]:temp_ends[i]:-1]
    ax2.plot(np.linspace(0,maxgens,len(temps)),temps,"--",color=colors[i])
ax2.set_ylabel("Temperature",color='r')
plt.title(f"{experiment[:-1]} all runs")

print("number of experiments in each slope: ",counts)
count = max(counts)

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
popu_avg_by_T,div_avg_by_T,enc_avg_by_T,coresize_avg_by_T,corediv_avg_by_T,Jtot_avg_by_T = [],[],[],[],[],[]
# loops through temperatures in ascending order
for slope in extra_folders:
    popu_avg_by_T.append(popu_sum_by_T[slope]/count)
    div_avg_by_T.append(div_sum_by_T[slope]/count)
    enc_avg_by_T.append(enc_sum_by_T[slope]/count)
    coresize_avg_by_T.append(coresize_sum_by_T[slope]/count)
    corediv_avg_by_T.append(corediv_sum_by_T[slope]/count)
    Jtot_avg_by_T.append(Jtot_sum_by_T[slope]/count)

    #popu_medi_by_T = np.median(popu_all,axis=1)
    # count full zero experiments
    #all_indiv_ever = np.sum(popu_all,axis=0)
    #if len(all_indiv_ever) != count:
    #    print("all_indiv_ever not calculated correctly, its length is ",len(all_indiv_ever))
    #print("number of fully empty experiments: ",sum(np.where(all_indiv_ever==0)))
    #print("all individuals ever in each experiment: ",all_indiv_ever)
    
    # upper and lower quartiles of population
    #popu_quart1 = np.quantile(popu_all,0.25,axis=1)
    #popu_quart3 = np.quantile(popu_all,0.75,axis=1)
    #popu_STD = [popu_quart1, popu_quart3]

    #medval = np.zeros(maxgens)
    #for g in range(maxgens):
    #    vals = popu_all[g].copy()
    #    if sum(vals) == 0:
    #        break
    #    non0s = max(np.nonzero(vals)[0]) # highest index with value in it
        # vals = vals[:non0s] 
    #    vals.sort()
    #    min_tmp = .25*len(vals)
    #    if np.floor(min_tmp) != np.ceil(min_tmp):
    #        minval1 = vals[int(np.floor(min_tmp))]
    #        minval2 = vals[int(np.ceil(min_tmp))]
    #        minval = np.mean((minval1,minval2))
     #   else:
    #        minval = vals[int(min_tmp)]
    #    max_tmp = .75*len(vals)
    #    maxval = vals[int(max_tmp)]
    #    med_tmp = .5*len(vals)
    #    medval[g] = vals[int(med_tmp)]
    #    popu_STD[g] = [minval,maxval]
    #popu_STD = np.array(popu_all).std(0) # std along 1st axis (std of each generation)

# %% Plots
# Make directory for all plots
if system == 'linux':
    figure_folder = locat[:-len(extra_folders[-1])] + "Figures_"+folder
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
  
if not multi_slope:
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
if not multi_slope:
    fig, ax = plt.subplots() 
    ax.set_xscale('log')
    # plt.title('%s: Avg pop\'n and div\'ty of ecosystems over %d experiments'%(folder,count))
    plt.title(f'Average population and diversity of ecosystems: \n({count} seeds, {folder[:-1]}, {experiment[:-1]})')
    
    # ax.errorbar(range(len(popu_STD)),popu_avg[:-(len(popu_avg)-len(popu_STD))], yerr = popu_STD)
    ax.plot(popu_avg,'black',label="avg population size")
    ax.plot(coresize_avg,'--',color='black',label="avg core population")
    ax.plot(popu_quart1,':b',label="lower quartile") # lower quratile
    ax.plot(popu_quart3,':g',label="upper quartile") # upper quartile
    #ax.plot(popu_STD[:,0],':b') # lower quartile
    #ax.plot(popu_STD[:,1],':g') # upper quartile
    #ax.plot(medval,'--r') # median
    ax.plot(popu_med,':m',label="median") # median
    ax.legend() #ax.legend(['Average Population Size','Lower Quartile','Upper Quartile','Median'])
    # ax.set_ylim(0,4*max(popu_avg))
    # how to get log scale with ax?
    ax.set_xlabel('Log of generations') 
    ax.set_ylabel('Ecosystem Size (individuals)', color = 'black') 
    
    #plot temperature on different axis
    ax2 = plt.twinx(ax)
    ax2.set_ylabel("Temperature (K)",color="red")
    ax2.plot(np.linspace(0,maxgens,len(temps)),temps,"--r")

    # # using the twinx() for creating another axes object for secondry y-Axis 
    # ax2 = ax.twinx()
    # ax2.plot(div_avg,'r')
    # # secondary y-axis label 
    # ax2.set_ylabel('Diversity (species in ecosystem)', color = 'r')
    fig,ax = plt.subplots()
    ax.plot(div_avg,label="all species")
    ax.plot(corediv_avg,"m",label="core species")
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("Number of species")
    ax.legend()
    ax.set_title("Average Diversity over %d Experiments"%count)
    ax2 = plt.twinx(ax)
    ax2.set_ylabel("Temperature (K)",color="red")
    ax2.plot(np.linspace(0,maxgens,len(temps)),temps,"--r")
    plt.tight_layout()
    # save figure in Figure Folder
    figname = figure_folder+'popu_div_avg_'+folder[:-1]+'.pdf'
    plt.savefig(figname,format='pdf')
    if want_plots:
        plt.show()

# MULTI SLOPE :
else: # (multi_temp:)** START HERE CAMILLE MARCH 28 22
    fig, ax = plt.subplots() 
    # ax.set_xscale('log')
    plt.title(f'Average population of ecosystems: \n({count} seeds, {folder[:-1]}, {experiment[:-1]})')
    
    for slope in range(len(extra_folders)):
        c = colors[slope] #c = colors[list(temperatures).index(int(temperatures[T]))]
        # ax.errorbar(range(len(popu_STD)),popu_avg[:-(len(popu_avg)-len(popu_STD))], yerr = popu_STD)
        ax.plot(popu_avg_by_T[slope],label=str(extra_folders[slope]),color=c)

#    ax.legend(ncol=2,loc='upper right')
    #ax.legend(bbox_to_anchor=(1,1),loc="upper left")
    ax2 = plt.twinx(ax)
    #ax2.set_yticks(np.r_[270:321:10])
    #ytick_labels = []
    #for temp in np.r_[270:321:10]:
    #    ytick_labels.append(str(int(temp)))
    #ax2.set_yticklabels(ytick_labels,color='r')
    for i in np.arange(len(temp_starts)):
        if temp_starts[i] < temp_ends[i]:
            temps = np.r_[temp_starts[i]:temp_ends[i]]
        else:
            temps = np.r_[temp_starts[i]:temp_ends[i]:-1]
        print("length of temps this slope: ",temps,len(temps))
        ax2.plot(np.linspace(0,maxgens,len(temps)),temps,"--",color=colors[i],label=extra_folders[i])
    nylabs = len(ax2.get_yticklabels())
    y_colors = bwr(np.linspace(0,1,nylabs))
    for i in range(nylabs):
        ax2.get_yticklabels()[i].set_color(y_colors[i])
    ax2.set_ylabel("Temperature",color='r')
    # ax.legend(['Average Population Size','Lower Quartile','Upper Quartile','Median'])
    # ax.set_ylim(0,4*max(popu_avg))
    # how to get log scale with ax?
    ax.set_xlabel('Time (generations)') 
    ax.set_ylabel('\# Indiv.s in ecosystem (avg w/in slope)', color = 'black') 

    plt.savefig(figure_folder+"plot1")
    
    # # using the twinx() for creating another axes object for secondry y-Axis 
    # ax2 = ax.twinx()
    # ax2.plot(div_avg,'r')
    # # secondary y-axis label 
    # ax2.set_ylabel('Diversity (species in ecosystem)', color = 'r')
    fig,ax = plt.subplots()
    ax2 = plt.twinx(ax)
#    ax2.set_yticks(temp_ends)
#    ax2.set_yticklabels(ytick_labels,color='r')
    for slope in range(len(extra_folders)):
        c = colors[slope] #[list(temperatures).index(int(temperatures[T]))]
        ax.plot(div_avg_by_T[slope],label=extra_folders[slope],color=c)
        if temp_starts[slope] < temp_ends[slope]:
            temps = np.r_[temp_starts[slope]:temp_ends[slope]]
        else:
            temps = np.r_[temp_starts[slope]:temp_ends[slope]:-1]
        ax2.plot(np.linspace(0,maxgens,len(temps)),temps,"--",color=c,label=extra_folders[slope])
    nylabs = len(ax2.get_yticklabels())
    y_colors = bwr(np.linspace(0,1,nylabs))
    for i in range(nylabs):
        ax2.get_yticklabels()[i].set_color(y_colors[i])
    ax2.set_ylabel("Temperature",color='r')
    # plt.ylim(1,5000)
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("\# species in ecosystem (avg w/in slope)")
#    ax.legend(ncol=2,loc='upper right')
    #ax.legend(bbox_to_anchor=(1,1),loc="upper left")
    plt.title(f'Average diversity of ecosystems: \n({count} seeds, {folder[:-1]}, {experiment[:-1]})')
    plt.tight_layout()
    # save figure in Figure Folder
    #figname = figure_folder+'popu_div_avg_'+experiment[:-1]+'.pdf'
    plt.savefig(figure_folder+"plot2",format='pdf')
    #if want_plots:
    #    plt.show()
        
    #plt.savefig(figure_folder+"population_timeseries_"+experiment[:-1]+".pdf")
        
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
    
    #plt.savefig(figure_folder+"diversity_timeseries_"+experiment[:-1]+".pdf")



# 3. Average interaction type
fig,ax = plt.subplots()
ax2 = plt.twinx(ax)
i = -1
    #plt.plot(Jtot_avg[:,0]/popu_avg)#[:40000])
for Jtot_avg in Jtot_avg_by_T:
    i += 1
    ax.plot(np.linspace(0,maxgens,len(Jtot_avg)),Jtot_avg/popu_avg_by_T[i],color=colors[i])
    if temp_starts[i] < temp_ends[i]:
        temps = np.r_[temp_starts[i]:temp_ends[i]]
    else:
        temps = np.r_[temp_starts[i]:temp_ends[i]:-1]
    ax2.plot(np.linspace(0,maxgens,len(temps)),temps,"--",color=colors[i])
    # verfiy lenghts match...
    #ax2 = plt.twinx(ax)
    #ax2.plot(np.linspace(0,maxgens,len(temps)),temps,"--r")
    #ax2.set_xlabel("Temperature",color="red")
    #ax.set_xscale('log')
ax.set_xlabel('Time (log(generations))')
    #ax.set_xlabel("Time (generations)")
ax.set_ylabel("Avg effect on eachother (normalized to population size)")
#    ax2.set_ylabel("Temperature",color="red")
plt.title('%s: Avg normalized interactions of ecosystems over %d experiments'%(folder,count))
figname = figure_folder+'Jtot_avg_normalized_'+folder[:-1]+'.pdf'
plt.savefig(figname,format='pdf')
    #output_file = './'+figure_folder+'/Jtot_avg_arr'+folder
    #np.save(output_file, Jtot_avg)
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
