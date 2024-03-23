""" Feb 25: Script to process SpPops dictionary output by TNM_all.

TNM_all produces dictionaries and saves them as np files.

The dictionaries are in the form:
    SpPops['111'] = [[t1,pop1],[t2,pop2],...[tn,popn]]

Load them using:
    SpPops = np.load('location/SpPops_file.npy',allow_pickle=True).itme()

"""
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import rc
from os.path import exists
import sys
import os
#if '/home/cfebvre/repos/tnm_febvre/geochem' not in sys.path:
#    sys.path.append('/home/cfebvre/repos/tnm_febvre/geochem')
# if 'C:/Users/camil/repos/tnm_febvre/geochem' not in sys.path:
#     sys.path.append('C:/Users/camil/repos/tnm_febvre/geochem')
# from geoTNM import temperature_effects as TM
from geoTNM.analysis import analyze_J 
#rc('text', usetex=True)
#plt.rcParams.update({"text.usetex": True})

verbose = True

# User inputs
temp_script = "temp_effects"
seed = np.r_[100:200] #:13009]
T = np.r_[274:320:3] #[274:320:3] #[274:320:4] #[283:317:3] #[288:310:5] #[283:317:3] #False #288
dates = ['Jun_20'] # ['Mar_21','Mar_22'] #'Feb_28'
other_file_dates = ['Jun_21','Jun_22','Jun_23','Jun_24','Jun_25'] #,'Jun_26','Jun_27','Jun_28','Jun_29','Jun_30','Jul_01','Jul_02'] #,'Jun_19','Jun_20'] # False or list of other dates that might be in file name. if all filenames match folder date, then can do False
experiment = 'SteadyT/' #'UnknownExperiment/' #'SteadyT/' #'SpunupT/' #'BasicTNM/' # 'SpunupT/'
extra_folder = "/poff_is_roff" #"/poff_is_roff" #'/skew_norm_new_parameters' #/carb_truncated' #'/carb_model_branch' #'/Feb_21_copy_truncated/truncated' # slash in front only or else ''
L = 20 #20
locat = '/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/'
BasicTNM_locat = locat + 'BasicTNM/Jun_17/'
# locat = f'/Users/camil/projects/TaNa_test/TTNM_output/{extra_folder}'
plotting = True
plot_multiple = True
if temp_script == "met_theory":
    import met_theory_simple as TM

# assemble filename of input
def find_file(date,seed,T,type="pypy"):
    # return locat+f"pypy_seed{str(seed)}{date}_{str(T)}K_TNM_L20.dat"
    #print("* * * f i n d i n g   f i l e * * * ")
    #print("looking for ",type," file")
    if type == "pypy": 
        #print("entered pypy for loop")
        file = f"{locat}{experiment}{date}{extra_folder}/pypy_seed{str(seed)}{date}_{str(T)}K_TNM_L{L}.dat"
        if not exists(file):
            #print("pypy file not found on first try, checking again")
            # check other possible dates for filename if this filename wasn't found
            for i in np.r_[0:len(other_file_dates)]:
                date2 = other_file_dates[i]
                # print('checking ',date2,' to see if it exists...')
                file = f"{locat}{experiment}{date}{extra_folder}/pypy_seed{str(seed)}{date2}_{str(T)}K_TNM_L{L}.dat"
                if exists(file):
                    #print('pypy file found with new date')
                    break
                #else: 
                #    print("NO FILE FOUND: ")
                #    print(file)
    elif type == "div":
        #print("entered div for loop")
        file = f"{locat}{experiment}{date}{extra_folder}/diversity_seed{str(seed)}{date}_{str(T)}K_TNM_L{L}.dat"
        if not exists(file):
            file2 = 0
            #print("div file not found on first try")
            # check other possible dates for filename
            end = len(other_file_dates)
            for j in range(0,end): # np.r_[0:end]: #len(other_file_dates)]:
                #print(j," out of ",end)
                date2 = other_file_dates[j]
                #print('checking ',date2,' to see if it exists...')
                file = f"{locat}{experiment}{date}{extra_folder}/diversity_seed{str(seed)}{date2}_{str(T)}K_TNM_L{L}.dat"
                #print(file,exists(file))
                if exists(file):
                    #print("div file found with new date")
                    break
                    #file2 = file1
            if not exists(file2): print("div file never found: ",file)
            #else: file = file2
        else: date2 = date
    else: print("Error, define file type as pypy or div")
    if exists(file):
        #print("File exists") #: ",file)
        return file, date2
    else: print("Can't find file: ",file)

# Get population and diversity time series from pypy file
def get_pop_div(date,seed,T,type="pypy",filename=None):
    if type == "basic": # find BasicTNM run files
        pypy_file = filename
        print('*******\nBasicTNM file:\n',pypy_file)
        print(exists(pypy_file))
    else:
        pypy_file,date2 = find_file(date,seed,T,"pypy")
    print("date, seed, T: ",date,seed,T)
    populations = []
    diversities = []
    core_pops = []
    core_divs = []
    gens_run = []
    if exists(pypy_file):
        with open(pypy_file,'r') as pypy:
            for line in pypy:
                elements = line.split(" ")
                # (tgen,sum(populations),len(species),len(encountered),core_pop,core_div,F))
                gens_run.append(int(elements[0]))
                populations.append(int(elements[1]))
                diversities.append(int(elements[2]))
                core_pops.append(int(elements[4]))
                core_divs.append(int(elements[5]))
#            populations.append(int(elements[1]))
#            diversitites.append(int(elements[2]))
    else:
        print(f"could not open file with seed {sd} and T = {T}")
#        populations.append(0)
#        diversities.append(0)
        return
    return gens_run, populations, diversities, core_pops, core_divs

def get_interaction(date,sd,temp,type="div",locat=locat):
    interactions = []
    if type == "basic":
        div_file = date # not the real date, just BasicTNM_locat+filename!
        seed_idx = div_file.index("seed")
        sd = int(div_file[seed_idx+4:seed_idx+7])
        date = div_file[seed_idx+7:seed_idx+13]
        date2 = date
        T = 'Fals'
        interactions = analyze_J.final_interaction(div_file,date,sd,T=T,locat=BasicTNM_locat)
    else:
        div_file,date2 = find_file(date,sd,temp,type)
    # print("get_interactions:")
        #if verbose: 
        #   print("FOUND ",div_file)
    # print(locat)
        if exists(div_file):
            interactions = analyze_J.final_interaction(div_file,date,sd,temp,locat,date2)
            if interactions == 0:
                print("NO INTERACTIONS FOUND! ",date,sd,temp,date2)
            else:
                pass
                # print("Interactions found for ",date,sd,temp,date2)
        else: print("Diversity file doesn't exist")
    return interactions

# %%
if plot_multiple:
    gens_by_T = []
    pops_by_T = []
    divs_by_T = []
    core_pops_by_T = []
    core_divs_by_T = []
    interactions_by_T = []

    # Basic TNM
    gens_this_T = []
    pops_this_T = []
    divs_this_T = []
    core_pops_this_T = []
    core_divs_this_T = []
    interactions_this_T = []
    n = 0
    basicTNM_contents = os.listdir(BasicTNM_locat)
    basic_seeds = [] # file names of basicTNM pypy files
    basic_div = [] # file names of basicTNM diversity files
    for out_file in basicTNM_contents:
        if out_file.startswith('pypy'):
            basic_seeds.append(out_file)
        elif out_file.startswith('diversity'):
            basic_div.append(out_file)
    i = -1
    for sd in basic_seeds:
        n+=1
        i+=1
        #try:
        gens_run,pop_temp,div_temp,core_pop,core_div = get_pop_div(0,0,False,type="basic",filename=BasicTNM_locat+sd)
        gens_this_T.append(gens_run[-1])
        pops_this_T.append(pop_temp[-1])
        divs_this_T.append(div_temp[-1])
        core_pops_this_T.append(core_pop[-1])
        core_divs_this_T.append(core_div[-1])
        # except:
            #print("Couldn't get pop and div for ",date,sd,temp)
#                     pop_temp,div_temp = get_pop_div(date,sd,temp)
        #     continue
        # final popualation at this temperature
        #try:
        interact_temp = get_interaction(BasicTNM_locat+basic_div[i],0,0,"basic",BasicTNM_locat)
        print("interact_temp: ",interact_temp)
        interactions_this_T.append(np.mean(interact_temp))
        #except:
            # interact_temp = get_interaction(date,sd,temp,"div",div_locat)
        #    print("interactions not determined, ",basic_div[i])
    print(f"T={T}, shape of popsthisT: {np.shape(pops_this_T)}")
    gens_by_T.append(np.array(gens_this_T))
    pops_by_T.append(np.array(pops_this_T))
    divs_by_T.append(np.array(divs_this_T))
    core_pops_by_T.append(np.array(core_pops_this_T))
    core_divs_by_T.append(np.array(core_divs_this_T))
    interactions_by_T.append(np.array(interactions_this_T))
    
    # Now go through each temperature
    for temp in T:
        gens_this_T = []
        pops_this_T = []
        divs_this_T = []
        core_pops_this_T = []
        core_divs_this_T = []
        interactions_this_T = []
        n = 0
        for date in dates:
            n = 0
            div_locat = f"{locat}{experiment}{date}{extra_folder}/"
            for sd in seed:
                n+=1
                try:
                    gens_run,pop_temp,div_temp,core_pop,core_div = get_pop_div(date,sd,temp)
                    gens_this_T.append(gens_run[-1])
                    pops_this_T.append(pop_temp[-1])
                    divs_this_T.append(div_temp[-1])
                    core_pops_this_T.append(core_pop[-1])
                    core_divs_this_T.append(core_div[-1])
                except:
                    #print("Couldn't get pop and div for ",date,sd,temp)
#                     pop_temp,div_temp = get_pop_div(date,sd,temp)
                    continue
                # final popualation at this temperature
                try:
                    interact_temp = get_interaction(date,sd,temp,"div",div_locat)
                    interactions_this_T.append(np.mean(interact_temp))
                except:
                    # interact_temp = get_interaction(date,sd,temp,"div",div_locat)
                    print("interactions not determined, ",sd,temp,date)
        print(f"T={T}, shape of popsthisT: {np.shape(pops_this_T)}")
        gens_by_T.append(np.array(gens_this_T))
        pops_by_T.append(np.array(pops_this_T))
        divs_by_T.append(np.array(divs_this_T))
        core_pops_by_T.append(np.array(core_pops_this_T))
        core_divs_by_T.append(np.array(core_divs_this_T))
        interactions_by_T.append(np.array(interactions_this_T))
    
    # boxplots
    if not plotting:
        print(pops_by_T, divs_by_T, interactions_by_T)
    else:
        if div_locat not in sys.path:
            sys.path.append(div_locat)
        try: import temperature_effects as TM
        except: 
            print("couldn't find temp effects in output path")
            from geoTNM import temperature_effects as TM

        print(f"T: {T}, number of boxes: {len(pops_by_T)}, {np.shape(pops_by_T)}")

        positions = np.insert(T,0,T[0] - 5)
        xlabels = np.array(T,dtype=str)
        xlabels = np.insert(xlabels,0,'TNM')
        fig,ax = plt.subplots(3,2,sharex='all') #True)
        part1 = ax[0,0].violinplot(pops_by_T[:],positions=positions,widths=4,showextrema=False,showmedians=True)
        part2 = ax[1,0].violinplot(divs_by_T[:],positions=positions,widths=4,showextrema=False,showmedians=True)
        # ax[1,0].tick_params(labelrotation=90)
        
        part3 = ax[0,1].violinplot(core_pops_by_T[:],positions=positions,widths=4,showextrema=False,showmedians=True)
        part4 = ax[1,1].violinplot(core_divs_by_T[:],positions=positions,widths=4,showextrema=False,showmedians=True)
        part5 = ax[2,0].violinplot(interactions_by_T[:],positions=positions,widths=4,showextrema=False,showmedians=True)
        ax[2,1].plot(T,TM.poff_total(0,T)-TM.pdeath(T),color='b')

        # Set y limits
        if temp_script == "met_simple":
            ax[2,1].set_ylim(-.1,.5)
        else: ax[2,1].set_ylim(0)

        # axis labels and orientation
        ax[0,0].set_ylabel(f"Total pop'ln",color="b") #blue")
        ax[0,1].set_ylabel("Core pop'ln",color="b")
        ax[1,1].set_ylabel("Core div.",color="b")
        ax[1,0].set_ylabel(f"Total div.",color="b")
        # ax[1,1].tick_params(labelrotation=90)
        ax[2,0].set_ylabel(f"Core interact'ns ",color="b")
        # ax[2,0].set_xticklabels(xlabels)
        # ax[1,0].set_xlabel("Temperature")
        ax[2,1].set_ylabel("Poff(fi=0)-Pdeath",color='b')
        ax[0,0].tick_params(labelrotation=90)
        ax[0,1].tick_params(labelrotation=90)
        ax[1,0].tick_params(labelrotation=90)
        ax[1,1].tick_params(labelrotation=90)
        ax[2,0].tick_params(labelrotation=90)
        ax[2,1].tick_params(labelrotation=90)

        # Color the violin faces according to temperature
        bwr = plt.get_cmap('bwr')
        colors = bwr(np.linspace(0,1,len(part1['bodies'])))
        i = -1
        for pc in part1['bodies']:
                i += 1
                if i == 0:
                    pc.set_facecolor("m")
                else:
                    pc.set_facecolor(colors[i])
                pc.set_edgecolor('black')
                pc.set_alpha(.8)

        i = -1
        for pc in part2['bodies']:
                i += 1
                if i == 0:
                    pc.set_facecolor("m")
                else:
                    pc.set_facecolor(colors[i])
                pc.set_edgecolor('black')
                pc.set_alpha(.8)

        i = -1
        for pc in part3['bodies']:
                i += 1
                if i == 0:
                    pc.set_facecolor("m")
                else:
                    pc.set_facecolor(colors[i])
                pc.set_edgecolor('black')
                pc.set_alpha(.8)

        i = -1
        for pc in part4['bodies']:
                i += 1
                if i == 0:
                    pc.set_facecolor("m")
                else:
                    pc.set_facecolor(colors[i])
                pc.set_edgecolor('black')
                pc.set_alpha(.8)

        i = -1
        for pc in part5['bodies']:
                i += 1
                if i == 0:
                    pc.set_facecolor("m")
                else:
                    pc.set_facecolor(colors[i])
                pc.set_edgecolor('black')
                pc.set_alpha(.8)

        # ax[2,1].set_xticklabels(xlabels)
        # ax[1,1].set_xlabel("Temperature")
        plt.suptitle(f"Final ecosystem characteristics:\n{n} experiments at each T ({experiment[:-1]}, {date})") 

        # set x axis label in center below subplots
        fig.text(0.5, 0.015, 'Temperature (K)', color="red",ha='center', va='center')        
        # label control case
        #fig.text(0.13, 0.08, 'control',color='m',ha='center',va='center') #,rotation=90)
        #fig.text(0.56, 0.08, 'control',color='m',ha='center',va='center') #,rotation=90)
        fig.text(0.19, 0.82, 'control',color='m',ha='center',va='center',rotation=45)

        #try:
        plt.savefig(locat+experiment+date+extra_folder+"/violins_"+experiment[:-1]+date+".pdf")
        #except:
            #print("figure couldn't be saved...")
        plt.show()
        
        ##
        """
        for pops in core_pops_by_T[:]:
            fig,ax = plt.subplots()
            ax.violinplot(pops)
            """
        ##

        fig,ax = plt.subplots() #(3,sharex='all')
        ax.boxplot(gens_by_T[:],positions=positions)
        ax.set_title("Generations completed in each experiment")
        ax.set_xlabel("Temperature (K)")
        
        # ax[0].boxplot(core_pops_by_T[:],positions=T)
        # ax[1].boxplot(core_divs_by_T[:],positions=T)
        # ax[2].boxplot(gens_by_T[:],positions=T)
        # ax[0].set_title("Core populations")
        # ax[1].set_title("Core diversity")
        # ax[1].set_ylabel("Count")
        # plt.suptitle(f"{n} experiments at each temperature ({experiment[:-1]}, {date})")
        # fig.text(0.5,0.015, 'Temperature (K)', ha='center', va='center')
        
        plt.show()



if plotting and not plot_multiple:
    populations, diversities = get_pop_div(seed,T)
    # plot population and diversity
    fix, ax = plt.subplots()
    ax.plot(range(len(populations)),populations,label='population')
    ax.set_ylabel("Population of Ecosystem",color="blue")
    plt.locator_params(axis=ax,nbins=10)
    ax2 = ax.twinx()
    ax2.plot(range(len(populations)),diversities,"red",label='diversity')
    ax2.set_ylabel("Diversity of Ecosystem",color="red")
    ax.set_xlabel("Time")
    plt.locator_params(axis='both',nbins=6)
    plt.title(f"{experiment} {date} {seed} {T}")
    plt.show()



