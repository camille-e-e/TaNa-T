""" Jan 23, 2023: This is a copy of final_stats, which I will modify to plot
final stats in grid format for various poff, pdeath combinations.

Feb 25, 2022: Script to process SpPops dictionary output by TNM_all.

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
from datetime import datetime, timedelta
#if '/home/cfebvre/repos/tnm_febvre/geochem' not in sys.path:
#    sys.path.append('/home/cfebvre/repos/tnm_febvre/geochem')
# if 'C:/Users/camil/repos/tnm_febvre/geochem' not in sys.path:
#     sys.path.append('C:/Users/camil/repos/tnm_febvre/geochem')
# from geoTNM import temperature_effects as TM
#from geoTNM.analysis import analyze_J 
#rc('text', usetex=True)
#plt.rcParams.update({"text.usetex": True})

verbose = True

# User inputs
# means or medians for abundance data?
plot_type = "median" # "med" or "mean"
temp_script = "temp_effects"
seed = np.r_[1000:1050] #[200:700] #[1000:2000] #:13009]
pdeath_range = [.1] #np.r_[.05:1:.05] #[.05:.45:.05,.5:1:.1] # [False] # np.r_[.1:1:.1]
poff_range = [1] # np.r_[.05:1:.05] # [False] # np.r_[.1:1:.1]
T_range = [False] #np.r_[274:320:3]
mu_range = np.r_[.04:.22:.03]
pmut_range = np.r_[.004:.022:.003]
C = 100
theta = 0.25
#T = np.r_[274:320:3] #[274:320:3] #[274:320:4] #[283:317:3] #[288:310:5] #[283:317:3] #False #288
dates = ['Mar_17_23','Mar_24_23','Mar_27_23'] # ['Feb_28'] # ['Jan_18','Jan_31','Feb_06'] #,'Oct_11'] # ['Sep_09'] #['Jul_21'] # ['Jun_20'] # ['Mar_21','Mar_22'] #'Feb_28'
maxgens = 10_000
sample_size = 50
num_tries=40 # number of other filenames to try  #other_file_dates = [str(dates[0][:4]) + i for i in list(np.array(np.r_[04:30],dtype=str))] # ['Jul_22','Jul_23','Jul_24','Jul_25','Jul_26','Jul_27','Jul_28','Jul_29'] #,'Jun_26','Jun_27','Jun_28','Jun_29','Jun_30','Jul_01','Jul_02'] #,'Jun_19','Jun_20'] # False or list of other dates that might be in file name. if all filenames match folder date, then can do False
experiment = 'prob_test/' # 'SpunupT/' # 'prob_test/' #'UnknownExperiment/' #'SteadyT/' #'SpunupT/' #'BasicTNM/' # 'SpunupT/'
extra_folder = "vary_mu_x_pmut/" # '/mu_.3' #'/prob_test' #"/MTE-env" # "/prob_test" # "/MTE_TPC_combo" # "/Variable_Tresponse" # "/poff_is_roff" #"/poff_is_roff" #'/skew_norm_new_parameters' #/carb_truncated' #'/carb_model_branch' #'/Feb_21_copy_truncated/truncated' # slash in front only or else ''
L = 20 #20
locat = '/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/'
BasicTNM_locat = locat + 'BasicTNM/Jun_17/'
locat = '/net/venus/kenes/user/cfebvre/out_venus/TTNM_outputs/'
# locat = f'/Users/camil/projects/TaNa_test/TTNM_output/{extra_folder}'
plotting = True
plot_multiple = True
if temp_script == "met_theory":
    import met_theory_simple as TM

# %% ---------------------
# produce output file and write date and time in it
error_file = "OUT_from_final_grid_plot.txt"
with open(error_file,'a') as f:
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    f.write(dt_string)

# %% ---------------------
# assemble filename of input
def find_file(date,seed,T=False,poff=False,pdeath=False,type="pypy",C='',mu='',theta='',pmut='',L=L):
    # return locat+f"pypy_seed{str(seed)}{date}_{str(T)}K_TNM_L20.dat"
    #print("* * * f i n d i n g   f i l e * * * ")
    #print("looking for ",type," file")
    date2 = date
    if type == "pypy": 
        #print("entered pypy for loop")
        if poff:
            file = f"{locat}{experiment}{extra_folder}{date}/pypy_seed{str(seed)}{date[:6]}_poff{poff:.2f}_pdeath{pdeath:.2f}K_TNM{C}{mu}{theta}{pmut}_L{L}.dat"
        else: file = f"{locat}{experiment}{date}{extra_folder}/pypy_seed{str(seed)}{date}_{str(T)}K_TNM_L{L}.dat"
        #print("file: ",file)
        #print(exists(file))
        if not exists(file):
            #print("pypy file not found on first try, checking again")
            # check other possible dates for filename if this filename wasn't found

            for j in range(0,num_tries): #,end): # np.r_[0:end]: #len(other_file_dates)]:
                #print(j," out of ",end)
                date1 = datetime.strptime(date,"%b_%d_%y")
                adjusted_date = date1 + timedelta(days=j)
                date2 = adjusted_date.strftime("%b_%d_%y")  #other_file_dates[j]
                #print('checking ',date2,' to see if it exists...')
                if poff:
                    file = f"{locat}{experiment}{extra_folder}{date}/pypy_seed{str(seed)}{date2[:6]}_poff{poff:.2f}_pdeath{pdeath:.2f}K_TNM{C}{mu}{theta}{pmut}_L{L}.dat"
                else:
                    file = f"{locat}{experiment}{date}{extra_folder}/pypy_seed{str(seed)}{date2}_{str(T)}K_TNM_L{L}.dat"
                #print(file,exists(file))
                if exists(file):
                    #print("div file found with new date")
                    break

                #else: 
                #    print("NO FILE FOUND: ")
                #    print(file)
            #print("---file found---")
    elif type == "div":
        #print("entered div for loop")
        if poff:
            file = f"{locat}{experiment}{date}{extra_folder}/diversity_seed{str(seed)}{date}_poff{poff:.2f}_pdeath{pdeath:.2f}K_TNM{C}{mu}{theta}{pmut}_L{L}.dat"
        else:
            file = f"{locat}{experiment}{date}{extra_folder}/diversity_seed{str(seed)}{date}_{str(T)}K_TNM_L{L}.dat"
        if not exists(file):
            file2 = 0
            #print("div file not found on first try")
            # check other possible dates for filename
            #end = len(other_file_dates)
            for j in range(0,num_tries): #,end): # np.r_[0:end]: #len(other_file_dates)]:
                #print(j," out of ",end)
                date1 = datetime.strptime(date,"%b_%d")
                adjusted_date = date1 + timedelta(days=j)
                date2 = adjusted_date.strftime("%b_%d")  #other_file_dates[j]
                #print('checking ',date2,' to see if it exists...')
                if poff:
                    file = f"{locat}{experiment}{date}{extra_folder}/diversity_seed{str(seed)}{date2}_poff{poff:.2f}_pdeath{pdeath:.2f}K_TNM{C}{mu}{theta}{pmut}_L{L}.dat"
                else:
                    file = f"{locat}{experiment}{date}{extra_folder}/diversity_seed{str(seed)}{date2}_{str(T)}K_TNM_L{L}.dat"
                #print(file,exists(file))
                if exists(file):
                    #print("div file found with new date")
                    break
                    #file2 = file1
            if not exists(file2): #print("div file never found: ",file)
                pass
            #else: file = file2
        else: date2 = date
    else: print("Error, define file type as pypy or div")
    if exists(file):
        #print("---File exists---: ",file)
        return file, date2
    else: print("Can't find file: ",file)
    return 0,0

def get_pop_div(date,seed,T=False,poff=False,pdeath=False,type="pypy",filename=None,C='',mu='',theta='',pmut='',L=L):
    """ Get population and diversity time series from pypy file"""
    #print("looking for ",type," file")
    if type == "basic": # find BasicTNM run files
        pypy_file = filename
#        print('*******\nBasicTNM file:\n',pypy_file)
#        print(exists(pypy_file))
    else:
        #print("looking for pypy file")
        pypy_file,date2 = find_file(date,seed,T=T,poff=poff,pdeath=pdeath,type="pypy",C=C,mu=mu,theta=theta,pmut=pmut,L=L)
        print('************\n'+experiment+'\n',pypy_file)
    if pypy_file == 0:
        return 0,0,0,0,0
    #print("file found: ",pypy_file) #"date, seed, T: ",date,seed,T)
    populations = list(np.zeros((maxgens,),dtype=int))
    diversities = list(np.zeros((maxgens,),dtype=int))
    core_pops = list(np.zeros((maxgens,),dtype=int))
    core_divs = list(np.zeros((maxgens,),dtype=int))
    #gens_run = list(np.zeros((maxgens,),dtype=int))
    i = -1
    if exists(pypy_file):
        with open(pypy_file,'r') as pypy:
            for line in pypy:
                i += 1
                if i >= maxgens:
                    break
                elements = line.split(" ")
                # (tgen,sum(populations),len(species),len(encountered),core_pop,core_div,F))
                gens_run = i #gens_run[i] = int(elements[0])
                populations[i] = int(elements[1])
                diversities[i] = int(elements[2])
                core_pops[i] = int(elements[4])
                core_divs[i] = int(elements[5])
#            populations.append(int(elements[1]))
#            diversitites.append(int(elements[2]))
    else:
        print(f"could not open file with seed {sd} and T = {T}")
#        populations.append(0)
#        diversities.append(0)
        return
    return gens_run, populations, diversities, core_pops, core_divs

def get_interaction(date,sd,temp=False,poff=False,pdeath=False,type="div",locat=locat):
    interactions = []
    if type == "basic":
        div_file = date # not the real date, just BasicTNM_locat+filename!
        seed_idx = div_file.index("seed")
        sd = int(div_file[seed_idx+4:seed_idx+7])
        date = div_file[seed_idx+7:seed_idx+13]
        date2 = date
        T = 'Fals'
        interactions = False #analyze_J.final_interaction(div_file,date,sd,T=T,locat=BasicTNM_locat)
    else:
        div_file,date2 = find_file(date,sd,T=temp,poff=poff,pdeath=pdeath,type=type,C=C,mu=mu,theta=theta,pmut=pmut,L=L)
    # print("get_interactions:")
        #if verbose: 
        #   print("FOUND ",div_file)
    # print(locat)
        if exists(div_file):
            interactions = False #analyze_J.final_interaction(div_file,date,sd,temp,locat,date2)
            if interactions == 0:
                print("NO INTERACTIONS FOUND! ",date,sd,temp,date2)
            else:
                pass
                # print("Interactions found for ",date,sd,temp,date2)
        else: print("Diversity file doesn't exist")
    return interactions

# %%
window = 300 # 300
if plot_multiple:
    indep_now = [] # keep track of T,poff,pdeath

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
        gens_this_T.append(gens_run) #[-1])
        # average last "window" of generations
        pops_this_T.append(np.mean(pop_temp[-window:]))
        divs_this_T.append(np.mean(div_temp[-window:]))
        core_pops_this_T.append(np.mean(core_pop[-window:]))
        core_divs_this_T.append(np.mean(core_div[-window:]))
        # except:
            #print("Couldn't get pop and div for ",date,sd,temp)
#                     pop_temp,div_temp = get_pop_div(date,sd,temp)
        #     continue
        # final popualation at this temperature
        #try:
        interact_temp = get_interaction(BasicTNM_locat+basic_div[i],0,type="basic",locat=BasicTNM_locat)
        #print("interact_temp: ",interact_temp)
        interactions_this_T.append(np.mean(interact_temp))
        #except:
            # interact_temp = get_interaction(date,sd,temp,"div",div_locat)
        #    print("interactions not determined, ",basic_div[i])

    print(f"T={T_range}, shape of popsthisT: {np.shape(pops_this_T)}")
    # append 1D list, showing final stats sorted by T, ..._by_T arrays 
    indep_now.append([False]) # no drivers in basic experiment
    gens_by_T.append(gens_run) #np.array(gens_this_T))
    pops_by_T.append(np.array(pops_this_T))
    divs_by_T.append(np.array(divs_this_T))
    core_pops_by_T.append(np.array(core_pops_this_T))
    core_divs_by_T.append(np.array(core_divs_this_T))
    interactions_by_T.append(np.array(interactions_this_T))
    
    # Now go through each temperature,poff, and pdeath
    # and gather average of last points in timeseries
    # *** NOTE *** these functions could also be used for timeseries! 
    #window = 10 # average last window of timeseries points
    for temp in T_range:
        for poff in poff_range:
            for pdeath in pdeath_range:
                for mu in mu_range:
                    for pmut in pmut_range:
                        # keep track of independent variables in 
                        if len(poff_range) > 1:
                            indep_now.append([poff,pdeath])
                        elif len(mu_range) > 1:
                            indep_now.append([mu,pmut])
                        gens_this_T = []
                        pops_this_T = []
                        divs_this_T = []
                        core_pops_this_T = []
                        core_divs_this_T = []
                        interactions_this_T = []
                        n = 0
                        sample = 0
                        for sd in seed:
                        #for date in dates:
                            n = 0
                            sample += 1
                            if sample >= sample_size:
                                break
                            for date in dates:
                                div_locat = f"{locat}{experiment}{date}{extra_folder}/"
                            #for sd in seed:
                                #n+=1
                                # collect only as many model runs as the desired sample size
                                #if sample <= sample_size:
                                if 1 == 1:
                                    #try:
                                    if 1 == 1:
                                #        sample += 1
                                        # get timeseries
                                        gens_run,pop_temp,div_temp,core_pop,core_div = get_pop_div(date,sd,T=temp,poff=poff,pdeath=pdeath,mu=f"_mu{mu:.2f}",pmut=f"_pmut{pmut:.3f}",C=f"_C{C}",theta=f"_theta{theta:.2f}",L=L)
                                        if gens_run == 0: # file never found, experiment didn't even start
                                            sample -= 1
                                            continue # skip this seed and go to next
                                        # number of generations completed in this run
                                        gens_this_T.append(gens_run) #[-1])
                                        # if pop_temp[-1] == 0:

                                        # final popualation & div at this temperature
                                        # average the last window of timepoints
                                        #with open(error_file,'a') as f:
                                        #    f.write(pop_temp[-window:])
                                        #    f.write(np.mean(pop_temp[-window:]))

                                        pops_this_T.append(np.mean(pop_temp[-window:]))
                                        divs_this_T.append(np.mean(div_temp[-window:]))
                                        core_pops_this_T.append(np.mean(core_pop[-window:]))
                                        core_divs_this_T.append(np.mean(core_div[-window:]))
                                        break # no need to see other dates with same seed
                                    #except:
                                    #    print("Couldn't get pop and div for ",date,sd,temp)
                                    #    print(div_locat)
                    #               #      pop_temp,div_temp = get_pop_div(date,sd,temp)
                                    #    continue

                                    # try to get interactions
                                    if 1 == 0:
                                        try:
                                            interact_temp = get_interaction(date,sd,T=temp,poff=poff,pdeath=pdeath,type="div",locat=div_locat)
                                            interactions_this_T.append(np.mean(interact_temp))
                                        except:
                                            # interact_temp = get_interaction(date,sd,temp,"div",div_locat)
                                            print("interactions not determined, ",sd,temp,date)
                        print(f"T={T_range}, shape of popsthisT: {np.shape(pops_this_T)}")
                        print("\nshape of pops_this_T: ",np.shape(pops_this_T))
                        gens_by_T.append(np.array(gens_this_T))
                        pops_by_T.append(np.array(pops_this_T))
                        divs_by_T.append(np.array(divs_this_T))
                        core_pops_by_T.append(np.array(core_pops_this_T))
                        core_divs_by_T.append(np.array(core_divs_this_T))
                        interactions_by_T.append(np.array(interactions_this_T))

    # Now find means and quantiles
    # TEST RUN
    # figure out which temperatures have no numbers
    # nonzero_by_T = np.nonzero(np.sum(pops_by_T,axis=0))
    #pops_by_T = np.reshape(pops_by_T,[17,100])
    #print("shape of pops_by_T: ", np.shape(pops_by_T))
    samplesize_by_T = []
    survived_by_T = []
    survival_by_T = []
    popmeans,divmeans,corepopmeans,coredivmeans,Jmeans = [],[],[],[],[]
    popmedians,divmedians,corepopmedians,coredivmedians,Jmedians = [],[],[],[],[]
    popQ1,divQ1,corepopQ1,coredivQ1,JQ1 = [],[],[],[],[]
    popQ3,divQ3,corepopQ3,coredivQ3,JQ3 = [],[],[],[],[]
    #if len(popmeans) > len(T_range):
    TTT = np.r_[271:320:3]
    #else: TTT = T_range
    
    # now cycle through all independent variables (stored in indep_now)
    # and find averages, medians etc
    for i in range(len(indep_now)): #17):
        samplesize_by_T.append(len(pops_by_T[i]))
        survived_by_T.append(len(pops_by_T[i].nonzero()[0]))
        print("survived: ",survived_by_T[i])
        if samplesize_by_T[i] == 0:
            survival_by_T.append(0)
        else:
            survival_by_T.append(survived_by_T[i]/samplesize_by_T[i])

        pops_this_T = np.array(pops_by_T[i]).astype('float')
        divs_this_T = np.array(divs_by_T[i]).astype('float')
        core_pops_this_T = np.array(core_pops_by_T[i]).astype('float')
        core_divs_this_T = np.array(core_divs_by_T[i]).astype('float')
        Jtot_this_T = np.array(interactions_by_T[i]).astype('float')

        #print(pops_this_T == 0)
        pops_this_T[pops_this_T==0] = np.nan
        divs_this_T[divs_this_T==0] = np.nan
        core_pops_this_T[core_pops_this_T==0] = np.nan
        core_divs_this_T[core_divs_this_T==0] = np.nan
        Jtot_this_T[Jtot_this_T==0] = np.nan

        popmeans.append(np.nanmean(pops_this_T))
        popQ1.append(np.nanquantile(pops_this_T,.25))
        popQ3.append(np.nanquantile(pops_this_T,.75))
        popmedians.append(np.nanquantile(pops_this_T,.5))
        
        divmeans.append(np.nanmean(divs_this_T))
        divQ1.append(np.nanquantile(divs_this_T,.25))
        divQ3.append(np.nanquantile(divs_this_T,.75))
        divmedians.append(np.nanquantile(divs_this_T,.5))

        corepopmeans.append(np.nanmean(core_pops_this_T))
        corepopQ1.append(np.nanquantile(core_pops_this_T,.25))
        corepopQ3.append(np.nanquantile(core_pops_this_T,.75))
        corepopmedians.append(np.nanquantile(core_pops_this_T,.5))

        coredivmeans.append(np.nanmean(core_divs_this_T))
        coredivQ1.append(np.nanquantile(core_divs_this_T,.25))
        coredivQ3.append(np.nanquantile(core_divs_this_T,.75))
        coredivmedians.append(np.nanquantile(core_divs_this_T,.5))

        Jmeans.append(np.nanmean(Jtot_this_T))
        JQ1.append(np.nanquantile(Jtot_this_T,.25))
        JQ3.append(np.nanquantile(Jtot_this_T,.75))
        Jmedians.append(np.nanquantile(Jtot_this_T,.5))
        
    print("Shape of survival_by_T: ",np.shape(survival_by_T))
    # Scatter populations agains survival probability
    f1,ax1 = plt.subplots() #figure()
    f2,ax2 = plt.subplots()
    f3,ax3 = plt.subplots()
    f4,ax4 = plt.subplots()
    i = -1
    for indep_var in indep_now: #[poff,pdeath]
        i += 1
        if len(indep_var) > 1:
            y_now = indep_var[1] # pdeath or pmut
            ax3.scatter(y_now*np.ones(len(pops_by_T[i])),pops_by_T[i])
            x_now = indep_var[0] # poff or mu
            ax4.scatter(x_now*np.ones(len(pops_by_T[i])),pops_by_T[i])
            ax1.scatter(survival_by_T[i]*np.ones(len(pops_by_T[i])),pops_by_T[i])
            ax2.scatter(survival_by_T[i]*np.ones(len(divs_by_T[i])),divs_by_T[i])

    ax1.set_xlabel("Survival probability")
    ax2.set_xlabel("Survival probability")
    if len(poff_range) > 1:
        ax3.set_xlabel("Death probability")
        ax4.set_xlabel("Scaling of reprod. prob.")
    elif len(mu_range) > 1:
        ax4.set_xlabel(r"$\mu$")
        ax3.set_xlabel("r$p_\mathrm{mut}$")
    ax1.set_ylabel("Final population")
    ax2.set_ylabel("Final diversity")
    ax3.set_ylabel("Final population")
    ax4.set_ylabel("Final population")

    
    if len(T_range) > 1:
        fig, ax = plt.subplots(2,2)
        # Final stats vs. temperature
        ax[0,0].plot(TTT,corepopmeans,label="mean")
        ax[0,0].plot(TTT,corepopmedians,"--",label="median")
        ax[0,0].fill_between(TTT,corepopQ1,corepopQ3,alpha=0.2)
        ax[0,0].plot(TTT,corepopQ1,":",label="Q1")
        ax[0,0].plot(TTT,corepopQ3,":",label="Q3")
        ax[0,0].set_ylim(0,1250)
        ax[0,0].set_xlim(TTT[0],TTT[-1])

        ax[0,1].plot(TTT,popmeans,label="mean")
        ax[0,1].plot(TTT,popmedians,"--",label="median")
        ax[0,1].fill_between(TTT,popQ1,popQ3,alpha=0.2)
        ax[0,1].plot(TTT,popQ1,":",label="Q1")
        ax[0,1].plot(TTT,popQ3,":",label="Q3")
        ax[0,1].set_ylim(0,1250)
        ax[0,1].set_xlim(TTT[0],TTT[-1])
        ax[0,1].set_xlim(TTT[0],TTT[-1])

        ax[1,0].plot(TTT,coredivmeans,label="mean")
        ax[1,0].plot(TTT,coredivmedians,"--",label="median")
        ax[1,0].fill_between(TTT,coredivQ1,coredivQ3,alpha=0.2)
        ax[1,0].plot(TTT,coredivQ1,":",label="Q1")
        ax[1,0].plot(TTT,coredivQ3,":",label="Q3")
        ax[1,0].set_ylim(0)
        ax[1,0].set_xlim(TTT[0],TTT[-1])

        ax[1,1].plot(TTT,divmeans,label="mean")
        ax[1,1].plot(TTT,divmedians,"--",label="median")
        ax[1,1].fill_between(TTT,divQ1,divQ3,alpha=0.2)
        ax[1,1].plot(TTT,divQ1,":",label="Q1")
        ax[1,1].plot(TTT,divQ3,":",label="Q3")
        ax[1,1].set_ylim(0)
        ax[1,1].set_xlim(TTT[0],TTT[-1])

        ax[1,1].legend()
        ax[1,1].set_xlabel("Temperature,T (K)")
        ax[1,0].set_xlabel("Temperature,T (K)")

        ax[0,0].set_ylabel(r"Avg. abundance,$\bar{N}$")
        ax[1,0].set_ylabel(r"Avg. diversity")

        ax[0,0].set_title("Core")
        ax[0,1].set_title("Ecosystem")
        
        
        plt.legend()
        plt.ylim(0)
        #plt.xlabel("Temperature,T (K)")
        #plt.ylabel(r"Avg. final abundance,$\bar{N}$")
        plt.savefig(locat+experiment+date+extra_folder+date+"/final_quartiles_"+experiment[:-1]+date+extra_folder[1:]+".pdf")

        # this one doesn't work well
        plt.figure()
        plt.plot(TTT,survival_by_T) #samplesize_by_T) #survival_by_T)
        plt.xlabel("Temperature (K)")
        plt.ylabel("Survival (%)")
        
        fig,ax = plt.subplots()
        ax.plot(TTT,Jmeans,label="mean")
        ax.fill_between(TTT,JQ1,JQ3,alpha=0.2)
        ax.plot(TTT,JQ1,":",label="Q1")
        ax.plot(TTT,JQ3,":",label="Q3")
        ax.set_ylim(0)
        
        ax.set_ylabel("Core-core interactions")
        ax.set_xlabel("Temperature,T (K)")
        
        plt.savefig(locat+experiment+date+extra_folder+"/final_interactions_"+experiment[:-1]+date+extra_folder[1:]+".pdf")
        plt.show()

    if len(poff_range) > 1:
#        fig,ax = plt.subplots(len(poff_range),len(pdeath_range))
        grid_surv = np.nan*np.zeros((len(pdeath_range),len(poff_range)))
        grid_abund = grid_surv.copy() # final abundance
        grid_div = grid_surv.copy()
        grid_N_med = grid_surv.copy()
        grid_D_med = grid_surv.copy()
        # poff along x axis, pdeath along y-axis
        col,i = -1,0
        for poff in poff_range:
            col += 1
            row = len(pdeath_range)
            for pdeath in pdeath_range:
                row -= 1
                i += 1
                if [poff,pdeath] != indep_now[i]:
                    print("error in poff,pdeath assignment")
                    poff, pdeath = indep_now[i]
                if survival_by_T[i] > 0:
                    grid_surv[row,col] = survival_by_T[i]
                grid_abund[row,col] = popmeans[i]
                grid_div[row,col] = divmeans[i]
                grid_N_med[row,col] = popmedians[i]
                grid_D_med[row,col] = divmedians[i]
        
        # define axes for survival, mean abundance and mean diversity
        sfig,sax = plt.subplots() # survival
        afig,aax = plt.subplots() # abundance
        dfig,dax = plt.subplots() # diversity
        
        slimits=[0,1]
        alimits=[200,2000]
        dlimits=[0,100]

        # make plots
        cmap = plt.get_cmap('YlGnBu') #Greys')
        # survival_image = sax.imshow(grid_surv,cmap=cmap,extent=[0,len(poff_range),0,len(pdeath_range)],interpolation='none')
        survival_image = sax.imshow(grid_surv,cmap=cmap,extent=[0,len(poff_range),0,len(pdeath_range)],interpolation='none',vmin=slimits[0],vmax=slimits[1])
        if plot_type == "mean":
            abundance_image = aax.imshow(grid_abund, cmap=cmap, extent=[0,len(poff_range),0,len(pdeath_range)],interpolation='none',vmin=alimits[0],vmax=alimits[1])
            diversity_image = dax.imshow(grid_div, cmap=cmap, extent=[0,len(poff_range),0,len(pdeath_range)],interpolation='none',vmin=dlimits[0],vmax=dlimits[1]) #vmin=30,vmax=90)
        else:
            abundance_image = aax.imshow(grid_N_med, cmap=cmap, extent=[0,len(poff_range),0,len(pdeath_range)],interpolation='none',vmin=alimits[0],vmax=alimits[1]) #,vmin=300,vmax=1400)
            diversity_image = dax.imshow(grid_D_med, cmap=cmap, extent=[0,len(poff_range),0,len(pdeath_range)],interpolation='none',vmin=dlimits[0],vmax=dlimits[1])

        x_labels,y_labels = [],[]
        for poff in poff_range:
            x_labels.append(f"{poff:.2f}")
        for pdeath in pdeath_range:
            y_labels.append(f"{pdeath:.2f}")

        for ax in [sax,aax,dax]:
            ax.grid(which='major',axis='both',linestyle='-',color='k',linewidth=2)
            ax.set_xticks(np.r_[1:len(poff_range)+1:1])
            ax.set_yticks(np.r_[1:len(pdeath_range)+1:1])
            ax.set_xticklabels(x_labels) #,rotation=90)
            ax.set_yticklabels(y_labels)
            ax.set_xlabel(r"scaling of $p_{off}$")
            ax.set_ylabel(r"$p_{death}$")

        cbar = sfig.colorbar(survival_image, label="survival probability", ax=sax, extend='both')
        cbar.minorticks_on()
        cbar = afig.colorbar(abundance_image, label=f"{plot_type} final abundance", ax=aax, extend='both')
        cbar.minorticks_on()
        cbar = dfig.colorbar(diversity_image, label=f"{plot_type} final diversity", ax=dax, extend='both')
        cbar.minorticks_on()
        
        # scale plot 
        scalex = len(poff_range)/max(poff_range)
        scaley = len(pdeath_range)/max(pdeath_range)
        #sax.plot(scalex*poff_range,scaley*poff_range**2,"--r",label="p(surv. & reprod.)")
        sax.set_ylim(0,scaley*max(pdeath_range))
        sax.legend()
            
        sax.set_title("Survival probability")
        aax.set_title(f"M{plot_type[1:]} abundance after {maxgens} gen.")
        dax.set_title(f"M{plot_type[1:]} diversity after {maxgens} gen.")

        sfig.savefig(locat+experiment+date+extra_folder+"/survival_grid_"+experiment[:-1]+date+extra_folder[1:]+".pdf")
        afig.savefig(locat+experiment+date+extra_folder+"/abundance_grid_"+experiment[:-1]+date+extra_folder[1:]+".pdf")
        #plt.imsave(locat+experiment+date+extra_folder+"/abundance_imsave_"+experiment[:-1]+date+extra_folder[1:]+".pdf")
        dfig.savefig(locat+experiment+date+extra_folder+"/diversity_grid_"+experiment[:-1]+date+extra_folder[1:]+".pdf")
        #plt.imsave(locat+experiment+date+extra_folder+"/diversity_imsave_"+experiment[:-1]+date+extra_folder[1:]+".pdf")

        plt.show()

    if len(mu_range) > 1:
#        fig,ax = plt.subplots(len(poff_range),len(pdeath_range))
        grid_surv = np.nan*np.zeros((len(pmut_range),len(mu_range)))
        grid_abund = grid_surv.copy() # final abundance
        grid_div = grid_surv.copy()
        grid_N_med = grid_surv.copy()
        grid_D_med = grid_surv.copy()
        # poff along x axis, pdeath along y-axis
        col,i = -1,0
        for mu in mu_range:
            col += 1
            row = len(pmut_range)
            for pmut in pmut_range:
                row -= 1
                i += 1
                if [mu,pmut] != indep_now[i]:
                    print("error in poff,pdeath assignment")
                    mu, pmut = indep_now[i]
                if survival_by_T[i] > 0:
                    grid_surv[row,col] = survival_by_T[i]
                grid_abund[row,col] = popmeans[i]
                grid_div[row,col] = divmeans[i]
                grid_N_med[row,col] = popmedians[i]
                grid_D_med[row,col] = divmedians[i]
        
        # define axes for survival, mean abundance and mean diversity
        sfig,sax = plt.subplots() # survival
        afig,aax = plt.subplots() # abundance
        dfig,dax = plt.subplots() # diversity

        slimits=[0,1]
        alimits=[200,2000]
        dlimits=[0,100]

        # make plots
        cmap = plt.get_cmap('YlGnBu') #Greys')
        #survival_image = sax.imshow(grid_surv,cmap=cmap,extent=[0,len(mu_range),0,len(pmut_range)],interpolation='none')
        survival_image = sax.imshow(grid_surv,cmap=cmap,extent=[0,len(mu_range),0,len(pmut_range)],interpolation='none',vmin=slimits[0],vmax=slimits[1])
        if plot_type == "mean":
            abundance_image = aax.imshow(grid_abund, cmap=cmap, extent=[0,len(mu_range),0,len(pmut_range)],interpolation='none',vmin=alimits[0],vmax=alimits[1])
            diversity_image = dax.imshow(grid_div, cmap=cmap, extent=[0,len(mu_range),0,len(pmut_range)],interpolation='none',vmin=dlimits[0],vmax=dlimits[1])
        else:
            abundance_image = aax.imshow(grid_N_med, cmap=cmap, extent=[0,len(mu_range),0,len(pmut_range)],interpolation='none',vmin=alimits[0],vmax=alimits[1])
            diversity_image = dax.imshow(grid_D_med, cmap=cmap, extent=[0,len(mu_range),0,len(pmut_range)],interpolation='none',vmin=dlimits[0],vmax=dlimits[1])

        x_labels,y_labels = [],[]
        for mu in mu_range:
            x_labels.append(f"{mu:.2f}")
        for pmut in pmut_range:
            y_labels.append(f"{pmut:.3f}")

        for ax in [sax,aax,dax]:
            ax.grid(which='major',axis='both',linestyle='-',color='k',linewidth=2)
            ax.set_xticks(np.r_[1:len(mu_range)+1:1])
            ax.set_yticks(np.r_[1:len(pmut_range)+1:1])
            ax.set_xticklabels(x_labels) #,rotation=90)
            ax.set_yticklabels(y_labels)
            ax.set_xlabel(r"$\mu$")
            ax.set_ylabel(r"$p_{mut}$")

        cbar = sfig.colorbar(survival_image, label="survival probability", ax=sax, extend='both')
        cbar.minorticks_on()
        cbar = afig.colorbar(abundance_image, label=f"{plot_type} final abundance", ax=aax, extend='both')
        cbar.minorticks_on()
        cbar = dfig.colorbar(diversity_image, label=f"{plot_type} final diversity", ax=dax, extend='both')
        cbar.minorticks_on()
        
        # scale plot 
        scalex = len(mu_range)/max(mu_range)
        scaley = len(pmut_range)/max(pmut_range)
        #sax.plot(scalex*poff_range,scaley*poff_range**2,"--r",label="p(surv. & reprod.)")
        sax.set_ylim(0,scaley*max(pmut_range))
        sax.legend()
            
        sax.set_title("Survival probability")
        aax.set_title(f"M{plot_type[1:]} abundance after {maxgens} gen.")
        dax.set_title(f"M{plot_type[1:]} diversity after {maxgens} gen.")

        sfig.savefig(locat+experiment+extra_folder+date+"/survival_grid_"+experiment[:-1]+date+extra_folder[:-1],format="pdf")
        afig.savefig(locat+experiment+extra_folder+date+"/abundance_grid_"+experiment[:-1]+date+extra_folder[:-1],format="pdf")
        #plt.imsave(locat+experiment+date+extra_folder+"/abundance_imsave_"+experiment[:-1]+date+extra_folder[1:]+".pdf")
        dfig.savefig(locat+experiment+extra_folder+date+"/diversity_grid_"+experiment[:-1]+date+extra_folder[:-1],format="pdf")
        #plt.imsave(locat+experiment+date+extra_folder+"/diversity_imsave_"+experiment[:-1]+date+extra_folder[1:]+".pdf")

        plt.show()









    
    #mean_pop_by_T = np.nanmean(pops_by_T,axis=1)
    #quant1_pop_by_T = np.nanquantile(pops_by_T,.25,axis=1)
    #quant2_pop_by_T = np.nanquantile(pops_by_T,.75,axis=1)
    #ax.plot(temperatures,mean_pop_by_T,label="mean")
    #ax.plot(temperatures,quant1_pop_by_T,":",label="Q1")
    #ax.plot(temperatures,quant2_pop_by_T,":",label="Q3")
    #plt.show()



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

        print(f"T: {T_range}, number of boxes: {len(pops_by_T)}, {np.shape(pops_by_T)}")

        positions = np.insert(T_range,0,T_range[0] - 5)
        xlabels = np.array(T_range,dtype=str)
        xlabels = np.insert(xlabels,0,'TNM')
        fig,ax = plt.subplots(3,2,sharex='all') #True)
        part1 = ax[0,0].violinplot(pops_by_T[:],positions=positions,widths=4,showextrema=False,showmedians=True)
        part2 = ax[1,0].violinplot(divs_by_T[:],positions=positions,widths=4,showextrema=False,showmedians=True)
        # ax[1,0].tick_params(labelrotation=90)
        
        part3 = ax[0,1].violinplot(core_pops_by_T[:],positions=positions,widths=4,showextrema=False,showmedians=True)
        part4 = ax[1,1].violinplot(core_divs_by_T[:],positions=positions,widths=4,showextrema=False,showmedians=True)
        part5 = ax[2,0].violinplot(interactions_by_T[:],positions=positions,widths=4,showextrema=False,showmedians=True)
        ax[2,1].plot(T_range,TM.poff_total(0,T_range)-TM.pdeath(T_range),color='b')

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
        plt.savefig(locat+experiment+date+extra_folder+"/violins_"+experiment[:-1]+date+extra_folder[1:]+".pdf")
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
    populations, diversities = get_pop_div(seed,T_range,poff,pdeath)
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
    plt.title(f"{experiment} {date} {seed} {T_range}")
    plt.show()



