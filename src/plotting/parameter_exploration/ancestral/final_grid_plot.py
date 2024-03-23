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
from scipy.interpolate import griddata
module_locat = "/home/cfebvre/repos/the_model_y3/src/geoTNM" # "~/repos/the_model_y3/src/geoTNM"
if os.path.exists(module_locat):
    # import modules from output location
    sys.path.append(module_locat)
    import MTE_TPC_combo as Teq
    from analysis import analyze_J
else: print("Error: cannot ipmort geoTNM modules from ",module_locat)
#if '/home/cfebvre/repos/tnm_febvre/geochem' not in sys.path:
#    sys.path.append('/home/cfebvre/repos/tnm_febvre/geochem')
# if 'C:/Users/camil/repos/tnm_febvre/geochem' not in sys.path:
#     sys.path.append('C:/Users/camil/repos/tnm_febvre/geochem')
# from geoTNM import temperature_effects as TM
#rc('text', usetex=True)
#plt.rcParams.update({"text.usetex": True})

show_TRC = True
verbose = True
# User inputs
plot_type = "median" # "med", "mean"
temp_script = "temp_effects" # with which to plot TRC
maxgens = 10_000
sample_size = 50
num_tries=200 # number of other filenames to try  
plotting = True
plot_multiple = True
if temp_script == "met_theory":
    import met_theory_simple as TM
# Constants
locat = '/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/'
BasicTNM_locat = locat + 'BasicTNM/Jun_17/'
sin_width = 3.54 # single column width
onehalf_width = 5.51 # 1.5 column width
doub_width = 7.48 # full page width


class Experiment:
    def __init__(self,date="",seed=np.r_[1000:1050],pdeath_range=np.r_[.1:1:.1],poff_range=np.r_[.1:1:.1],T_range=[False],dates=[],experiment="prob_test/",extra_folder="",L=20,C=100,theta=.25,mu=.1,pmut=0.01,locat='/net/venus/kenes/user/cfebvre/out_venus/TTNM_outputs/'):
        self.date = date
        self.seed = seed
        self.pdeath_range = pdeath_range
        self.poff_range = poff_range
        self.T_range = T_range
        self.dates = dates
        self.experiment = experiment
        self.extra_folder = extra_folder
        self.L = L
        self.C = C
        self.theta = theta
        self.mu = mu
        self.pmut = pmut
        self.locat = locat

# Define each experiment run
mu005 = Experiment(dates=['Mar_20_23'],extra_folder='mu0.05/',mu=0.05)
mu002 = Experiment(dates=['Mar_20_23'],extra_folder='mu0.02/',mu=0.02)
mu02 = Experiment(dates=['Feb_28'],extra_folder='/mu_.2',mu=0.2,locat = '/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/',poff_range=np.r_[.1:1:.1],pdeath_range=np.r_[.1:1:.1])
mu01 = Experiment(dates=['Feb_09'],extra_folder='/D_init_60',mu=0.1,locat = '/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/',poff_range=np.r_[.05:1.05:.05],pdeath_range=np.r_[.05:1:.05])

# for which of these experiments do you want plots
plots_to_make = [mu01] #[mu005]


# %% ---------------------
# assemble filename of input
def find_file(date,seed,T=False,poff=False,pdeath=False,type="pypy",C='',mu='',theta='',pmut='',L=20):
    # return locat+f"pypy_seed{str(seed)}{date}_{str(T)}K_TNM_L20.dat"
    #print("* * * f i n d i n g   f i l e * * * ")
    #print("looking for ",type," file")
    date2 = date
    if type == "pypy": 
        #print("entered pypy for loop")/
        if poff:
            if locat.startswith('/net/'):
                file = f"{locat}{experiment}{extra_folder}{date}/pypy_seed{str(seed)}{date[:6]}_poff{poff:.2f}_pdeath{pdeath:.2f}K_TNM{C}{mu}{theta}{pmut}_L{L}.dat"
            else:
                file = f"{locat}{experiment}{date}{extra_folder}/pypy_seed{str(seed)}{date}_poff{poff:.2f}_pdeath{pdeath:.2f}K_TNM_L{L}.dat"
        else: file = f"{locat}{experiment}{date}{extra_folder}/pypy_seed{str(seed)}{date}_{str(T)}K_TNM_L{L}.dat"
        #print("file: ",file)
        #print(exists(file))
        if not exists(file):
            #print("pypy file not found on first try, checking again")
            # check other possible dates for filename if this filename wasn't found

            for j in range(0,num_tries): #,end): # np.r_[0:end]: #len(other_file_dates)]:
                #print(j," out of ",end)
                if locat.startswith('/net/'):
                    date1 = datetime.strptime(date,"%b_%d_%y")    
                else: date1 = datetime.strptime(date,"%b_%d")
                adjusted_date = date1 + timedelta(days=j)
                date2 = adjusted_date.strftime("%b_%d")  #other_file_dates[j]
                #print('checking ',date2,' to see if it exists...')
                if poff:
                    if locat.startswith('/net/'):
                        file = f"{locat}{experiment}{extra_folder}{date}/pypy_seed{str(seed)}{date2}_poff{poff:.2f}_pdeath{pdeath:.2f}K_TNM{C}{mu}{theta}{pmut}_L{L}.dat"
                    else:
                        if j > 30:
                            file = f"{locat}{experiment}{date}{extra_folder}/pypy_seed{str(seed)}{date2}_poff{poff:.2f}_pdeath{pdeath:.2f}K_TNM_C100_mu0.10_theta0.25_pmut0.010_L{L}.dat"
                        else:
                            file = f"{locat}{experiment}{date}{extra_folder}/pypy_seed{str(seed)}{date2}_poff{poff:.2f}_pdeath{pdeath:.2f}K_TNM_L{L}.dat"
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
            if locat.startswith('/net/'):
                file = f"{locat}{experiment}{extra_folder}{date}/diversity_seed{str(seed)}{date[:6]}_poff{poff:.2f}_pdeath{pdeath:.2f}K_TNM{C}{mu}{theta}{pmut}_L{L}.dat"
            else:
                file = f"{locat}{experiment}{date}{extra_folder}/diversity_seed{str(seed)}{date}_poff{poff:.2f}_pdeath{pdeath:.2f}K_TNM_L{L}.dat"
        else:
            file = f"{locat}{experiment}{date}{extra_folder}/diversity_seed{str(seed)}{date}_{str(T)}K_TNM_L{L}.dat"
        if not exists(file):
            file2 = 0
            #print("div file not found on first try")
            # check other possible dates for filename
            #end = len(other_file_dates)
            for j in range(0,num_tries): #,end): # np.r_[0:end]: #len(other_file_dates)]:
                #print(j," out of ",end)
                if locat.startswith('/net/'):
                    date1 = datetime.strptime(date,"%b_%d_%y")    
                else: date1 = datetime.strptime(date,"%b_%d")
                adjusted_date = date1 + timedelta(days=j)
                if locat.startswith('/net/'): date2 = adjusted_date.strftime("%b_%d_%y")
                else: date2 = adjusted_date.strftime("%b_%d")  #other_file_dates[j]
                #print('checking ',date2,' to see if it exists...')
                if poff:
                    if locat.startswith("/net/"):
                        file = f"{locat}{experiment}{extra_folder}{date}/diversity_seed{str(seed)}{date[:6]}_poff{poff:.2f}_pdeath{pdeath:.2f}K_TNM{C}{mu}{theta}{pmut}_L{L}.dat"
                    else:
                        if j > 30:
                            file = f"{locat}{experiment}{date}{extra_folder}/diversity_seed{str(seed)}{date2}_poff{poff:.2f}_pdeath{pdeath:.2f}K_TNM_C100_mu0.10_theta0.25_pmut0.010_L{L}.dat"
                        else:
                            file = f"{locat}{experiment}{date}{extra_folder}/diversity_seed{str(seed)}{date2}_poff{poff:.2f}_pdeath{pdeath:.2f}K_TNM_L{L}.dat"
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

def get_pop_div(date,seed,T=False,poff=False,pdeath=False,type="pypy",filename=None,C='',mu='',theta='',pmut='',L=20):
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
        interactions = analyze_J.final_interaction(div_file,date,sd,T=T,locat=BasicTNM_locat)
    else:
        div_file,date2 = find_file(date,sd,T=temp,poff=poff,pdeath=pdeath,type=type)
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

# %% ---------------------
# %% Loop through all experiments
for this_plot in plots_to_make:
    seed = this_plot.seed
    pdeath_range = this_plot.pdeath_range
    poff_range = this_plot.poff_range
    T_range = this_plot.T_range
    dates = this_plot.dates
    experiment = this_plot.experiment
    extra_folder = this_plot.extra_folder
    L,C,theta,mu,pmut = this_plot.L, this_plot.C, this_plot.theta, this_plot.mu, this_plot.pmut
    locat = this_plot.locat
    temp = False

#    # import modules from one of the folders
#    found = 0
#    for d in dates:
#        module_locat = f"{locat}{experiment}{d}{extra_folder}"
#        if os.path.exists(module_locat):
#            found = 1
#            # import modules from output location
#            sys.path.append(module_locat)
#            print("****locat: ",sys.path[-1],"****")
#            import temperature_effects as Teq
#            from analysis import analyze_J
#            break
#    if not found: print("Could not find modules. Checked: ",f"{locat}{experiment}{d}{extra_folder}")

    # produce output file and write date and time in it
    error_file = "OUT_from_final_grid_plot.txt"
    with open(error_file,'a') as f:
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string)

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
        for poff in poff_range:
            for pdeath in pdeath_range:
                # keep track of independent variables in 
                indep_now.append([poff,pdeath])
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
                                if locat.startswith('/net/'):
                                    gens_run,pop_temp,div_temp,core_pop,core_div = get_pop_div(date,sd,T=temp,poff=poff,pdeath=pdeath,mu=f"_mu{mu:.2f}",pmut=f"_pmut{pmut:.3f}",C=f"_C{C}",theta=f"_theta{theta:.2f}",L=L)
                                else:
                                    gens_run,pop_temp,div_temp,core_pop,core_div = get_pop_div(date,sd,T=temp,poff=poff,pdeath=pdeath)
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
            pd_now = indep_var[1]
            ax3.scatter(pd_now*np.ones(len(pops_by_T[i])),pops_by_T[i])
            po_now = indep_var[0]
            ax4.scatter(po_now*np.ones(len(pops_by_T[i])),pops_by_T[i])
            ax1.scatter(survival_by_T[i]*np.ones(len(pops_by_T[i])),pops_by_T[i])
            ax2.scatter(survival_by_T[i]*np.ones(len(divs_by_T[i])),divs_by_T[i])

    ax1.set_xlabel("Survival probability")
    ax2.set_xlabel("Survival probability")
    ax3.set_xlabel("Death probability")
    ax4.set_xlabel("Scaling of reprod. prob.")
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
        plt.savefig(locat+experiment+date+extra_folder+"/final_quartiles_"+experiment[:-1]+date+extra_folder[1:]+".pdf")

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
        # scatter plot poff/pdeath vs survival
        ratios = []
        differences = []
        for poff in poff_range:
            for pdeath in pdeath_range:
                ratios.append(poff/pdeath)
                differences.append(poff-pdeath)
        fig,ax = plt.subplots(1,2,sharey=True)
        fsurv,asurv = plt.subplots()
        print("shape of survival_by_T: ",np.shape(survival_by_T))
        for a in [ax[0],asurv]:
            a.scatter(ratios,survival_by_T[:len(ratios)])
            a.set_xlabel(r"p$_\mathrm{off,scaler}$/p$_\mathrm{death}$")
            a.set_ylabel("Survival Fraction")

        ax[1].scatter(differences,survival_by_T[:len(ratios)])
        ax[1].set_xlabel(r"p$_\mathrm{off}$ - p$_\mathrm{death}$")
        
        np.save(f"survival_ratio-po-pd_{experiment[:-1]}{date}{extra_folder[1:]}.npy",np.array([survival_by_T[:len(ratios)],ratios]))
        fig.savefig("figures/poff-pdeath-ratios-differences.pdf")

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

        median_N_overall = np.nanmedian(grid_N_med)
        np.save("expected_N.npy",np.array(median_N_overall))
        
        # define axes for survival, mean abundance and mean diversity
        sfig,sax = plt.subplots() # survival
        afig,aax = plt.subplots() # abundance
        dfig,dax = plt.subplots() # diversity
        multi_fig,multi_ax = plt.subplots(3,sharex=True,sharey=True,figsize=(sin_width,1.8*sin_width))
        all_sfig,all_sax = plt.subplots(figsize=(doub_width,.7*doub_width))
        s_v_ratio_fig,s_v_ratio_ax = plt.subplots(2,figsize=(sin_width,1.6*sin_width))
        #grey_cmap = plt.get_cmap("Greys")
        #colors_surv = grey_cmap(np.array(differences)/max(differences))
        s_v_ratio_ax[1].scatter(ratios,survival_by_T[:len(ratios)],s=5,c='k') #colors_surv)
        s_v_ratio_ax[1].set_xlabel(r"p$_\mathrm{off,scaler}$/p$_\mathrm{death}$")
        s_v_ratio_ax[1].set_ylabel("Survival Fraction")
        s_v_ratio_ax[1].set_xlim(0,6)
        s_v_ratio_ax[1].set_xticks(range(6))

        # make plots
        interpolation = "none" #"sinc"
        cmap = plt.get_cmap('YlGnBu') #Greys')
        survival_image = sax.imshow(grid_surv,cmap=cmap,extent=[0,len(poff_range),0,len(pdeath_range)],interpolation=interpolation)
        surv_im2 = multi_ax[0].imshow(grid_surv,cmap=cmap,extent=[0,len(poff_range),0,len(pdeath_range)],interpolation=interpolation)
        surv_im3 = s_v_ratio_ax[0].imshow(grid_surv,cmap=cmap,extent=[0,len(poff_range),0,len(pdeath_range)],interpolation=interpolation)
        if plot_type == "mean":
            abundance_image = aax.imshow(grid_abund, cmap=cmap, extent=[0,len(poff_range),0,len(pdeath_range)],interpolation=interpolation,vmin=300,vmax=1400)
            diversity_image = dax.imshow(grid_div, cmap=cmap, extent=[0,len(poff_range),0,len(pdeath_range)],interpolation=interpolation,vmin=30,vmax=90)
        else:
            abundance_image = aax.imshow(grid_N_med, cmap=cmap, extent=[0,len(poff_range),0,len(pdeath_range)],interpolation=interpolation,vmin=300,vmax=1400)
            diversity_image = dax.imshow(grid_D_med, cmap=cmap, extent=[0,len(poff_range),0,len(pdeath_range)],interpolation=interpolation,vmin=30,vmax=90)
            ab_im2 = multi_ax[1].imshow(grid_N_med, cmap=cmap, extent=[0,len(poff_range),0,len(pdeath_range)],interpolation=interpolation,vmin=300,vmax=1400)
            div_im2 = multi_ax[2].imshow(grid_D_med, cmap=cmap, extent=[0,len(poff_range),0,len(pdeath_range)],interpolation=interpolation,vmin=30,vmax=90)

        x_labels,y_labels = [],[]
        skip = 1
        for poff in poff_range:
            if not skip:
                x_labels.append(f"{poff:.2f}")
                skip = 1
            else: 
                x_labels.append("")
                skip = 0
        skip = 1
        for pdeath in pdeath_range:
            if not skip:
                y_labels.append(f"{pdeath:.2f}")
                skip = 1
            else: 
                y_labels.append("")
                skip = 0

        # label ticks
        for fig,ax in zip([sfig,afig,dfig,multi_fig,multi_fig,multi_fig,s_v_ratio_fig],[sax,aax,dax,multi_ax[0],multi_ax[1],multi_ax[2],s_v_ratio_ax[0]]):
            ax.grid(which='major',axis='both',visible=False) #linestyle='-',color='k',linewidth=2)
            if ax != multi_ax[0] and ax != multi_ax[1]:
                ax.set_xticks(np.r_[1:len(poff_range)+1:1])
                ax.set_xticklabels(x_labels,rotation=90)
                ax.set_xlabel(r"p$_\mathrm{off,scaler}$")
            ax.set_yticks(np.r_[1:len(pdeath_range)+1:1])
            ax.set_yticklabels(y_labels)
            ax.set_ylabel(r"$p_{death}$")
            if fig != multi_fig:
                fig.subplots_adjust(bottom = 0.2)
            plt.tight_layout()

        cbar = sfig.colorbar(survival_image, label="survival probability", ax=sax, extend='neither')
        cbar.minorticks_on()
        cbar = multi_fig.colorbar(surv_im2, label="survival probability",ax=multi_ax[0],extend='neither',pad=0.1) #,shrink=0.8)
        cbar.minorticks_on()
        cbar = s_v_ratio_fig.colorbar(survival_image, label="survival probability", ax=s_v_ratio_ax[0], extend='neither')
        cbar.minorticks_on()
        cbar = afig.colorbar(abundance_image, label=f"{plot_type} final abundance", ax=aax, extend='both')
        cbar.minorticks_on()
        cbar = multi_fig.colorbar(ab_im2, label=f"{plot_type} final abundance", ax=multi_ax[1], extend='both',pad=0.1) #,shrink=0.8)
        cbar.minorticks_on()
        cbar = dfig.colorbar(diversity_image, label=f"{plot_type} final diversity", ax=dax, extend='both')
        cbar.minorticks_on()
        cbar = multi_fig.colorbar(div_im2, label=f"{plot_type} final diversity", ax=multi_ax[2], extend='both',pad=0.1) #,shrink=0.8)
        cbar.minorticks_on()
        
        # scale plot 
        scalex = len(poff_range)/max(poff_range)
        scaley = len(pdeath_range)/max(pdeath_range)
        #sax.plot(scalex*poff_range,scaley*poff_range**2,"--r",label="p(surv. & reprod.)")
        sax.set_ylim(0,scaley*max(pdeath_range))
        multi_ax[0].set_ylim(0,scaley*max(pdeath_range))
        
        if show_TRC:
            # lay TRC ontop
            all_temps = np.r_[274:320]
            experiment_temps = np.r_[274:320:3]
            poff_T = Teq.poff_T(all_temps)
            pdeath_T = Teq.pdeath(all_temps)
            sax.plot(len(poff_range)*poff_T,len(pdeath_range)*pdeath_T,"r",label="TRC")
            for ax in multi_ax:
                ax.plot(len(poff_range)*poff_T,len(pdeath_range)*pdeath_T,"r",label="TRC")

            s_v_ratio_ax[0].plot(len(poff_range)*poff_T,len(pdeath_range)*pdeath_T,"r",label="TRC")

            for Ti in experiment_temps:
                x = len(poff_range)*Teq.poff_T(Ti)
                y = len(pdeath_range)*Teq.pdeath(Ti)
                for ax in [sax,multi_ax[0],multi_ax[1],multi_ax[2],s_v_ratio_ax[0]]:
                    ax.plot(x,y,'ro')
                    if Ti in experiment_temps[1::3]:
                        ax.text(x,y+1,Ti,rotation=90,color='r')
                #multi_ax[0].plot(x,y,'ro')
                #multi_ax[0].text(x+.02,y+.02,Ti,rotation=45,color='r')
            #poff_T = factor*Teq.poff_T(experiment_temps)
            #pdeath_T = Teq.pdeath(experiment_temps)
            #sax.plot(len(poff_range)*poff_T,len(pdeath_range)*pdeath_T,"ro")
            #interpolate survival values at each T
            pred_fig, pred_ax = plt.subplots()
            # find interval between each poff
            interval = (max(poff_range)-min(poff_range))/(len(poff_range)-1)
            # include final point in poff_range 
            fill_last_col = False # True if you don't have points for poff = 1
            if fill_last_col:
                poff_extended = np.r_[min(poff_range):max(poff_range)+2*interval:interval]
                # make grid of poff, pdeath values
                poff_grid,pdeath_grid = np.meshgrid(poff_extended,pdeath_range)
            else: #poff_extended = poff_range
                # make grid of poff, pdeath values
                poff_grid,pdeath_grid = np.meshgrid(poff_range,pdeath_range)
            # convert to lists of coordinates
            points = np.array([poff_grid.flatten(),pdeath_grid.flatten()]).T
            print("points: \n",points)
            # extend survival values with a column of ones on right side--assume 100% survival for that column
            if fill_last_col:
                surv_extended = np.hstack((grid_surv[-1::-1],np.ones((len(poff_range),1))))
                # survival values corresponding with coordinates in "points"
                values = surv_extended.flatten()
            else: values = grid_surv[-1::-1].flatten()
            for i in range(len(values)):
                if np.isnan(values[i]):
                    values[i] = 0
            print("survival grid: \n",grid_surv)
            print("flattened: \n",values)

            # interpolate points at poff_T,pdeath_T corresponding to poff,pdeath for every 1K between 274 and 320K
            #predicted_surv_at_T = griddata(points,values,(poff_T,pdeath_T),method="linear")
            #pred_ax.plot(all_temps,predicted_surv_at_T,label="linear")
            #predicted_surv_at_T = griddata(points,values,(poff_T,pdeath_T),method="nearest")
            #pred_ax.plot(all_temps,predicted_surv_at_T,label="nearest")
            predicted_surv_at_T = griddata(points,values,(poff_T,pdeath_T)) #,method="cubic")
            print("Predicted survival: ",predicted_surv_at_T)
            pred_ax.plot(all_temps,predicted_surv_at_T,label="cubic")
            #pred_ax.legend()
            pred_ax.set_xlabel("Temperature, T (K)")
            pred_ax.set_ylabel("Predicted survival")
            all_sax.plot(all_temps,predicted_surv_at_T,"r--",label=r"Survival: p$_\mathrm{mut}$=0.01")
            all_sax.set_xlabel("Temperature,T (K)")
            all_sax.set_ylabel("Fraction")
            #multi_ax[1].set_aspect(20,anchor="S")
            #multi_fig.subplots_adjust(left=.15,bottom = 0.08) #,top=.9)

        # plot survival on multi_ax[1]
        singleTRC_surv = np.load("npy_files/survival_single-TRC_Jun_21_23.npy")/50
        varTRC_surv = np.load("npy_files/survival_var-TRC_Apr_13_23.npy")/50
        all_sax.plot(np.r_[274:320:3],singleTRC_surv,"k",label="Survival: Single TRC")
        all_sax.plot(np.r_[274:320:3],varTRC_surv,"k",alpha=.5,label="Survival: Various TRC")
        all_sax.plot(all_temps,Teq.poff_T(all_temps),"k-.",alpha=.2,label=r"p$_\mathrm{off,T}$: Single TRC")
        all_sax.plot(all_temps,Teq.pdeath(all_temps),"k:",alpha=.2,label=r"p$_\mathrm{death}$")
        #all_sax.plot(all_temps,Teq.poff_T(all_temps) - Teq.pdeath(all_temps),linestyle="loosely dashed",c="k",alpha=.2,label=r"single TRC r$_\mathrm{max}$")
        all_sax.set_ylim(0,1)

        if show_TRC: 
            sax.legend()
            multi_ax[0].legend()
            multi_ax[1].legend()
            multi_ax[2].legend()
            all_sax.legend()

        # label with a b c
        letter = iter(["a)","b)","c)"])
        for a in multi_ax:
            a.text(0.05,0.9,next(letter),transform=a.transAxes,horizontalalignment="left",verticalalignment="center")

        all_sfig.tight_layout()
            
        sax.set_title("Survival probability")
        aax.set_title(f"M{plot_type[1:]} abundance after {maxgens} gen.")
        dax.set_title(f"M{plot_type[1:]} diversity after {maxgens} gen.")

        if locat.startswith('/net/'):
            sfig.savefig(locat+experiment+extra_folder+date+"/survival_grid_"+experiment[:-1]+date+extra_folder[:-1]+".pdf")
            s_v_ratio_fig.savefig(locat+experiment+extra_folder+date+"/survival_grid_and_ratios_"+experiment[:-1]+date+extra_folder[:-1]+".pdf")
            afig.savefig(locat+experiment+extra_folder+date+"/abundance_grid_"+experiment[:-1]+date+extra_folder[:-1]+".pdf")
            #plt.imsave(locat+experiment+date+extra_folder+"/abundance_imsave_"+experiment[:-1]+date+extra_folder[1:]+".pdf")
            dfig.savefig(locat+experiment+extra_folder+date+"/diversity_grid_"+experiment[:-1]+date+extra_folder[:-1]+".pdf")
            multi_fig.savefig("figures/survival_with_interpolation"+experiment[:-1]+date+extra_folder[:-1]+".pdf")
        else:
            if show_TRC:
                np.save(f"predicted_survival_{experiment[:-1]}{date}{extra_folder[1:]}.npy",np.array(predicted_surv_at_T))
            multi_fig.subplots_adjust(top=.95)
            sfig.savefig(locat+experiment+date+extra_folder+"/survival_grid_"+experiment[:-1]+date+extra_folder[1:]+".pdf")
            sfig.savefig(locat+experiment+date+extra_folder+"/survival_grid_and_ratios_"+experiment[:-1]+date+extra_folder[1:]+".pdf")
            afig.savefig(locat+experiment+date+extra_folder+"/abundance_grid_"+experiment[:-1]+date+extra_folder[1:]+".pdf")
            #plt.imsave(locat+experiment+date+extra_folder+"/abundance_imsave_"+experiment[:-1]+date+extra_folder[1:]+".pdf")
            dfig.savefig(locat+experiment+date+extra_folder+"/diversity_grid_"+experiment[:-1]+date+extra_folder[1:]+".pdf")
            multi_fig.savefig("figures/all_grids_with_interpolation"+experiment[:-1]+date+extra_folder[1:]+".pdf")
            all_sfig.savefig("figures/survival_all_experiments.pdf")
        #plt.imsave(locat+experiment+date+extra_folder+"/diversity_imsave_"+experiment[:-1]+date+extra_folder[1:]+".pdf")

        print("Current directory: ",os.path.abspath(os.getcwd()))

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



