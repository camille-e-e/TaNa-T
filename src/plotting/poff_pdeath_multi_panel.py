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
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
outpath = '/net/venus/kenes/user/cfebvre/out_venus/TTNM_outputs/prob_test/control/Mar_10_23'
sys.path.append(outpath)
from analysis import analyze_J 
#rc('text', usetex=True)
#plt.rcParams.update({"text.usetex": True})

show_TRC = False
plot_scatters = False
if show_TRC: 
    from geoTNM import MTE_TPC_combo as Teq
verbose = True
# User inputs
plot_type = "median" # "med", "mean"
temp_script = "temp_effects" # with which to plot TRC
maxgens = 10_000
sample_size = 50
num_tries=40 # number of other filenames to try  
plotting = True
plot_multiple = True
if temp_script == "met_theory":
    import met_theory_simple as TM
# Constants
locat = '/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/'
BasicTNM_locat = locat + 'BasicTNM/Jun_17/'
fig_locat = '/net/venus/kenes/user/cfebvre/out_venus/TTNM_outputs/' # FINISH THIS *******
sin_width = 3.54 # one column in JTB
doub_width = 7.48 # full width in JTB

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
mu01 = Experiment(dates=['Feb_09'],extra_folder='/D_init_60',mu=0.1,locat = '/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/',poff_range=np.r_[.1:1:.1],pdeath_range=np.r_[.1:1:.1])
mu02 = Experiment(dates=['Feb_28'],extra_folder='/mu_.2',mu=0.2,locat = '/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/',poff_range=np.r_[.1:1:.1],pdeath_range=np.r_[.1:1:.1])
mu03 = Experiment(dates=['Feb_28'],extra_folder='/mu_.3',mu=0.3,locat = '/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/',poff_range=np.r_[.1:1:.1],pdeath_range=np.r_[.1:1:.1])

# various pmut
pmut004 = Experiment(dates=['Jun_21_23'],extra_folder='pmut-poff-pdeath/',mu=0.1,pmut=.004,poff_range=np.r_[.1:1.1:.1],pdeath_range=np.r_[.1:1.1:.1])
pmut01 = Experiment(dates=['Jun_21_23'],extra_folder='pmut-poff-pdeath/',mu=0.1,pmut=.01,poff_range=np.r_[.1:1.1:.1],pdeath_range=np.r_[.1:1.1:.1])
pmut019 = Experiment(dates=['Jun_21_23'],extra_folder='pmut-poff-pdeath/',mu=0.1,pmut=.019,poff_range=np.r_[.1:1.1:.1],pdeath_range=np.r_[.1:1.1:.1])

# for which of these experiments do you want plots
#plots_to_make = [mu002,mu005,mu01,mu02,mu03] #[mu005]
#plots_to_make = [mu002,mu01,mu03] #[mu005]
plots_to_make = [pmut004,pmut01,pmut019] #[mu005]
param_to_vary = 'pmut'


# %% ---------------------
# assemble filename of input
def find_file(date,seed,T=False,poff=False,pdeath=False,type="pypy",C='',mu='',theta='',pmut='',L=20):
    # return locat+f"pypy_seed{str(seed)}{date}_{str(T)}K_TNM_L20.dat"
    #print("* * * f i n d i n g   f i l e * * * ")
    #print("looking for ",type," file")
    date2 = date
    if type == "pypy": 
        #print("entered pypy for loop")
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
if len(plots_to_make) > 1:
    cols = []
    for _ in plots_to_make: cols.append(1)
    cols.append(.1)
    gridspec = {'width_ratios':cols,'height_ratios':[1,1,1]}
    multi_fig,multi_ax = plt.subplots(3,len(plots_to_make)+1,sharex=False,sharey=False,figsize=(doub_width,doub_width),gridspec_kw=gridspec)
    ratio_fig,ratio_ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(doub_width,.4*doub_width))
    for ra in ratio_ax:
        ra.set_xlabel(r"p$_\mathrm{off,T}$/p$_\mathrm{death}$")
        ra.plot([1,1],[0,1],"r--")
        ra.set_xlim(0.5,5)
    ratio_ax[0].set_ylabel("Survival fraction")
    ratio_ax[0].set_title(r"p$_\mathrm{mut}$=0.004")
    ratio_ax[1].set_title(r"p$_\mathrm{mut}$=0.01")
    ratio_ax[2].set_title(r"p$_\mathrm{mut}$=0.019")

    print(f"Making {len(plots_to_make)} columns for different experiments") 

    for col in range(len(plots_to_make)):
        if param_to_vary == "mu":
            mu = plots_to_make[col].mu
            multi_ax[0,col].set_title(r"1/$\mu$ = "+f"{1/mu:.2f}")
        elif param_to_vary == "pmut":
            pmut = plots_to_make[col].pmut
            multi_ax[0,col].set_title(r"p$_\mathrm{mut}$ = "+f"{pmut:.3f}")


big_col = -1
for this_plot in plots_to_make:
    big_col += 1
    print("Column: ",big_col)
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
    ratios_this_pmut = []

    # determin ratios of poff/pdeath vs survival
    ratios = []
    for poff in poff_range:
        for pdeath in pdeath_range:
            ratios.append(poff/pdeath)

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
        # ratios now
        print("indep_now[i]: ",indep_now[i])
        print("survival_by_T[-1]: ",survival_by_T[-1])
        if len(indep_now[i]) > 1:
            ratio = indep_now[i][0]/indep_now[i][1]
            ratios_this_pmut.append([ratio,survival_by_T[-1]])
            # plot ratio of poff/pdeath vs survival 
            ratio_ax[big_col].scatter(ratio,survival_by_T[-1],color="k",s=5)

            letters = iter(["a)","b)","c)","d)","e)","f)","g)","h)","i)","j)","k)","l)"])
            for a in ratio_ax:
                letter = next(letters)
                a.text(0.05,0.9,letter,transform=a.transAxes,horizontalalignment="left",verticalalignment="center")

            if indep_now[i] == indep_now[-1]: 
                ratio_fig.tight_layout()
                ratio_fig.savefig("figures/ratios_po_pd_surv.pdf")
                np.save(f"ratio_surv_pmut{pmut}.npy",np.array(ratios_this_pmut))

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

    if plot_scatters:
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
        if len(plots_to_make) > 1:
            sax = multi_ax[0,big_col]
            aax = multi_ax[1,big_col]
            dax = multi_ax[2,big_col]
            #cbar_ax1 = multi_ax[0,len(plots_to_make)]
            #cbar_ax2 = multi_ax[1,len(plots_to_make)]
            #cbar_ax3 = multi_ax[2,len(plots_to_make)]
            print("Big_col: ",big_col," used to make axes")
        else:
            sfig,sax = plt.subplots() # survival
            afig,aax = plt.subplots() # abundance
            dfig,dax = plt.subplots() # diversity

        # make plots
        cmap = plt.get_cmap('YlGnBu') #Greys')

        # make mask to show where there is no data
        no_surv = np.isnan(grid_surv)
        mask_layer = np.ma.masked_where(no_surv == 0,no_surv)

        survival_image = sax.imshow(grid_surv,cmap=cmap,extent=[0,len(poff_range),0,len(pdeath_range)],interpolation='none')
        if big_col == len(plots_to_make)-1:
            divider = make_axes_locatable(sax)
            #c_sax = divider.append_axes("right", size="10%", pad=0.1)
            c_sax = multi_ax[0,-1]
        if plot_type == "mean":
            abundance_image = aax.imshow(grid_abund, cmap=cmap, extent=[0,len(poff_range),0,len(pdeath_range)],interpolation='none',vmin=300,vmax=1400)
            diversity_image = dax.imshow(grid_div, cmap=cmap, extent=[0,len(poff_range),0,len(pdeath_range)],interpolation='none',vmin=30,vmax=90)
        else:
            abundance_image = aax.imshow(grid_N_med, cmap=cmap, extent=[0,len(poff_range),0,len(pdeath_range)],interpolation='none',vmin=300,vmax=1400)
            if big_col == len(plots_to_make)-1:
                divider = make_axes_locatable(aax)
                #c_aax = divider.append_axes("right", size="10%", pad=0.1)
                c_aax = multi_ax[1,-1]
            diversity_image = dax.imshow(grid_D_med, cmap=cmap, extent=[0,len(poff_range),0,len(pdeath_range)],interpolation='none',vmin=30,vmax=90)
            if big_col == len(plots_to_make)-1:
                divider = make_axes_locatable(dax)
                #c_dax = divider.append_axes("right", size="10%", pad=0.1)
                c_dax = multi_ax[2,-1]

        # plot red in boxes where there was no survival
        for a in [sax,aax,dax]:
            #a.imshow(mask_layer,extent=[0,len(poff_range),0,len(pdeath_range)],alpha=.5,cmap='Reds',hatch='//',vmin=0,vmax=2)
            X,Y = np.meshgrid(np.arange(len(poff_range)+1),np.arange(len(pdeath_range)+1))
            a.pcolor(X,Y[::-1],mask_layer,hatch=r'\\',alpha=0) #,vmin=0,vmax=2)

        # make list of labels, skipping every other number to prevent crowding
        x_labels,y_labels = [],[]
        i = 1
        for poff in poff_range:
            if i == 1:
                x_labels.append(f"{poff:.1f}")
                i = 0
            else: 
                x_labels.append('')
                i = 1
        for pdeath in pdeath_range:
            y_labels.append(f"{pdeath:.2f}")

        # put tick labels at desired locations
        for aa in [sax,aax,dax]:
            aa.grid(which='major',axis='both',linestyle='-',color='k',linewidth=0)
            aa.set_xticks(np.r_[.5:len(poff_range):1])
            aa.set_yticks(np.r_[.5:len(pdeath_range):1])
            aa.set_xticklabels(x_labels) #,rotation=90)
            aa.set_yticklabels(y_labels)
        if big_col == 0:
            aax.set_ylabel(r"Death probability, p$_\mathrm{death}$")
                #aa.set_ylabel(r"$p_{death}$")
        if big_col == 1:
            dax.set_xlabel(r"Temperature dependent component of reproduction probabilitiy, $p_{off,T}$")

        if len(plots_to_make) > 1:
            for aa in multi_ax[:,1:].flatten():
                plt.setp(aa.get_yticklabels(), visible=False)
            for aa in multi_ax[:-1,:].flatten():
                plt.setp(aa.get_xticklabels(), visible=False)
            if big_col == len(plots_to_make)-1:
                print("Making colorbars, col = ",big_col)
                cbar1 = multi_fig.colorbar(survival_image, label="Survival probability", cax=c_sax) #multi_ax[0,-1],shrink=.8 ) #cbar_ax1, extend='both')
                #cbar.minorticks_on()
                cbar2 = multi_fig.colorbar(abundance_image, label=f"M{plot_type[1:]} final abundance", cax=c_aax) #multi_ax[1,-1],shrink=.8) #cbar_ax2, extend='both')
                #cbar.minorticks_on()
                cbar3 = multi_fig.colorbar(diversity_image, label=f"M{plot_type[1:]} final diversity", cax=c_dax) #multi_ax[2,-1],shrink=.8) #cbar_ax3, extend='both')
                #cbar.minorticks_on()
        else:
            cbar = sfig.colorbar(survival_image, label="Survival probability", ax=sax, extend='both')
            cbar.minorticks_on()
            cbar = afig.colorbar(abundance_image, label=f"M{plot_type[1:]} final abundance", ax=aax, extend='both')
            cbar.minorticks_on()
            cbar = dfig.colorbar(diversity_image, label=f"M{plot_type[1:]} final diversity", ax=dax, extend='both')
            cbar.minorticks_on()
        
        # scale plot 
        scalex = len(poff_range)/max(poff_range)
        scaley = len(pdeath_range)/max(pdeath_range)
        #sax.plot(scalex*poff_range,scaley*poff_range**2,"--r",label="p(surv. & reprod.)")
        sax.set_ylim(0,scaley*max(pdeath_range))
        
        if show_TRC:
            # lay TRC ontop
            factor = 1
            all_temps = np.r_[274:320]
            experiment_temps = np.r_[274:320:3]
            poff_T = factor*Teq.poff_T(all_temps)
            pdeath_T = Teq.pdeath(all_temps)
            sax.plot(len(poff_range)*poff_T,len(pdeath_range)*pdeath_T,"r",label="TRC")
            for Ti in experiment_temps:
                x = len(poff_range)*Teq.poff_T(Ti)
                y = len(pdeath_range)*Teq.pdeath(Ti)
                sax.plot(x,y,'ro')
                sax.text(x+.02,y+.02,Ti,rotation=45,color='r')
            #poff_T = factor*Teq.poff_T(experiment_temps)
            #pdeath_T = Teq.pdeath(experiment_temps)
            #sax.plot(len(poff_range)*poff_T,len(pdeath_range)*pdeath_T,"ro")

            sax.legend()
            
        if len(plots_to_make) == 1:
            sax.set_title("Survival probability")
            aax.set_title(f"M{plot_type[1:]} abundance after {maxgens} gen.")
            dax.set_title(f"M{plot_type[1:]} diversity after {maxgens} gen.")

            if locat.startswith('/net/'):
                if len(plots_to_make) > 1:
                    pass
                else:
                    sfig.savefig(locat+experiment+extra_folder+date+"/survival_grid_"+experiment[:-1]+date+extra_folder[:-1]+".pdf")
                    afig.savefig(locat+experiment+extra_folder+date+"/abundance_grid_"+experiment[:-1]+date+extra_folder[:-1]+".pdf")
                    #plt.imsave(locat+experiment+date+extra_folder+"/abundance_imsave_"+experiment[:-1]+date+extra_folder[1:]+".pdf")
                    dfig.savefig(locat+experiment+extra_folder+date+"/diversity_grid_"+experiment[:-1]+date+extra_folder[:-1]+".pdf")
            else:
                sfig.savefig(locat+experiment+date+extra_folder+"/survival_grid_"+experiment[:-1]+date+extra_folder[1:]+".pdf")
                afig.savefig(locat+experiment+date+extra_folder+"/abundance_grid_"+experiment[:-1]+date+extra_folder[1:]+".pdf")
                #plt.imsave(locat+experiment+date+extra_folder+"/abundance_imsave_"+experiment[:-1]+date+extra_folder[1:]+".pdf")
                dfig.savefig(locat+experiment+date+extra_folder+"/diversity_grid_"+experiment[:-1]+date+extra_folder[1:]+".pdf")
            #plt.imsave(locat+experiment+date+extra_folder+"/diversity_imsave_"+experiment[:-1]+date+extra_folder[1:]+".pdf")
                plt.show()

#plt.tight_layout()
if len(plots_to_make) > 1:
    try:
        multi_fig.savefig(fig_locat+experiment+"multi_grid_"+experiment[:-1]+"vary"+param_to_vary+".png")
    except:
        print("Can't save file: ")
        print(fig_locat+experiment+"multi_grid_"+experiment[:-1]+dategg+".png")
#plt.subplots_adjust(right=.9) #wspace=0,hspace=0.1,right=.7)

multi_fig.savefig("figures/multi_grid_poff_pdeath_pmut.png")

plt.show()
