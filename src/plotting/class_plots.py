import os
import numpy as np
import matplotlib.pyplot as plt
import sys
# import Teq and classes from output location

multi_inputs = True
indiv_seeds = 0 #6
maxgens = 10_000 #50000
plot_expectation = False #True
plot_po_pd = False
expectation_location = './'
pmut_vals = [0.,0.004,0.007,0.01,0.02,0.03]
div_at_pmut = [2.0, 31.25, 50.52, 64.825, 101.87, 138.95]
all_temps = np.r_[274:320]
prediction_Ts = np.r_[274:320:3]
predicted_survival_at_T = np.load("npy_files/predicted_survival_prob_testFeb_09D_init_60.npy") # [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.97766991, 0.94626122, 0.97005099, 0.98249544, 0.99126496, 0.99047478, 0.99219141, 1., 1., 1., 1, 1., 0.99541175, 0.89838632, 0.59938348, 0.46386786, 0.10610297, 0., 0., 0., np.nan, np.nan]

# figure sizes
sin_width = 3.54 # single column width (inches)
onehalf_width = 5.51 # 1.5x columnu
doub_width = 7.48 # full width


class Experiment:
    def __init__(self,date="",seeds=np.r_[1000:1050],pdeath_range=False,poff_range=False,T_range=np.r_[274:320:3],dates=[],experiment="SteadyT/",extra_folder="",L=20,C=100,theta=.25,mu=.1,pmut=0.01,locat='/net/venus/kenes/user/cfebvre/out_venus/TTNM_outputs/',write_freq=4000,poffeco_inState=True):
        self.date = date
        self.seeds = seeds
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
        self.write_freq = write_freq
        self.locat = locat
        if locat.startswith("/net/"):
            self.outpath = locat + f'{experiment}{extra_folder}/{date}/'
        else:
            if type(extra_folder) == str:
                self.outpath = locat+f'{experiment}{date}/{extra_folder}/'
                if not os.path.exists(self.outpath):
                    print("Error: output path doesn't exist. ",self.outpath)
            else:
                self.outpath = locat+f'{experiment}/{date}/{extra_folder[0]}/'
        self.fig_locat = self.outpath + "Figures/"
        if not os.path.exists(self.fig_locat):
            os.mkdir(self.fig_locat)
        self.poffeco_inState = poffeco_inState

# %% DEFINE EXPERIMENTS:
## if only one date, use date = <str>, otherwise use date = [<str>,<str>,...]
## if only one extra_folder, use extra_folder = <str>
#    # otherwise, use [<str>,<str>,...] and len(extr_folder) must equal len(dates)

# vary poff and pdeath only with T
#singleTRC = Experiment(seeds=np.r_[2200:2250],date="Feb_24",extra_folder="one-TRC-scaledup",locat='/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/')
singleTRC = Experiment(date="Sep_22_23",extra_folder="single-TRC/constpmut-Tref303")
oneTRC_constpmut = Experiment(date="Feb_04_24",extra_folder="single-TRC/constpmut-Tref303")
varTRC_constpmut = Experiment(date="Feb_05_24",extra_folder="var-TRC/constpmut")
#MTEenv = Experiment(seeds=np.r_[100:700],date="Sep_29",dates=["Sep_29","Oct_11"],extra_folder=["MTE_TPC_combo","MTE-env"],locat='/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/')
MTEenv = Experiment(seeds=np.r_[100:200],date="Sep_29",extra_folder="MTE_TPC_combo",locat='/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/')


# vary pmut with T
#oneTRC_varypmut = Experiment(date="Mar_30_23",extra_folder="single-TRC/varypmut")
oneTRC_varypmut = Experiment(date="Jun_21_23",extra_folder="single-TRC/varypmut-Tref303") #,locat='/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/')
varTRC_varypmut = Experiment(date="Apr_13_23",extra_folder="var-TRC/varypmut")
#MTEenv_varypmut = Experiment(date="Apr_20_23",extra_folder="MTE-env/varypmut")
MTEenv_varypmut = Experiment(date="Apr_20_23",dates=["Apr_20_23","May_01_23","May_02_23"],extra_folder="MTE-env/varypmut",seeds=np.r_[1000:1300])
MTEenv_varypmut_moreTopt = Experiment(date="May_03_23",dates=["May_03_23"],extra_folder="MTE-env/varypmut",seeds=np.r_[1100:1300])
#MTEenv2_varypmut = Experiment(date="Apr_25_23",extra_folder="MTE-env/varypmut-TRCscale")

# larger sample sizes
oneTRC_varypmut_more = Experiment(date="Dec_14_23",extra_folder="single-TRC/varypmut-Tref303",seeds=np.r_[1050:1250]) #,locat='/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/')
varTRC_varypmut_more = Experiment(date="Dec_14_23",extra_folder="var-TRC/varypmut",seeds=np.r_[1050:1250])


# keep the integral of MTE-env TRCs constant
MTEenv2_varypmut = Experiment(date="Apr_25_23",dates=["Apr_25_23","Apr_27_23"],extra_folder="MTE-env/varypmut-TRCscale",seeds=np.r_[1000:1100])

# test TRC widths in single TRC (pmut is still varying with T in these experiments)
oneTRC_width13 = Experiment(date="May_03_23",extra_folder="single-TRC/varypmut-width13")
oneTRC_width15 = Experiment(date="May_03_23",extra_folder="single-TRC/varypmut-width15")
varTRC_width13 = Experiment(date="May_05_23",extra_folder="var-TRC/varypmut-width13")
varTRC_width15 = Experiment(date="May_05_23",extra_folder="var-TRC/varypmut-width15")

# scale up MTE-env
MTEenv_scaledup = Experiment(date="May_09_23",dates=["May_09_23","May_19_23"],seeds=np.r_[1000:1300],extra_folder="MTE-env/scaledup")
MTEenv_scaledup_symmetrical = Experiment(date="May_09_23",extra_folder="MTE-env/scaledup-skew0")

# weird tests
MTEenv_scaledup_backwards = Experiment(date="May_29_23",extra_folder="MTE-env/scaledup-skew3")
varTRC_constdeath= Experiment(date="May_30_23",dates=["May_30_23","Jun_01_23"],seeds=np.r_[1000:1300],extra_folder="var-TRC/const-pdeath")
varTRC_constdeathmut = Experiment(date="Jun_01_23",seeds=np.r_[1000:1300],extra_folder="var-TRC/const-pdeath-pmut")


experiments = [varTRC_constpmut] #constnat pmut
#experiments = [oneTRC_varypmut] #_scaledup_backwards] #_moreTopt]
#experiments = [varTRC_varypmut] #constdeathmut] #_scaledup_backwards] #_moreTopt]
#experiments = [MTEenv_varypmut] #_scaledup_backwards] #_moreTopt]

# FOR LATER USE???
"""
seeds = []
dates = []
extra_folders = []
locats = []
outpaths = []
"""
for this_exp in experiments:
    seeds = this_exp.seeds
    pdeath_range = this_exp.pdeath_range
    poff_range = this_exp.poff_range
    T_range = this_exp.T_range
    dates = this_exp.dates
    date = this_exp.date
    experiment = this_exp.experiment
    extra_folder = this_exp.extra_folder
    L,C,theta,mu,pmut = this_exp.L, this_exp.C, this_exp.theta, this_exp.mu, this_exp.pmut
    locat = this_exp.locat
    write_freq = this_exp.write_freq
    outpath = this_exp.outpath

    if type(extra_folder) == str:
        if "/" in extra_folder:
            idx = extra_folder.index("/")
            exp = extra_folder[:idx]
        else: exp = extra_folder
    else:
        if "/" in extra_folder[0]:
            idx = extra_folder[0].index("/")
            exp = extra_folder[0][:idx]
        else: exp = extra_folder[0]

    Tuniv = 307.8
    
    if len(dates) > 1:
        outpath_list = []
        i = -1
        for d in dates:
            i += 1
            if locat.startswith("/net/"):
                outpath_list.append(locat + f'{experiment}{extra_folder}/{d}/')
            else: 
                if type(extra_folder) == str:
                    outpath_list.append(locat+f'{experiment}/{d}/{extra_folder}/')
                else:
                    outpath_list.append(locat+f'{experiment}/{d}/{extra_folder[i]}/')
    else: outpath_list = [outpath]
    poffeco_inState = this_exp.poffeco_inState
    fig_locat = this_exp.fig_locat

for op in outpath_list:
    print("outpath: ",op)
    print("exists? ",os.path.exists(op))

# determine gens in last chunk
if write_freq:
    lastgens = int(maxgens - write_freq*int(maxgens/write_freq))

#outpath = '/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/NewTNM/Feb_10/testTNM/'
#outpath = '/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/NewTNM/Feb_11/test_Species_class/'
# import modules from output path
sys.path.append(outpath)
import MTE_TPC_combo as Teq
from test_classes import Species, State

plot_examples = False# True

def format_outputs(seed,T,verbose=False):
    if T:
        if T < 1:
            T = f"{T:.2f}"
        else: T = f"{int(T)}"
    else: T = "no"

    # Final chunks or only chunks
    found = 0
    for op in outpath_list:
        # if filename doesn't end with 4000gen or 8000gen, its the final chunk
        if os.path.exists(op+f"modelrun_seed{seed}_{T}K.npy"):
            modelrun = np.load(op+f"modelrun_seed{seed}_{T}K.npy",allow_pickle=True)
            found = 1
            break
    if not found: 
        print(f"+++Seed {seed} not found+++")
        return [False]

    # Determine if there are previous generations of the same T and seed in different files
    count = 0
    list1 = []
    for f in os.listdir(op):
        if f.startswith(f"modelrun_seed{seed}_{T}K"):
            count += 1
            list1.append(f)
    if count > 1:
        order = []
        filelist = [] # all files associated with this seed and T, in order
        filelist_states = [] # files sorted by gen chunk
        for f in list1: # os.listdir(outpath):
            if "gen" in f:
                #print(f)
                idx1 = f.index('seed') + len(f"seed{seed}_{T}K_")
                idx2 = f.index('gen')
                finalgen = int(f[idx1:idx2])
                order.append(finalgen)
                filelist.append(f)
        order_sorted = np.sort(order)
        indices = []
        for o in order_sorted:
            indices.append(order.index(o))
        for i in indices:
            filelist_states.append(filelist[i])

    if count > 1:
        # make new State object to accumulate all time chunks
        big_state = State(seed)
        big_state.N_timeseries = np.zeros((maxgens,))
        big_state.D_timeseries = np.zeros((maxgens,))
        big_state.coreN_timeseries = np.zeros((maxgens,))
        big_state.coreD_timeseries = np.zeros((maxgens,))
        big_state.inSS_timeseries = np.zeros((maxgens,))
        big_state.poffeco_t = np.zeros((maxgens,))
        
        # find number gens run in last chunk
        gensrun = len(modelrun[0].N_timeseries)
        #print("Gens run in last chunk: ",gensrun)
        # final chunk of big_state timeseries is from the file wihtout "gens" in its name
        
        # how many time chunks did this experiment survive?
        chunks_survived = len(filelist) 
        if chunks_survived > 1:
            chunk_length = order_sorted[0]
        else: chunk_length = 0

        # add each time chunk to the overall state timeseries
        for f in filelist_states: #elif f.startswith(f"modelrun_seed{seed}"):
            if 'gen' in f: # already added output from this file
                m_list = np.load(op+f,allow_pickle=True)
                idx1 = f.index('seed') + len(f"seed{seed}_{T}K_")
                idx2 = f.index('gen')
                starting_gen = list(big_state.N_timeseries).index(0)
                finalgen = int(f[idx1:idx2])
                if verbose:
                    print(seed,"starting_gen: ",starting_gen,"finalgen: ",finalgen)
                for m in m_list:
                    big_state.N_timeseries[starting_gen:finalgen] = m.N_timeseries
                    big_state.D_timeseries[starting_gen:finalgen] = m.D_timeseries
                    big_state.coreN_timeseries[starting_gen:finalgen] = m.coreN_timeseries
                    big_state.coreD_timeseries[starting_gen:finalgen] = m.coreD_timeseries
                    big_state.inSS_timeseries[starting_gen:finalgen] = m.inSS_timeseries
                    big_state.poffeco_t[starting_gen:finalgen] = m.poffeco_t

        # final chunk
        starting_gen = list(big_state.N_timeseries).index(0) # chunks_survived*chunk_length 
        finishing_gen = starting_gen + len(modelrun[0].N_timeseries) #gensrun 
        # double check we're starting at right place
        
        if verbose:
            print(seed,"starting_gen: ",starting_gen,"finishing-gen: ",finishing_gen)
        #if idx != starting_gen:
        #    print(f"starting_gen: {starting_gen}, but first zero: {idx}")
        #if starting_gen == 0:
        #    finishing_gen -= 1
        #print("starting & finishing gens of the final chunk: ",starting_gen,finishing_gen)
        if gensrun > 0:
            big_state.N_timeseries[starting_gen:finishing_gen] = modelrun[0].N_timeseries
            big_state.D_timeseries[starting_gen:finishing_gen] = modelrun[0].D_timeseries
            big_state.coreN_timeseries[starting_gen:finishing_gen] = modelrun[0].coreN_timeseries
            big_state.coreD_timeseries[starting_gen:finishing_gen] = modelrun[0].coreD_timeseries
            big_state.inSS_timeseries[starting_gen:finishing_gen] = modelrun[0].inSS_timeseries
            if len(modelrun[0].poffeco_t) == gensrun:
                big_state.poffeco_t[starting_gen:finishing_gen] = modelrun[0].poffeco_t
            else: 
                gens_poff = len(modelrun[0].poffeco_t)
                #print("gens_poff: ",gens_poff)
                if gens_poff > 0:
                    big_state.poffeco_t[starting_gen:starting_gen+len(modelrun[0].poffeco_t)] = modelrun[0].poffeco_t
                #print("gens run: ",gensrun," but len(poffeco): ",len(modelrun[0].poffeco_t))

        if verbose:
            if chunks_survived*write_freq + len(modelrun[0].N_timeseries) != len(big_state.N_timeseries):
                print("ERROR")
                print("len(big_state.N_timeseries): ",len(big_state.N_timeseries))
                print("expected: ",chunks_survived*write_freq + len(modelrun[0].N_timeseries))
            if 0 in list(big_state.N_timeseries):
                print("First zero at: ",list(big_state.N_timeseries).index(0))
                print("but it should have been at ",chunks_survived*write_freq + len(modelrun[0].N_timeseries))


    if count > 1: 
        modelrun = [big_state]
    return modelrun
                
# For each seed and T, collect outputs
if plot_examples:
    fig,ax = plt.subplots(4,4)
    row,col = 0,-1
files_found = 0
if multi_inputs:
    N_by_T = []
    Ni_by_T = []
    D_by_T = []
    coreN_by_T = []
    coreNi_by_T = []
    coreD_by_T = []
    poffeco_by_T = []
    Jij_by_T = []
    fourier_by_T = []
    survival_by_T = []
    quakes_by_T = []
    SS_dur_by_T = []
    time_of_extinction = []
    quakes_by_extinction = []
    extinctions_per_quake = []
    survivals_per_quake = []
    quake_number_by_T = []
    quake_times_by_T = []
    i = -1
    for T in T_range:
        if plot_examples:
            col += 1
            if col == 4:
                col = 0
                row += 1
            ax[row,col].set_title(T)
        if indiv_seeds > 0:
            if T in [280,292,301,310]:
                indiv_fig,indiv_ax = plt.subplots(5,indiv_seeds,sharex=True,sharey='row',figsize=(8,6))
                for k in range(indiv_seeds):
                    indiv_ax[4,k].set_xlabel("Time (gen)")
                indiv_ax[0,0].set_ylabel("Abundance")
                indiv_ax[1,0].set_ylabel("Core abund.")
                indiv_ax[2,0].set_ylabel("Diversity")
                indiv_ax[3,0].set_ylabel("Core div.")
                indiv_ax[4,0].set_ylabel("Eco. poff")
                indiv_ax[0,0].set_title(f"T: {T}K")

        N_by_T.append([]) #np.zeros(maxgens))
        Ni_by_T.append([]) #np.zeros(maxgens))
        coreN_by_T.append([]) #np.zeros(maxgens))
        coreNi_by_T.append([]) #np.zeros(maxgens))
        D_by_T.append([]) #np.zeros(maxgens))
        coreD_by_T.append([]) #np.zeros(maxgens))
        poffeco_by_T.append([])
        Jij_by_T.append([])
        fourier_by_T.append(np.zeros(maxgens))
        survival_by_T.append(np.zeros(maxgens))
        quakes_by_T.append([])
        SS_dur_by_T.append([])
        time_of_extinction.append([])
        quakes_by_extinction.append([])
        extinctions_per_quake.append(np.zeros((3000,)))
        survivals_per_quake.append(np.zeros((3000,)))
        quake_number_by_T.append([])
        quake_times_by_T.append([])
        i += 1
        j = -1
        for seed in seeds:
            j += 1
            modelrun = format_outputs(seed,T)
            if plot_examples:
                if 0 in modelrun[0].N_timeseries:
                    ax[row,col].plot(modelrun[0].N_timeseries,"k",linewidth=2)
                    ax2 = plt.twinx(ax[row,col])
                    #ax2.plot(modelrun[0].inSS_timeseries,"r--",linewidth=2)
                    ax2.scatter([list(modelrun[0].N_timeseries).index(0)],[0],color="b",marker="x")
            if type(modelrun[0]) != bool:
                files_found += 1
                # perform Fourier transform
                srs = np.zeros(maxgens)
                m = modelrun[0]
                srs = m.N_timeseries
                #N_by_T[i][:len(srs)] += srs
                N_by_T[i].append(srs)
                D_by_T[i].append(m.D_timeseries)
                Ni_by_T[i].append(list(np.array(srs)/np.array(m.D_timeseries)))
                coreN_by_T[i].append(m.coreN_timeseries)
                coreD_by_T[i].append(m.coreD_timeseries)
                coreNi_by_T[i].append(list(np.array(m.coreN_timeseries)/np.array(m.coreD_timeseries)))
                poffeco_by_T[i].append(m.poffeco_t)
                fav = - np.log(1/np.array(m.poffeco_t) -1)
                mu = 0.1
                Jij_av = list(fav + mu*np.array(m.N_timeseries[:len(fav)]))
                Jij_by_T[i].append(Jij_av)
                fft = np.fft.fft(srs)
                reverse = np.fft.ifft(fft[:100])
                fourier_by_T[i][:len(fft)] += np.array(fft,dtype=float)
                survival_by_T[i][:len(srs)] += np.ones(len(srs))
                # count quakes in this run starting after 10 generations
                quakes_this_run = 0
                SS_durations_this_run = []
                quaketimes_this_run = []
                quakenums_this_run = []
                SS_start = 0
                for t in range(10,len(m.inSS_timeseries)):
                    if m.inSS_timeseries[t-1] == 0 and m.inSS_timeseries[t] == 1:
                        SS_start = t
                    if m.inSS_timeseries[t-1] == 1 and m.inSS_timeseries[t] == 0:
                        quakes_this_run += 1
                        if SS_start > 0:
                            SS_durations_this_run.append(t-SS_start)
                        #else: SS_duration_this_run.append(t)
                        quaketimes_this_run.append(t)
                        quakenums_this_run.append(quakes_this_run)
                quake_number_by_T[i].append(quakenums_this_run)
                quake_times_by_T[i].append(quaketimes_this_run)
                quakes_by_T[i].append(quakes_this_run)
                SS_dur_by_T[i].append(SS_durations_this_run)
                # if this experiment didn't survive til end, figure out time of death and number of quakes at that time
                time_of_death = 0
                if 0 in srs[:-2]:
                    # extinction occurred
                    if type(srs) == list:
                        time_of_death = srs.index(0)
                    else: time_of_death = list(srs).index(0)
                    time_of_extinction[-1].append(time_of_death)
                    quakes_by_extinction[-1].append(quakes_this_run)
                elif len(srs) < maxgens: 
                    time_of_death = len(srs)
                    time_of_extinction[-1].append(time_of_death)
                    quakes_by_extinction[-1].append(quakes_this_run)
                if time_of_death > 0: # an extinction occurred in this experiment
                    if quakes_this_run > 0:
                        survivals_per_quake[i][:quakes_this_run-1] += np.ones((quakes_this_run-1,))
                        extinctions_per_quake[i][quakes_this_run] += 1
                    else: extinctions_per_quake[i][quakes_this_run] += 1
                
                # plot individual runs
                if T in [280,292,301,310] and j < indiv_seeds:
                    indiv_ax[0,j].plot(m.N_timeseries)
                    indiv_ax[1,j].plot(m.coreN_timeseries)
                    indiv_ax[2,j].plot(m.D_timeseries)
                    indiv_ax[3,j].plot(m.coreD_timeseries)
                    indiv_ax[4,j].plot(m.poffeco_t)

                # add nans to makeup empty spots
                add_nans = maxgens - len(srs)
                if add_nans > 0:
                    N_by_T[i][-1].append(np.nan)
                    Ni_by_T[i][-1].append(np.nan)
                    coreN_by_T[i][-1].append(np.nan)
                    coreNi_by_T[i][-1].append(np.nan)
                    D_by_T[i][-1].append(np.nan)
                    coreD_by_T[i][-1].append(np.nan)
                    poffeco_by_T[i][-1].append(np.nan)
                    Jij_by_T[i][-1].append(np.nan)
                    add_nans -= 1
                    if add_nans > 0:
                        N_by_T[i][-1][-1:-1] = np.ones(add_nans)*np.nan
                        Ni_by_T[i][-1][-1:-1] = np.ones(add_nans)*np.nan
                        coreN_by_T[i][-1][-1:-1] = np.ones(add_nans)*np.nan
                        coreNi_by_T[i][-1][-1:-1] = np.ones(add_nans)*np.nan
                        D_by_T[i][-1][-1:-1] = np.ones(add_nans)*np.nan
                        coreD_by_T[i][-1][-1:-1] = np.ones(add_nans)*np.nan
                        poffeco_by_T[i][-1][-1:-1] = np.ones(add_nans)*np.nan
                        Jij_by_T[i][-1][-1:-1] = np.ones(add_nans)*np.nan
                add_nans = maxgens - len(m.poffeco_t)
                if add_nans > 0:
                    poffeco_by_T[i][-1].append(np.nan)
                    Jij_by_T[i][-1].append(np.nan)
                    add_nans -= 1
                    if add_nans > 0:
                        poffeco_by_T[i][-1][-1:-1] = np.ones(add_nans)*np.nan
                        Jij_by_T[i][-1][-1:-1] = np.ones(add_nans)*np.nan

    #plt.figure()
    #for i in range(len(T_range)): 
    #    plt.plot(N_by_T[i][:])
    print("Files found: ", files_found)

    # allocate space for timeseries of medians and quartiles for eac T
    medN_by_T,medNi_by_T, medD_by_T, medcoreN_by_T,medcoreNi_by_T, medcoreD_by_T, medpoffeco_by_T,medJij_by_T = [],[],[],[],[],[],[],[]
    q1N_by_T,q1Ni_by_T, q1D_by_T, q1coreN_by_T,q1coreNi_by_T, q1coreD_by_T, q1poffeco_by_T,q1Jij_by_T = [],[],[],[],[],[],[],[]
    q3N_by_T,q3Ni_by_T, q3D_by_T, q3coreN_by_T, q3coreNi_by_T, q3coreD_by_T, q3poffeco_by_T,q3Jij_by_T = [],[],[],[],[],[],[],[]
    
    i = -1
    for T in T_range:
        i += 1
        medN_by_T.append(np.nanquantile(N_by_T[i],.5,axis=0))
        medNi_by_T.append(np.nanquantile(Ni_by_T[i],.5,axis=0))
        medD_by_T.append(np.nanquantile(D_by_T[i],.5,axis=0))
        medcoreN_by_T.append(np.nanquantile(coreN_by_T[i],.5,axis=0))
        medcoreNi_by_T.append(np.nanquantile(coreNi_by_T[i],.5,axis=0))
        medcoreD_by_T.append(np.nanquantile(coreD_by_T[i],.5,axis=0))
        print(f"shape of Jij_by_T at T={T}: {np.shape(Jij_by_T[i])}")
        print(f"shape of poffeco_by_T at T={T}: {np.shape(poffeco_by_T[i])}")
        if len(np.shape(poffeco_by_T[i])) < 2:
            print("shape of medN_by_T[-1]: ",np.shape(medN_by_T[-1]))
            nanlist = np.nan*np.ones(len(N_by_T[i][1]))
            print("shape of nanlist: ",np.shape(nanlist))
            medpoffeco_by_T.append(nanlist)
            q1poffeco_by_T.append(nanlist)
            q3poffeco_by_T.append(nanlist)

            medJij_by_T.append(nanlist)
            q1Jij_by_T.append(nanlist)
            q3Jij_by_T.append(nanlist)
        else:
            medpoffeco_by_T.append(np.nanquantile(poffeco_by_T[i],.5,axis=0))
            q1poffeco_by_T.append(np.nanquantile(poffeco_by_T[i],.25,axis=0))
            q3poffeco_by_T.append(np.nanquantile(poffeco_by_T[i],.75,axis=0))

            medJij_by_T.append(np.nanquantile(Jij_by_T[i],.5,axis=0))
            q1Jij_by_T.append(np.nanquantile(Jij_by_T[i],.25,axis=0))
            q3Jij_by_T.append(np.nanquantile(Jij_by_T[i],.75,axis=0))

        q1N_by_T.append(np.nanquantile(N_by_T[i],.25,axis=0))
        q1Ni_by_T.append(np.nanquantile(Ni_by_T[i],.25,axis=0))
        q1D_by_T.append(np.nanquantile(D_by_T[i],.25,axis=0))
        q1coreN_by_T.append(np.nanquantile(coreN_by_T[i],.25,axis=0))
        q1coreNi_by_T.append(np.nanquantile(coreNi_by_T[i],.25,axis=0))
        q1coreD_by_T.append(np.nanquantile(coreD_by_T[i],.25,axis=0))

        q3N_by_T.append(np.nanquantile(N_by_T[i],.75,axis=0))
        q3Ni_by_T.append(np.nanquantile(Ni_by_T[i],.75,axis=0))
        q3D_by_T.append(np.nanquantile(D_by_T[i],.75,axis=0))
        q3coreN_by_T.append(np.nanquantile(coreN_by_T[i],.75,axis=0))
        q3coreNi_by_T.append(np.nanquantile(coreNi_by_T[i],.75,axis=0))
        q3coreD_by_T.append(np.nanquantile(coreD_by_T[i],.75,axis=0))

    final_N_by_T = np.array(medN_by_T)[:,-1]
    final_Ni_by_T = np.array(medNi_by_T)[:,-1]
    final_D_by_T = np.array(medD_by_T)[:,-1]
    final_coreN_by_T = np.array(medcoreN_by_T)[:,-1]
    final_coreNi_by_T = np.array(medcoreNi_by_T)[:,-1]
    final_coreD_by_T = np.array(medcoreD_by_T)[:,-1]
    print("shape of medJij_by_T: ",np.shape(medJij_by_T))
    print("shape of medpoffeco_by_T: ",np.shape(medpoffeco_by_T))
    final_poffeco_by_T = np.array(medpoffeco_by_T)[:,-1]
    final_Jij_by_T = np.array(medJij_by_T)[:,-1]

    if sum(np.array(final_D_by_T) == 0) > 1: # if any elements of final_D are zero, eliminate
        final_N_by_T = np.array(final_N_by_T)[np.array(final_N_by_T) > 2]
        final_Ni_by_T = np.array(final_Ni_by_T)[np.array(final_Ni_by_T) > 2]
        final_D_by_T = np.array(final_D_by_T)[np.array(final_D_by_T) > 2]
        final_coreN_by_T = np.array(final_coreN_by_T)[np.array(final_D_by_T) > 2]
        final_coreD_by_T = np.array(final_coreD_by_T)[np.array(final_D_by_T) > 2]
        final_poffeco_by_T = np.array(final_poffeco_by_T)[np.array(final_D_by_T) > 2]
    
    # define some colors
    cmap = plt.get_cmap("inferno") #RdPu")
    colors = cmap(np.linspace(.2,1,len(T_range)))
    cmap2 = plt.get_cmap("Greys")
    colors2 = cmap2(np.linspace(.2,1,5))


    # plot N(T) at a few example ts
    fig,ax = plt.subplots(2,2,sharex="col",sharey=False,figsize=(doub_width,.6*doub_width))
    #dfig,dax = plt.subplots(1,2,sharey=False,figsize=(doub_width,.7*doub_width))
    i = -1
    for T in T_range:
        i += 1
        ax[0,0].plot(medN_by_T[i],color=colors[i],label=f"T={T}")
        ax[1,0].plot(medD_by_T[i],color=colors[i],label=f"T={T}")
    k = 8.62e-5 #boltzman constant
    i = -1
    for t in [1,10,100,1000,9900]:
        i += 1
        #ax[1].plot(T_range,np.mean(np.array(medN_by_T)[:,t-1:t+10],axis=1),c=colors2[i],label=f"t={t}")
        ax[0,1].plot(1/k/T_range,np.log(np.mean(np.array(medN_by_T)[:,t-1:t+10],axis=1)),c=colors2[i],label=f"t={t}")
        ax[0,0].plot([t,t],[0,1100],"--",c=colors2[i])
        #dax[1].plot(T_range,np.mean(np.array(medD_by_T)[:,t-1:t+10],axis=1),c=colors2[i],label=f"t={t}")
        ax[1,1].plot(1/k/T_range,np.log(np.mean(np.array(medD_by_T)[:,t-1:t+10],axis=1)),c=colors2[i],label=f"t={t}")
        ax[1,0].plot([t,t],[0,90],"--",c=colors2[i])
    ax[0,0].set_ylabel("Median abundance")
    ax[1,0].set_ylabel("Median species richness")
    ax[0,1].set_ylabel("Log median abundance")
    ax[1,1].set_ylabel("Log median richness")
    #fig.adjust_right(0.8)
    ax[1,0].set_xlabel("Time (t)")
    ax[1,0].set_xscale("log")
    ax[0,0].legend(bbox_to_anchor=(1,1))
    ax[1,1].set_xlabel("1/kT (eV^-1)")
    ax[1,1].legend(loc="lower left")
    #fig.tight_layout()
    plt.subplots_adjust(wspace=0.7)
    fig.savefig(f"figures/N_D_vs_t_T_{exp}_{date}.pdf")
    #dfig.tight_layout()
    #dfig.savefig(f"figures/D_vs_t_T_{exp}_{date}.pdf")

    # plot abundance
    afig,aax = plt.subplots()
    
    aax.fill_between(T_range,np.array(q1N_by_T)[:,-1],np.array(q3N_by_T)[:,-1],alpha=.2,label="interquartile range")
    aax.plot(T_range,np.array(final_N_by_T),label="median")
    aax.set_ylabel("Final abundance")
    aax.set_xlabel("Temperature (K)")
    aax.legend()

    # plot final medians and quartiles for each T
    fig,ax = plt.subplots(2,2,sharex=True,figsize=(doub_width,.75*doub_width))
    ax[0,0].fill_between(T_range,np.array(q1N_by_T)[:,-1],np.array(q3N_by_T)[:,-1],alpha=.2,label="interquartile range")
    ax[0,0].plot(T_range,np.array(medN_by_T)[:,-1],label="median")
    ax[0,0].set_ylabel("Final med. N")

    ax[1,0].fill_between(T_range,np.array(q1coreN_by_T)[:,-1],np.array(q3coreN_by_T)[:,-1],alpha=.2)
    ax[1,0].plot(T_range,np.array(medcoreN_by_T)[:,-1])
    ax[1,0].set_ylabel("Final med. core N")

    ax[0,1].fill_between(T_range,np.array(q1D_by_T)[:,-1],np.array(q3D_by_T)[:,-1],alpha=.2,label="interquartile range")
    ax[0,1].plot(T_range,np.array(medD_by_T)[:,-1],label="median")
    ax[0,1].set_ylabel("Final med. D")

    ax[1,1].fill_between(T_range,np.array(q1coreD_by_T)[:,-1],np.array(q3coreD_by_T)[:,-1],alpha=.2,label="interquartile range")
    ax[1,1].plot(T_range,np.array(medcoreD_by_T)[:,-1],label="median")
    ax[1,1].set_ylabel("Final med. core D")
    #ax[1,1].legend()

    # predictions
    if plot_expectation:
        from scipy import interp
        # interpolate line between points of div vs. pmut
        T_vals = np.r_[274:320:3]
        pmut_at_T = Teq.pmut(T_vals)
        expected_div = interp(pmut_at_T,pmut_vals,div_at_pmut)
        # plot predicted div(T) on pred_ax
        #ax.plot(T_vals,expected_div,"o-")
        #expected_div =  #np.load(expectation_locat)
        ax[0,1].plot(T_range,expected_div,"r--",label="expectation")
            
        expected_N = np.load("npy_files/expected_N.npy")*np.ones(len(T_range))
        ax[0,0].plot(T_range,expected_N,"r--",label="expectation")
        
        # get handles of legend items
        ax[0,0].legend(loc="upper left",fancybox=True, framealpha=0.3)

    for a in [ax[1,0],ax[1,1]]:
        a.set_xlabel("Temperature (K)")

    # label axes A-D
    labels = ["A","C","B","D"]
    for n,a in enumerate(ax.flatten()):
        a.text(-0.1, 1.1, labels[n], transform=a.transAxes,size=12)

    plt.tight_layout()
    plt.savefig(fig_locat+f"final_stats_by_T_{experiment[:-1]}_{date}.pdf")
    plt.savefig(f"figures/final_stats_by_T_{exp}_{date}.pdf")
    
    # save these outputs for later
    four_outputs = {}
    four_outputs['temps'] = T_range
    if plot_expectation:
        four_outputs['refN'] = expected_N
        four_outputs['refD'] = expected_div

    four_outputs['med_N'] = final_N_by_T
    four_outputs['q1_N'] = np.array(q1N_by_T)[:,-1]
    four_outputs['q3_N'] = np.array(q3N_by_T)[:,-1]

    four_outputs['med_Ni'] = final_Ni_by_T
    four_outputs['q1_Ni'] = np.array(q1Ni_by_T)[:,-1]
    four_outputs['q3_Ni'] = np.array(q3Ni_by_T)[:,-1]

    four_outputs['med_D'] = final_D_by_T
    four_outputs['q1_D'] = np.array(q1D_by_T)[:,-1]
    four_outputs['q3_D'] = np.array(q3D_by_T)[:,-1]

    four_outputs['med_coreN'] = np.array(medcoreN_by_T)[:,-1]
    four_outputs['q1_coreN'] =  np.array(q1coreN_by_T)[:,-1]
    four_outputs['q3_coreN'] = np.array(q3coreN_by_T)[:,-1]

    four_outputs['med_coreNi'] = np.array(medcoreNi_by_T)[:,-1]
    four_outputs['q1_coreNi'] =  np.array(q1coreNi_by_T)[:,-1]
    four_outputs['q3_coreNi'] = np.array(q3coreNi_by_T)[:,-1]

    four_outputs['med_coreD'] = np.array(medcoreD_by_T)[:,-1]
    four_outputs['q1_coreD'] = np.array(q1coreD_by_T)[:,-1]
    four_outputs['q3_coreD'] = np.array(q3coreD_by_T)[:,-1]

    np.save(f"npy_files/final_stats_{exp}_{date}.npy",np.array(four_outputs))
    
    # plot diversity on its own
    fig,ax = plt.subplots()
    ax.fill_between(T_range,np.array(q1D_by_T)[:,-1],np.array(q3D_by_T)[:,-1],alpha=.2)
    ax.plot(T_range,np.array(final_D_by_T),"o-",label="median diversity")
    if plot_expectation:
        from scipy import interp
        # interpolate line between points of div vs. pmut
        T_vals = np.r_[274:320:3]
        pmut_at_T = Teq.pmut(T_vals)
        expected_div = interp(pmut_at_T,pmut_vals,div_at_pmut)
        # plot predicted div(T) on pred_ax
        #ax.plot(T_vals,expected_div,"o-")
        #expected_div =  #np.load(expectation_locat)
        ax.plot(T_range,expected_div,label="predicted diversity")
        ax.legend()
    ax.set_ylabel("Final av. D")
    ax.set_xlabel("Temperature (K)")
    plt.savefig(fig_locat+f"diversity_by_T_{experiment[:-1]}_{date}.pdf")

    # plot average poffeco for each T
    fig,ax = plt.subplots(figsize=(doub_width,.7*doub_width))
    ax.plot(T_range,final_poffeco_by_T)
    ax.fill_between(T_range,np.array(q1poffeco_by_T)[:,-1],np.array(q3poffeco_by_T)[:,-1],alpha=.5)
    #i = -1
    #for T in T_range:
    #    i += 1
    #    ax.boxplot(poffeco_by_T[i],positions=[T],showfliers=False)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Average p$_\mathrm{off,i}$ in each experiment")
    
    # plot average interaction
    fig,ax = plt.subplots(figsize=(doub_width,.7*doub_width))
    ax.plot(T_range,final_Jij_by_T)
    ax.fill_between(T_range,np.array(q1Jij_by_T)[:,-1],np.array(q3Jij_by_T)[:,-1],alpha=.5)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Average interaction in each experiment")


    # plot quake counts
    fig,ax = plt.subplots(figsize=(doub_width,.7*doub_width))
    i = -1
    for T in T_range:
        i += 1
        ax.boxplot(quakes_by_T[i],positions=[T],widths=2,showfliers=False)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Number of quakes per experiment")
    plt.savefig(f"figures/quake_boxplots_{exp}_{date}.pdf")

    # plot quake number vs time for each temperature
    np.save(f"npy_files/quaketimes_by_T_{exp}_{date}.npy",quake_times_by_T)
    np.save(f"npy_files/quakenums_by_T_{exp}_{date}.npy",quake_number_by_T)
    plot_quake_counts = True
    if plot_quake_counts:
        fig,ax = plt.subplots(4,4,figsize=(doub_width,.7*doub_width))
        i = -1
        col = -1
        row = 0
        for T in T_range:
            i+= 1
            col += 1
            if col == 4:
                col = 0
                row += 1
            print("row,col: ",row,col)
            for j in range(len(quake_times_by_T[i])):
                ax[row,col].scatter(quake_times_by_T[i][j],quake_number_by_T[i][j],color=colors[i],label=f"{T}K")
        #    ax[row,col].scatter(quake_times_by_T[i],quake_number_by_T[i],color=colors[i],label=f"{T}K")
        for a in ax[-1,:]:
            a.set_xlabel("Time, t (gen)")
        for a in ax[:,0]:
            a.set_ylabel("Quake number")
        #ax.legend(bbox_to_anchor=(1,1))
        fig.tight_layout()
        plt.savefig(f"figures/quaketimes_by_T_{exp}_{date}.pdf")

    # scatter time of extinction vs. quake number
    fig,ax = plt.subplots(4,4,figsize=(doub_width,.7*doub_width))
    all_fig,all_ax = plt.subplots(figsize=(doub_width,.7*doub_width))
    i = -1
    row,col = 0,-2
    for T in T_range:
        i+= 1
        print("T: ",T,"number of extinctions: ",len(time_of_extinction[i]))
        col += 1
        if col == 4:
            col = 0
            row += 1
        ax[row,col].scatter(time_of_extinction[i],quakes_by_extinction[i],color=colors[i],label=T)
        all_ax.scatter(time_of_extinction[i],quakes_by_extinction[i],color=colors[i],label=T)
    for a in ax[-1,:]:
        a.set_xlabel("Time of extinction (gen)")
    for a in ax[:,0]:
        a.set_ylabel("Quakes by extinction")
    all_ax.set_xlabel("Time of extinction (gen)")
    all_ax.set_ylabel("Quakes by extinction")
    all_ax.legend(bbox_to_anchor=(1,1))
    #all_fig.subplots_adjust(right=.8)

    # plot probability of surviving a quake versus temperature
    fig,ax = plt.subplots(4,4,figsize=(doub_width,.7*doub_width))
    all_fig,all_ax = plt.subplots(2,figsize=(doub_width,doub_width))
    i = -1
    row,col = 0,-2
    for T in T_range:
        i+= 1
        print("T: ",T,"number of extinctions: ",len(time_of_extinction[i]))
        col += 1
        if col == 4:
            col = 0
            row += 1
        ax[row,col].set_title(T)
        # figure out the fraction of survival at each quake number
        n_of_quakes_at_each_quakenumber = extinctions_per_quake[i] + survivals_per_quake[i]
        p_extinction_at_each_quakenumber = extinctions_per_quake[i]/n_of_quakes_at_each_quakenumber
        p_survival_at_each_quakenumber = survivals_per_quake[i]/n_of_quakes_at_each_quakenumber

        ax[row,col].plot(p_extinction_at_each_quakenumber,color="r",label="extinction")
        all_ax[0].plot(p_extinction_at_each_quakenumber,color=colors[i],label=T)
        ax[row,col].plot(p_survival_at_each_quakenumber,color="b",label="survival")
        all_ax[1].plot(p_survival_at_each_quakenumber,color=colors[i],label=T)
    
    for a in ax[-1,:]:
        a.set_xlabel("Quake number")
    for a in ax[:,0]:
        a.set_ylabel("Probability")
    ax[0,-1].legend(bbox_to_anchor=(1,1))
    all_ax[1].set_xlabel("Quake number")
    all_ax[0].set_ylabel("P. extinction")
    all_ax[1].set_ylabel("P. survival")
    all_ax[0].legend(bbox_to_anchor=(1,1))

    plt.show()

    # plot SS_durations
    fig,ax = plt.subplots(figsize=(doub_width,.7*doub_width))
    i = -1
    for T in T_range:
        i+= 1
        print("shape of SS_dur_by_T[i].flatten(): ",np.shape(np.array(SS_dur_by_T[i]).flatten()))
        SS_dur_this_T = []
        for SS_dur_list in SS_dur_by_T[i]:
            if type(SS_dur_list) == list:
                SS_dur_this_T += SS_dur_list
            else: SS_dur_this_T += list(SS_dur_list)
        ax.hist(SS_dur_this_T,color=colors[i],label=T,histtype="step",density=True)
    ax.set_xlabel("SS duration (gen)")
    ax.set_ylabel("PDF")
        #for j in range(len(SS_dur_by_T[i])):
        #    ax[row,col].scatter(quake_times_by_T[i][j],quake_number_by_T[i][j],color=colors[i],label=f"{T}K")

    # quakes vs survival
    fig,ax = plt.subplots(figsize=(doub_width,.7*doub_width))
    i = -1
    max_surv = 0
    for T in T_range:
        i += 1
        surv_now = np.array(survival_by_T)[i,-1]/len(seeds)
        print("Surv_now: ",surv_now)
        if surv_now > max_surv: max_surv = surv_now
        quakes_now = quakes_by_T[i]
        #ax.scatter(np.ones(len(quakes_now))*surv_now,quakes_now,c=colors[i])
        ax.boxplot(quakes_now,positions=[surv_now],widths=.1,showfliers=False)
    ax.set_xlim(0,max_surv+.1)
    ax.set_xlabel("Survival fraction")
    ax.set_ylabel("Number of quakes per experiment")

    # Plot N_timeseries and fourier transform for each T
    fig0,ax0 = plt.subplots(figsize=(doub_width,.7*doub_width)) # med N timeseries # len(T_range),figsize=(4,10),sharex=True)
    fig1,ax1 = plt.subplots(figsize=(doub_width,.7*doub_width)) # Fourier transform of med N # len(T_range),figsize=(4,10),sharex=True)
    fig2,ax2 = plt.subplots(figsize=(doub_width,.7*doub_width)) # sum of fft last ten values #len(T_range),figsize=(4,10),sharex=True)
    fig3,ax3 = plt.subplots(figsize=(doub_width,.7*doub_width)) # final survival vs T
    fig4,ax4 = plt.subplots(figsize=(doub_width,.7*doub_width)) # timeseries of survival
    fig5,ax5 = plt.subplots(1,2,sharey=True,figsize=(doub_width,.9*doub_width)) # poff/pdeath vs survival
    i = -1
    for T in T_range:
        i += 1
        fin = sum(survival_by_T[i]>0)
        #N_timeseries_now = N_by_T[i)
        ax0.plot(medN_by_T[i],color=colors[i],label=f"T={T}")
        fft = fourier_by_T[i]# [:100]
        fft_normzd = fft/sum(fft)
        reverse = np.fft.ifft(fft)
        ax1.plot(fft_normzd,color=colors[i],label=f"T={T}")
        ax2.scatter(T,sum(fft[:10]))
        #ax2.plot(np.linspace(0,maxgens,len(reverse)),reverse,color=colors[i],label=f"T={T}")
        #ax2.plot(N_timeseries_now,color=colors[i],label=f"T={T}")
        #ax3.plot(survival_by_T[i],color=colors[i],label=f"T={T}")
    ax3.plot(T_range,np.array(survival_by_T)[:,-1]/len(seeds),"o-",label="survival")
    np.save(f"npy_files/survival_{exp}_{date}.npy",np.array(survival_by_T)[:,-1])
    total_survival = np.sum(np.array(survival_by_T)[:,-1]/len(seeds)/len(T_range))
    fig3.text(.75,.9,f"Total: {100*total_survival}%")
    if plot_expectation:
        if type(extra_folder) == str and "single" in extra_folder or "single" in extra_folder[0]:
            ax3.plot(all_temps, predicted_survival_at_T, "r", label="predicted survival")
        else:
            ax3.plot(all_temps, predicted_survival_at_T, "r:", label="reference survival")

        # Find poff(T,Topt=T)/pdeath(T) for every T
        if type(extra_folder) == str and "const_pdeath" in extra_folder or "const_pdeath" in extra_folder[0]:
            pd_range = 0.2*np.ones(len(all_temps))
        else: pd_range = np.array(Teq.pdeath(all_temps))
        if type(extra_folder) == str and "var-TRC" in extra_folder or "var-TRC" in extra_folder[0]:
            # var-TRC: max poff is 1 at every T
            po_range = np.ones(len(all_temps))
        elif type(extra_folder) == str and "MTE" in extra_folder or "MTE" in extra_folder[0]:
            # MTE-env: max poff is the MTE-envelope
            po_range = np.array(Teq.MTE(all_temps))
        else: 
            # single TRC
            po_range = Teq.poff_T(all_temps)

        # Also find \sum_Topt(poff(T,Topt))/pdeath(T) for every T
        po_sum_over_Topt_by_T = []
        n_viable_by_T = []
        for T in all_temps:
            po_sum_this_T = 0
            n_viable_by_T.append(0)
            for To in np.r_[264:330]:
                if "single" in exp or "one" in exp:
                    po_this_Topt = Teq.poff_T(T,[Tuniv,11,-3])
                elif "var-TRC" == exp:
                    po_this_Topt = Teq.poff_T(T,[To,11,-3])
                else:
                    po_this_Topt = Teq.poff_T(T,[To,11,-3])*Teq.MTE(T)
                if po_this_Topt > Teq.pdeath(T):
                    n_viable_by_T[-1] += 1
                    po_sum_this_T += po_this_Topt
            po_sum_over_Topt_by_T.append(po_sum_this_T)

        if plot_po_pd:
            axt = plt.twinx(ax3)
            axt.plot(all_temps, po_range/pd_range, "g.-", label=r"p$_\mathrm{off,max}$/p$_\mathrm{death}(T)$")
            axt.plot(all_temps, po_sum_over_Topt_by_T/pd_range/n_viable_by_T, "g--", label=r"$\frac{1}{N_\mathrm{viable}}$ $\sum_\mathrm{T_\mathrm{opt}}$ p$_\mathrm{off}$(T,T$_\mathrm{opt}$)/p$_\mathrm{death}(T)$")
            axt.set_ylabel(r"p$_\mathrm{off}$/p$_\mathrm{death}$",color="g")
            axt.legend()
            axt.set_ylim(0,10)
        # plot expected survival thresholds
        from scipy.optimize import root,fsolve
        def all_survive(T):
            return Teq.poff_T(T) - 3*Teq.pdeath(T)

        def min_survival(T):
            return Teq.poff_T(T) - Teq.pdeath(T)

        if type(extra_folder) == str and "MTE" in extra_folder or "MTE" in extra_folder[0]:
            ax3.plot(all_temps, Teq.Mscale*Teq.MTE(all_temps), 'k', alpha=.2,label="MTE")
        elif type(extra_folder) == str and "var-TRC" in extra_folder:
            pass
        else:
            # find T threshold for guaranteed survival
            Tguess = 298 # initial guess
            T_all_survival = fsolve(all_survive,[Tguess-10,Tguess+10])
            # then find T threshold for minimum survival
            T_min_survival = fsolve(min_survival,[Tguess-10,Tguess+10])
            #ax3.plot([T_all_survival[0],T_all_survival[0]],[0,1],"k",label=r"p$_\mathrm{off}$=3p$_\mathrm{death}$")
            #ax3.plot([T_all_survival[1],T_all_survival[1]],[0,1],"k")
            ax3.plot([T_min_survival[0],T_min_survival[0]],[0,1],"k:",label=r"p$_\mathrm{off}$=p$_\mathrm{death}$")
            ax3.plot([T_min_survival[1],T_min_survival[1]],[0,1],"k:")
            ax3.plot(all_temps, Teq.poff_T(all_temps), "k",alpha=.2,label=r"p$_\mathrm{off}$")
            ax3.set_xlim(all_temps[0],all_temps[-1])
        ax3.plot(all_temps, np.ones(len(all_temps))*Teq.pdeath(all_temps), "k--",alpha=.2,label=r"p$_\mathrm{death}$")
        ax3.legend() #loc="lower left")
    ii = -1
    for survival_this_T in survival_by_T:
        ii += 1
        ax4.plot(survival_this_T/len(seeds),label=T_range[ii],color=colors[ii])
    ax0.set_xlabel("Time (gen)")
    ax0.set_xscale("log")
    ax0.set_ylabel("Median abundance")
    #ax0.subplots_adjust(right=.8)
    pos = ax0.get_position()
    ax0.set_position([pos.x0, pos.y0, pos.width * 0.87, pos.height])
    ax0.legend(bbox_to_anchor=(1,1))
    ax1.legend(loc="upper right")
    ax1.set_xlabel(f"Avg. perturbation frequency (quakes/{maxgens})")
    ax1.set_ylabel("Avg. perturbation magnitude")
    pos = ax1.get_position()
    ax1.set_position([pos.x0, pos.y0, pos.width * 0.87, pos.height])
    ax1.legend(bbox_to_anchor=(1,1))
    ax2.set_xlabel("Temperature (K)")
    ax2.set_ylabel("Energy") # (reverse Fourier transform)")
    pos = ax2.get_position()
    #ax2.set_position([pos.x0, pos.y0, pos.width * 0.87, pos.height])
    #ax2.legend(bbox_to_anchor=(1,1))
    ax3.set_xlabel("Temperature (K)")
    ax3.set_ylabel("Survival fraction")
    ax3.set_ylim(0,1)
    plt.tight_layout()
    ax4.set_xlabel("Time (gen)")
    ax4.set_ylabel("Survival fraction")
    ax4.legend(bbox_to_anchor=[1,1])
    plt.tight_layout()

    # fig5: survival vs poff/pdeath
    prob_ratio = []
    prob_diffs = []
    for T in T_range:
        if type(extra_folder) == str and "const_pdeath" in extra_folder or "const_pdeath" in extra_folder[0]:
            pd = 0.2
        else: pd = Teq.pdeath(T)
        if type(extra_folder) == str and "var-TRC" in extra_folder or "var-TRC" in extra_folder[0]:
            po = 1
        elif type(extra_folder) == str and "MTE" in extra_folder or "MTE" in extra_folder[0]:
            po = Teq.MTE(T)
        else: po = Teq.poff_T(T)
        prob_ratio.append(po/pd)
        prob_diffs.append(po-pd)

    np.save(f"npy_files/survival_{exp}_{date}.npy",np.array(survival_by_T)[:,-1])
    np.save(f"npy_files/prob_ratio_{exp}_{date}.npy",np.array(prob_ratio))
    np.save(f"npy_files/prob_diffs_{exp}_{date}.npy",np.array(prob_diffs))

    ax5[0].scatter(prob_ratio,np.array(survival_by_T)[:,-1]/len(seeds),label="survival")
    ax5[0].set_xlabel(r"p$_\mathrm{off,scaler}$/p$_\mathrm{death}$")
    ax5[0].set_ylabel("Survival fraction")
    ax5[1].scatter(prob_diffs,np.array(survival_by_T)[:,-1]/len(seeds))
    ax5[1].set_xlabel(r"p$_\mathrm{off,scaler}$ - p$_\mathrm{death}$")
    plt.tight_layout()

    fig0.savefig(f"figures/N_timeseries_{exp}_{date}.pdf")
    fig1.savefig(fig_locat+f"fft_{experiment[:-1]}_{date}.pdf")
    fig3.savefig(fig_locat+f"survival_by_T_{experiment[:-1]}_{date}.pdf")

    plt.show()


else:
    modelrun, all_spcs = format_outputs(seed,T)

    # %% PLOTTING
    # TEST PLOT
    fig,ax = plt.subplots()
    total_repeats = 0
    repeat_times = []
    for spc in all_spcs:
        ax.plot(spc.times_alive)
        repeats = - len(set(spc.times_alive)) + len(spc.times_alive)
        if repeats != 0:
            if spc.times_alive[-1] == spc.times_alive[-2]:
                spc.times_alive.pop(-1)
                spc.populations.pop(-1)
                spc.is_core.pop(-1)
                spc.f_timeseries.pop(-1)
            #print(spc.times_alive[-3:])
            total_repeats+=1
            #print(f"Repeat in times alive: species {spc.ID}, {repeats} repeats")
            new_list = []
            for t in spc.times_alive:
                if t not in new_list:
                    new_list.append(t)
                else: repeat_times.append(t)
    #print("Total repeats: ",total_repeats)
    ax.set_ylabel("Times alive of each spc.")
    ax.set_xlabel("Cumulative time existing of each spc.")
    if total_repeats>0:
        fig,ax = plt.subplots()
        ax.hist(repeat_times)
        ax.set_xlabel("Repeat times")
        ax.set_ylabel("Number or repeats")
        plt.show()


    # PLOT: species populations over time
    fig,ax = plt.subplots()
    for spc in all_spcs:
        if len(spc.populations) > 0:
            ax.plot(spc.times_alive,spc.populations)
    ax.set_xlabel("Time (gen)")
    ax.set_ylabel("Species population")

    # PLOT: fitness and interactions of ecosystem
    fig,ax = plt.subplots(2,sharex=True)
    for spc in all_spcs:
        if len(spc.times_alive) > 0:
            N_at_t = np.array(modelrun[0].N_timeseries)[np.array(spc.times_alive)-1]
        else:
            print("Can't use these times alive as indices: ",spc.times_alive,type(spc.times_alive))
        ax[0].plot(spc.times_alive,spc.f_timeseries)
        ax[1].plot(spc.times_alive,np.array(spc.f_timeseries)+.2*N_at_t)
    ax[1].set_xlabel("Time (gen)")
    ax[0].set_ylabel(r"$f_i$(t)")
    ax[1].set_ylabel(r"$J_{ij} N_j$")

    # PLOT: fitness and interactions of core
    fig,ax = plt.subplots(2,sharex=True)
    for spc in all_spcs:
        if sum(spc.is_core) > 10:
            times_in_core = np.array(spc.times_alive)[np.where(np.array(spc.is_core)>0)]
            f_in_core = []
            for i in range(len(spc.f_timeseries)):
                if spc.times_alive[i] in times_in_core:
                    f_in_core.append(spc.f_timeseries[i])
            if len(f_in_core) != len(times_in_core):
                print("error: shapes don't match")
                print("f_in_core: ",f_in_core)
                print("times_in_core: ",times_in_core)
                for i in range(len(spc.f_timeseries)):
                    if spc.times_alive[i] in times_in_core:
                        print("time in core: ",spc.times_alive[i])
            #N_at_t = np.array(modelrun[0].N_timeseries)[np.array(spc.times_alive)-1]
            N_at_t = np.array(modelrun[0].N_timeseries)[times_in_core-1]
            ax[0].plot(times_in_core,f_in_core)
            ax[1].plot(times_in_core,f_in_core+.2*N_at_t)
    ax[1].set_xlabel("Time (gen)")
    ax[0].set_ylabel(r"$f_i$(t) of core species")
    ax[1].set_ylabel(r"$\sum_j(J_{ij} n_j)$ of core species")

    # PLOT: poff,eco and poff,core
    rscale = Teq.poff_T(int(T))
    print(f"rscale at {T}K: ",rscale)
    pdeath = Teq.pdeath(int(T))
    print(f"pdeath at {T}K: ",pdeath)

    fig,ax = plt.subplots(2,sharex=True)
    if poffeco_inState:
        for m in modelrun:
            poffeco = m.poffeco_t
            ax[0].plot(poffeco)
            ax[1].plot(np.array(poffeco)*rscale - pdeath,"r")
            ax[1].plot(np.zeros((len(poffeco))),'k')
        ax[1].set_xlabel("Time (gen)")
        ax[0].set_ylabel(r"$p_\mathrm{off,eco}$")
        ax[1].set_ylabel(r"$r_\mathrm{scale} p_\mathrm{off,eco}-p_\mathrm{death}$")
        plt.tight_layout()
    if 1 ==1: 
        gensrun = len(modelrun[0].N_timeseries)
        poffeco = np.zeros((gensrun,))
        fig,ax = plt.subplots(3,sharex=True)
        for spc in all_spcs:
            if len(spc.times_alive) > 0:
                poff_i = 1/(1+np.exp(-np.array(spc.f_timeseries)))
                poffeco[np.array(spc.times_alive)-1] += poff_i*spc.populations
                if sum(spc.is_core) > 10:
                    times_in_core = np.array(spc.times_alive)[np.where(np.array(spc.is_core)>0)]
                    poff_core = []
                    for t in range(len(spc.f_timeseries)):
                        if spc.times_alive[t] in times_in_core:
                            poff_core.append(poff_i[t])
                    #N_at_t = np.array(modelrun[0].N_timeseries)[np.array(spc.times_alive)-1]
                    #N_at_t = np.array(modelrun[0].N_timeseries)[times_in_core-1]
                    ax[0].plot(spc.times_alive,poff_i)
                    ax[1].plot(times_in_core,poff_core)
                    #ax[1].plot(times_in_core,f_in_core+.2*N_at_t)
        ax[2].plot(poffeco/np.array(modelrun[0].N_timeseries),label=r"$p_\mathrm{off}$")
        ax[2].plot(rscale*poffeco/np.array(modelrun[0].N_timeseries),label=r"$r_\mathrm{scale}*p_\mathrm{off}$")
        ax[2].plot(np.ones((len(poffeco),))*pdeath,'r--',label=r"p$_\mathrm{death}$")
        ax[2].legend(loc="upper right")
        ax[2].set_xlabel("Time (gen)")
        ax[0].set_ylabel(r"$p_\mathrm{off,i}$")
        ax[1].set_ylabel(r"$p_\mathrm{off,i}$ of core species")
        ax[2].set_ylabel(r"$p_\mathrm{off,eco}$")

    # PLOT: Fourier analysis of ecosystem abundance
    fig,ax = plt.subplots(3)
    for m in modelrun:
        srs = m.N_timeseries
        fft = np.fft.fft(srs)
        reverse = np.fft.ifft(fft[:100])
        ax[0].plot(srs)
        ax[1].plot(fft)
        ax[2].plot(np.linspace(0,gensrun,len(reverse)),reverse)
    for a in [ax[0],ax[2]]:
        a.set_xlabel("Time")
        a.set_ylabel("Abundance")
    ax[1].set_xlabel(f"Frequency (quakes/{maxgens/1000}k gen)")
    ax[1].set_ylabel("Power")
    plt.tight_layout()


    if (type(extra_folder) == str and "MTE" in extra_folder) or ("MTE" in extra_folder[0]) or "var" in extra_folder:
        fig,ax = plt.subplots()
        for spc in all_spcs:
            if len(spc.times_alive) > 0:
                ax.plot(spc.times_alive,spc.Topt*np.ones(len(spc.times_alive)))
        ax.set_xlabel("Time (gen)")
        ax.set_ylabel(r"$T_\mathrm{opt,i}$ (K)")

    plt.show()
