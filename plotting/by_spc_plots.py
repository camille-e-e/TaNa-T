import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import skew
# import Teq and classes from output location

multi_inputs = True
indiv_seeds = 12
maxgens = 10_000 #50000
plot_expectation = True
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
singleTRC = Experiment(seeds=np.r_[2200:2250],date="Feb_24",extra_folder="one-TRC-scaledup",locat='/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/')
#MTEenv = Experiment(seeds=np.r_[100:700],date="Sep_29",dates=["Sep_29","Oct_11"],extra_folder=["MTE_TPC_combo","MTE-env"],locat='/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/')
MTEenv = Experiment(seeds=np.r_[100:200],date="Sep_29",extra_folder="MTE_TPC_combo",locat='/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/')


# vary pmut with T
#oneTRC_varypmut = Experiment(date="Mar_30_23",extra_folder="single-TRC/varypmut")
oneTRC_varypmut = Experiment(date="Jun_21_23",extra_folder="single-TRC/varypmut-Tref303") #,locat='/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/')
varTRC_varypmut = Experiment(date="Apr_13_23",extra_folder="var-TRC/varypmut")
#MTEenv_varypmut = Experiment(date="Apr_20_23",extra_folder="MTE-env/varypmut")
MTEenv_varypmut = Experiment(date="Apr_20_23",dates=["Apr_20_23","May_01_23","May_02_23"],extra_folder="MTE-env/varypmut",seeds=np.r_[1000:1050])
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
MTEenv_scaledup = Experiment(date="May_09_23",dates=["May_09_23","May_19_23"],seeds=np.r_[1000:1050],extra_folder="MTE-env/scaledup")
MTEenv_scaledup_symmetrical = Experiment(date="May_09_23",extra_folder="MTE-env/scaledup-skew0")

# weird tests
MTEenv_scaledup_backwards = Experiment(date="May_29_23",extra_folder="MTE-env/scaledup-skew3")
varTRC_constdeath= Experiment(date="May_30_23",dates=["May_30_23","Jun_01_23"],seeds=np.r_[1000:1300],extra_folder="var-TRC/const-pdeath")
varTRC_constdeathmut = Experiment(date="Jun_01_23",seeds=np.r_[1000:1300],extra_folder="var-TRC/const-pdeath-pmut")

#experiments = [oneTRC_varypmut] #_scaledup_backwards] #_moreTopt]
experiments = [oneTRC_varypmut_more] #constdeathmut] #_scaledup_backwards] #_moreTopt]
#experiments = [MTEenv_scaledup] #_scaledup_backwards] #_moreTopt]

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

def format_outputs(seed,T):
    if T:
        if T < 1:
            T = f"{T:.2f}"
        else: T = f"{int(T)}"
    else: T = "no"

    # Final chunks or only chunks
    found = 0
    for op in outpath_list:
        if os.path.exists(op+f"modelrun_seed{seed}_{T}K.npy"):
            modelrun = np.load(op+f"modelrun_seed{seed}_{T}K.npy",allow_pickle=True)
            all_spcs = np.load(op+f"species_objects_seed{seed}_{T}K.npy",allow_pickle=True)
            found = 1
            break
    if not found: 
        print(f"+++Seed {seed} not found+++")
        return [False]

    # Determine if there are previous generations of the same T and seed in different files
    count = 0
    list1 = []
    list2 = []
    for f in os.listdir(op):
        if f.startswith(f"modelrun_seed{seed}_{T}K"):
            count += 1
            list1.append(f)
        elif f.startswith(f"species_objects_seed{seed}_{T}K"):
            list2.append(f)
    if count > 1:
        order = []
        filelist = []
        filelist2 = []
        filelist_states = []
        filelist_species = []
        for f in list1: # os.listdir(outpath):
            if "gen" in f:
                #print(f)
                idx1 = f.index('seed') + len(f"seed{seed}_{T}K_")
                idx2 = f.index('gen')
                finalgen = int(f[idx1:idx2])
                order.append(finalgen)
                filelist.append(f)
        for f in list2:
            if "gen" in f:
                filelist2.append(f)
        order_sorted = np.sort(order)
        indices = []
        for o in order_sorted:
            indices.append(order.index(o))
        for i in indices:
            filelist_states.append(filelist[i])
            filelist_species.append(filelist2[i])

    if count > 1:

        # parse together species from each file
        all_spcs = list(all_spcs) #convert from np.array
        
        # get list of IDs in all_spcs
        IDs = []
        for spc in all_spcs:
            IDs.append(spc.ID)

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
        if gensrun > 0:
            big_state.N_timeseries[-gensrun:] = modelrun[0].N_timeseries
            big_state.D_timeseries[-gensrun:] = modelrun[0].D_timeseries
            big_state.coreN_timeseries[-gensrun:] = modelrun[0].coreN_timeseries
            big_state.coreD_timeseries[-gensrun:] = modelrun[0].coreD_timeseries
            big_state.inSS_timeseries[-gensrun:] = modelrun[0].inSS_timeseries
            if len(modelrun[0].poffeco_t) == gensrun:
                big_state.poffeco_t[-gensrun:] = modelrun[0].poffeco_t
            else: 
                gens_poff = len(modelrun[0].poffeco_t)
                #print("gens_poff: ",gens_poff)
                if gens_poff > 0:
                    big_state.poffeco_t[-gens_poff:] = modelrun[0].poffeco_t
                #print("gens run: ",gensrun," but len(poffeco): ",len(modelrun[0].poffeco_t))

        for f in filelist_species:
            these_spcs = np.load(outpath+f,allow_pickle=True)
            for spc in these_spcs:
                if spc.ID in IDs:
                    found = 0
                    for spc2 in all_spcs:
                        if spc2.ID == spc.ID:
                            spc2.times_alive = spc2.times_alive + spc.times_alive
                            spc2.populations = spc2.populations + spc.populations
                            spc2.is_core = spc2.is_core + spc.is_core
                            spc2.f_timeseries = spc2.f_timeseries + spc.f_timeseries
                            found = 1
                            break
                    if not found: print("error, species not found")
                else:
                    IDs.append(spc.ID)
                    all_spcs.append(spc)

        # add each time chunk to the overall state timeseries
        for f in filelist_states: #elif f.startswith(f"modelrun_seed{seed}"):
            if 'gen' in f: # already added output from this file
                m_list = np.load(op+f,allow_pickle=True)
                idx1 = f.index('seed') + len(f"seed{seed}_{T}K_")
                idx2 = f.index('gen')
                finalgen = int(f[idx1:idx2])
                #print("finalgen: ",finalgen)
                for m in m_list:
                    big_state.N_timeseries[finalgen-write_freq:finalgen] = m.N_timeseries
                    big_state.D_timeseries[finalgen-write_freq:finalgen] = m.D_timeseries
                    big_state.coreN_timeseries[finalgen-write_freq:finalgen] = m.coreN_timeseries
                    big_state.coreD_timeseries[finalgen-write_freq:finalgen] = m.coreD_timeseries
                    big_state.inSS_timeseries[finalgen-write_freq:finalgen] = m.inSS_timeseries
                    big_state.poffeco_t[finalgen-write_freq:finalgen] = m.poffeco_t

    if count > 1: 
        modelrun = [big_state]
    return modelrun, all_spcs
                
# For each seed and T, collect outputs
files_found = 0
if multi_inputs:
    N_by_T = []
    D_by_T = []
    coreN_by_T = []
    coreD_by_T = []
    poffeco_by_T = []
    Jij_by_T = []
    fourier_by_T = []
    survival_by_T = []
    quakes_by_T = []
    SAD_by_T = []
    skew_by_T = []
    
    T_to_sample = [280,292,301,310]
    SAD_fig, SAD_ax = plt.subplots(4,indiv_seeds,sharex=True,sharey=True,figsize=(doub_width,.7*doub_width))
    SAD_ax[0,1].set_title(f"Species abundance distributions")
    for a in SAD_ax[3,:]:
        a.set_xlabel("Population (%)")
    i = -1
    row = -1
    for a in SAD_ax[:,0]:
        T = T_to_sample[i]
        a.set_ylabel(f"T={T}")
    
    i = -1
    for T in T_range:
        if indiv_seeds > 0:
            if T in T_to_sample:
                row += 1
                indiv_fig,indiv_ax = plt.subplots(5,indiv_seeds,sharex=True,sharey='row',figsize=(8,6))
                # SAD_fig, SAD_ax = plt.subplots(1,indiv_seeds)
                for k in range(indiv_seeds):
                    indiv_ax[4,k].set_xlabel("Time (gen)")
                indiv_ax[0,0].set_ylabel("Abundance")
                indiv_ax[1,0].set_ylabel("Core abund.")
                indiv_ax[2,0].set_ylabel("Diversity")
                indiv_ax[3,0].set_ylabel("Core div.")
                indiv_ax[4,0].set_ylabel("Eco. poff")
                indiv_ax[0,0].set_title(f"T: {T}K")


        N_by_T.append([]) #np.zeros(maxgens))
        coreN_by_T.append([]) #np.zeros(maxgens))
        D_by_T.append([]) #np.zeros(maxgens))
        coreD_by_T.append([]) #np.zeros(maxgens))
        poffeco_by_T.append([])
        Jij_by_T.append([])
        fourier_by_T.append(np.zeros(maxgens))
        survival_by_T.append(np.zeros(maxgens))
        quakes_by_T.append([])
        SAD_by_T.append([])
        skew_by_T.append([])
        i += 1
        j = -1
        for seed in seeds:
            j += 1
            modelrun,all_spcs = format_outputs(seed,T)
            if type(modelrun[0]) != bool:
                files_found += 1
                # perform Fourier transform
                srs = np.zeros(maxgens)
                m = modelrun[0]
                srs = m.N_timeseries
                #N_by_T[i][:len(srs)] += srs
                N_by_T[i].append(srs)
                D_by_T[i].append(m.D_timeseries)
                coreN_by_T[i].append(m.coreN_timeseries)
                coreD_by_T[i].append(m.coreD_timeseries)
                poffeco_by_T[i].append(m.poffeco_t)
                # find SAD at end of this run
                this_SAD = []
                final_N = m.N_timeseries[-1]
                g = maxgens-1
                for spc in all_spcs:
                    if g in spc.times_alive: # >= maxgens-1:
                        idx = list(spc.times_alive).index(g)
                        this_SAD.append(spc.populations[idx]) #/final_N)
                SAD_by_T[i] = np.hstack([SAD_by_T[i],this_SAD])
                # skewness of SAD
                if len(m.N_timeseries) >= maxgens-1:
                    skew_now = skew(this_SAD)
                else: skew_now = np.nan
                skew_by_T[i].append(skew_now)

                # find average fitness (fav)
                fav = - np.log(1/np.array(m.poffeco_t) -1)
                mu = 0.1
                Jij_av = list(fav + mu*np.array(m.N_timeseries[:len(fav)]))
                Jij_by_T[i].append(Jij_av)
                fft = np.fft.fft(srs)
                reverse = np.fft.ifft(fft[:100])
                fourier_by_T[i][:len(fft)] += np.array(fft,dtype=float)
                survival_by_T[i][:len(srs)] += np.ones(len(srs))
                # count quakes in this run
                quakes_this_run = 0
                for t in range(10,len(m.inSS_timeseries)):
                    if m.inSS_timeseries[t-1] == 1 and m.inSS_timeseries[t] == 0:
                        quakes_this_run += 1
                quakes_by_T[i].append(quakes_this_run)
                
                # plot individual runs
                if T in [280,292,301,310] and j < indiv_seeds:
                    indiv_ax[0,j].plot(m.N_timeseries)
                    indiv_ax[1,j].plot(m.coreN_timeseries)
                    indiv_ax[2,j].plot(m.D_timeseries)
                    indiv_ax[3,j].plot(m.coreD_timeseries)
                    indiv_ax[4,j].plot(m.poffeco_t)

                    # find SAD for every 5 gens of last 20 
                    SADs_by_t = []
                    for g in maxgens - np.r_[1,6,11,16]:
                        this_SAD = []
                        final_N = m.N_timeseries[-1]
                        for spc in all_spcs:
                            if g in spc.times_alive: # >= maxgens-1:
                                idx = list(spc.times_alive).index(g)
                                this_SAD.append(spc.populations[idx]/final_N)
                        SADs_by_t.append(this_SAD)
                    SAD_ax[row,j].hist(SADs_by_t)

                if j == indiv_seeds and T == T_to_sample[-1]:
                    plt.savefig(f"figures/SAD_{exp}_{date}.pdf")
                #    plt.show()

                # add nans to makeup empty spots
                add_nans = maxgens - len(srs)
                if add_nans > 0:
                    N_by_T[i][-1].append(np.nan)
                    coreN_by_T[i][-1].append(np.nan)
                    D_by_T[i][-1].append(np.nan)
                    coreD_by_T[i][-1].append(np.nan)
                    poffeco_by_T[i][-1].append(np.nan)
                    Jij_by_T[i][-1].append(np.nan)
                    #SAD_by_T[i][-1].append(np.nan)
                    add_nans -= 1
                    if add_nans > 0:
                        N_by_T[i][-1][-1:-1] = np.ones(add_nans)*np.nan
                        coreN_by_T[i][-1][-1:-1] = np.ones(add_nans)*np.nan
                        D_by_T[i][-1][-1:-1] = np.ones(add_nans)*np.nan
                        coreD_by_T[i][-1][-1:-1] = np.ones(add_nans)*np.nan
                        poffeco_by_T[i][-1][-1:-1] = np.ones(add_nans)*np.nan
                        Jij_by_T[i][-1][-1:-1] = np.ones(add_nans)*np.nan
                        #SAD_by_T[i][-1:-1] = np.ones(add_nans)*np.nan
                add_nans = maxgens - len(m.poffeco_t)
                if add_nans > 0:
                    poffeco_by_T[i][-1].append(np.nan)
                    Jij_by_T[i][-1].append(np.nan)
                    add_nans -= 1
                    if add_nans > 0:
                        poffeco_by_T[i][-1][-1:-1] = np.ones(add_nans)*np.nan
                        Jij_by_T[i][-1][-1:-1] = np.ones(add_nans)*np.nan

    np.save(f"npy_files/skew_by_T_{exp}_{date}.npy",skew_by_T)

    #plt.figure()
    #for i in range(len(T_range)): 
    #    plt.plot(N_by_T[i][:])
    print("Files found: ", files_found)

    # allocate space for timeseries of medians and quartiles for eac T
    medN_by_T, medD_by_T, medcoreN_by_T, medcoreD_by_T, medpoffeco_by_T,medJij_by_T = [],[],[],[],[],[]
    q1N_by_T, q1D_by_T, q1coreN_by_T, q1coreD_by_T, q1poffeco_by_T,q1Jij_by_T = [],[],[],[],[],[]
    q3N_by_T, q3D_by_T, q3coreN_by_T, q3coreD_by_T, q3poffeco_by_T,q3Jij_by_T = [],[],[],[],[],[]
    
    i = -1
    for T in T_range:
        i += 1
        medN_by_T.append(np.nanquantile(N_by_T[i],.5,axis=0))
        medD_by_T.append(np.nanquantile(D_by_T[i],.5,axis=0))
        medcoreN_by_T.append(np.nanquantile(coreN_by_T[i],.5,axis=0))
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
        q1D_by_T.append(np.nanquantile(D_by_T[i],.25,axis=0))
        q1coreN_by_T.append(np.nanquantile(coreN_by_T[i],.25,axis=0))
        q1coreD_by_T.append(np.nanquantile(coreD_by_T[i],.25,axis=0))

        q3N_by_T.append(np.nanquantile(N_by_T[i],.75,axis=0))
        q3D_by_T.append(np.nanquantile(D_by_T[i],.75,axis=0))
        q3coreN_by_T.append(np.nanquantile(coreN_by_T[i],.75,axis=0))
        q3coreD_by_T.append(np.nanquantile(coreD_by_T[i],.75,axis=0))

    final_N_by_T = np.array(medN_by_T)[:,-1]
    final_D_by_T = np.array(medD_by_T)[:,-1]
    final_coreN_by_T = np.array(medcoreN_by_T)[:,-1]
    final_coreD_by_T = np.array(medcoreD_by_T)[:,-1]
    print("shape of medJij_by_T: ",np.shape(medJij_by_T))
    print("shape of medpoffeco_by_T: ",np.shape(medpoffeco_by_T))
    final_poffeco_by_T = np.array(medpoffeco_by_T)[:,-1]
    final_Jij_by_T = np.array(medJij_by_T)[:,-1]

    if sum(np.array(final_D_by_T) == 0) > 1: # if any elements of final_D are zero, eliminate
        final_N_by_T = np.array(final_D_by_T)[np.array(final_D_by_T) > 2]
        final_D_by_T = np.array(final_D_by_T)[np.array(final_D_by_T) > 2]
        final_coreN_by_T = np.array(final_coreN_by_T)[np.array(final_D_by_T) > 2]
        final_coreD_by_T = np.array(final_coreD_by_T)[np.array(final_D_by_T) > 2]
        final_poffeco_by_T = np.array(final_poffeco_by_T)[np.array(final_D_by_T) > 2]
    
    # define some colors
    cmap = plt.get_cmap("inferno") #RdPu")
    colors = cmap(np.linspace(.2,1,len(T_range)))
    cmap2 = plt.get_cmap("Greys")
    colors2 = cmap2(np.linspace(.2,1,5))


    # plot cumulative SADs 
    fig,ax = plt.subplots(1,3,figsize=(doub_width,.5*doub_width))
    i = -1
    binwidth= 25 #0.05
    binmax = 1000
    for T in T_range:
        i += 1
        SAD_now = np.array(SAD_by_T[i])
        # eliminate nans
        SAD_now = SAD_now[np.logical_not(np.isnan(SAD_now))]
        # if there are more than 1 value of non-nan species in the SAD:
        if len(SAD_now) > 1: # not np.isnan(SAD_now).all():
            #SAD_nonzero = SAD_now[SAD_now != 0]
            #SAD_nonzero = SAD_nonzero[np.logical_not(np.isnan(SAD_nonzero))]
            SAD_log = np.log(SAD_now)
            SAD_log = SAD_log[np.isfinite(SAD_log)]
            #logSAD_finite_by_T.append(np.log(SAD)[np.isfinite(np.log(SAD))])
            #print("is SAD_log finite? ",sum(np.isnan(SAD_log))
            ax[0].hist(SAD_now,color=colors[i],bins=np.r_[0:binmax+binwidth:binwidth],histtype="step",density=True)
            ax[1].hist(SAD_log,color=colors[i],density=True,histtype="step")
            ax[2].hist(SAD_now,color=colors[i],bins=np.r_[0:binmax+binwidth:binwidth],histtype="step",cumulative=True,density=True)
    for a in ax:
        a.set_xlabel("Population") # fraction")
    ax[1].set_xlabel("log Population")
    ax[0].set_ylabel("Probability density")
    ax[1].set_ylabel("Probability density")
    #ax[1].set_xscale("log")
    ax[2].set_ylabel("Cumulative probability density")
    np.save(f"npy_files/SAD_by_T_{exp}_{date}.npy",np.array(SAD_by_T))
    plt.tight_layout()
    plt.savefig(f"figures/SAD_all_{exp}_{date}.pdf")

    fig,ax = plt.subplots(figsize=(doub_width,.7*doub_width))
    logSAD_by_T = []
    i = -1
    for SAD in SAD_by_T:
        i += 1
        logSAD = np.log(SAD)
        ax.hist(logSAD[np.isfinite(logSAD)],histtype='step',cumulative=True,color=colors[i],bins=50,density=True,label=f"T={T_range[i]}")
    ax.set_xlabel("log population")
    ax.set_ylabel("probability density")
    ax.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig(f"figures/logSADs_by_T_{exp}_{date}.pdf")

    # scatter skewness and abundance
    fig,ax = plt.subplots(figsize=(doub_width,.7*doub_width))
    dfig,dax = plt.subplots(figsize=(doub_width,.7*doub_width))
    adfig,adax = plt.subplots(figsize=(doub_width,.7*doub_width))
    Tfig,Tax = plt.subplots(2,2,sharey=True,figsize=(doub_width,.7*doub_width))
    #Tafig,Taax = plt.subplots(figsize=(doub_width,.7*doub_width))

    for i in range(len(T_range)):
        color = colors[i]
        div_this_T = np.array(D_by_T)[i,:,-2]
        ab_this_T = np.array(N_by_T)[i,:,-2]
        div_normed = div_this_T/max(div_this_T)
        ab_normed = ab_this_T/max(ab_this_T)
        div_colors = cmap(div_normed)
        ab_colors = cmap(ab_normed)
        print(f"size of np.array(N_by_T)[i,:,-2]: {np.shape(np.array(N_by_T)[i,:,-2])}")
        print(f"size of skew_by_T[{i}]: {np.shape(skew_by_T[i])}")
        if len(np.array(N_by_T)[i,:,-2]) == len(skew_by_T[i]):
            ax.scatter(np.array(N_by_T)[i,:,-2],skew_by_T[i],c=color,label=f"T={T_range[i]}K")
            dax.scatter(np.array(D_by_T)[i,:,-2],skew_by_T[i],c=color,label=f"T={T_range[i]}K")
            Tax[0,0].scatter(np.array(N_by_T)[i,:,-2],skew_by_T[i],c=color,label=f"T={T_range[i]}K")
            Tax[0,1].scatter(np.array(D_by_T)[i,:,-2],skew_by_T[i],c=color,label=f"T={T_range[i]}K")
        adax.scatter(np.array(N_by_T)[i,:,-2],np.array(D_by_T)[i,:,-2],c=color,label=f"T={T_range[i]}K")
        Tax[1,1].scatter(np.ones(len(skew_by_T[i]))*T_range[i],skew_by_T[i],c=div_colors,label=f"T={T_range[i]}K")
        Tax[1,0].scatter(np.ones(len(skew_by_T[i]))*T_range[i],skew_by_T[i],c=ab_colors,label=f"T={T_range[i]}K")
    ax.set_xlabel("Final abundance")
    dax.set_xlabel("Final species richness")
    for a in [ax,dax]:
        a.set_ylabel("Skewness of SAD")
        a.legend(bbox_to_anchor=(1,1))
    fig.tight_layout()
    dfig.tight_layout()

    adax.set_xlabel("Abundance")
    adax.set_ylabel("Species richness")
    adax.legend(bbox_to_anchor=(1,1))
    adfig.tight_layout()

    for a in [Tax[1,0],Tax[1,1]]:
        a.set_xlabel("Temperature (K)")
    Tax[0,0].set_xlabel("Abundance")
    Tax[0,1].set_xlabel("Species richness")
    Tax[0,0].set_ylabel("Skewness of SAD")
    Tax[1,0].set_ylabel("Skewness of SAD")

    Tax[0,0].text(0.05,0.9,"a)",transform=Tax[0,0].transAxes,horizontalalignment="left",verticalalignment="center")
    Tax[0,1].text(0.05,0.9,"b)",transform=Tax[0,1].transAxes,horizontalalignment="left",verticalalignment="center")
    Tax[1,0].text(0.05,0.9,"c)",transform=Tax[1,0].transAxes,horizontalalignment="left",verticalalignment="center")
    Tax[1,1].text(0.05,0.9,"d)",transform=Tax[1,1].transAxes,horizontalalignment="left",verticalalignment="center")

    Tfig.tight_layout()
    Tfig.savefig(f"figures/skew_vs_T_coloredND_{exp}_{date}.pdf")
    #Tafig.savefig(f"figures/skew_vs_T_coloredN_{exp}_{date}.pdf")
    # dummy plot to get image for colorbar--i think it's not necessary but bandaid for now
    #fig,ax = plt.subplots()
    #Tim = plt.imshow(np.vstack([np.r_[0:150],np.r_[0:150]]),vmax=150/.9,cmap=cmap)
    #Tdfig.colorbar(Tim,ax=Tax,label="species richness")
    #Tdfig.tight_layout()
    #Tdfig.savefig(f"figures/skew_vs_T_coloredD_{exp}_{date}.pdf")

    plt.show()
    

    
    # plot N(T) at a few example ts
    fig,ax = plt.subplots(1,2,sharey=True,figsize=(doub_width,.7*doub_width))
    dfig,dax = plt.subplots(1,2,sharey=True,figsize=(doub_width,.7*doub_width))
    i = -1
    for T in T_range:
        i += 1
        ax[0].plot(medN_by_T[i],color=colors[i],label=f"T={T}")
        dax[0].plot(medD_by_T[i],color=colors[i],label=f"T={T}")
    i = -1
    for t in [1,10,100,1000,9900]:
        i += 1
        ax[1].plot(T_range,np.mean(np.array(medN_by_T)[:,t-1:t+10],axis=1),c=colors2[i],label=f"t={t}")
        dax[1].plot(T_range,np.mean(np.array(medD_by_T)[:,t-1:t+10],axis=1),c=colors2[i],label=f"t={t}")
    ax[0].set_ylabel("Median abundance")
    dax[0].set_ylabel("Median species richness")
    #fig.adjust_right(0.8)
    for a in [ax[1],dax[1]]:
        a.set_xlabel("Temperature (K)")
        a.legend(loc="upper left")
    for a in [ax[0],dax[0]]:
        a.set_xlabel("Time (t)")
        a.set_xscale("log")
        a.legend(bbox_to_anchor=(1,1))
    fig.tight_layout()
    plt.savefig(f"figures/N_vs_t_T_{exp}_{date}.pdf")
    dfig.tight_layout()

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
            
        expected_N = np.load("expected_N.npy")*np.ones(len(T_range))
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
    four_outputs['refN'] = expected_N
    four_outputs['refD'] = expected_div

    four_outputs['med_N'] = final_N_by_T
    four_outputs['q1_N'] = np.array(q1N_by_T)[:,-1]
    four_outputs['q3_N'] = np.array(q3N_by_T)[:,-1]

    four_outputs['med_D'] = final_D_by_T
    four_outputs['q1_D'] = np.array(q1D_by_T)[:,-1]
    four_outputs['q3_D'] = np.array(q3D_by_T)[:,-1]

    four_outputs['med_coreN'] = np.array(medcoreN_by_T)[:,-1]
    four_outputs['q1_coreN'] =  np.array(q1coreN_by_T)[:,-1]
    four_outputs['q3_coreN'] = np.array(q3coreN_by_T)[:,-1]

    four_outputs['med_coreD'] = np.array(medcoreD_by_T)[:,-1]
    four_outputs['q1_coreD'] = np.array(q1coreD_by_T)[:,-1]
    four_outputs['q3_coreD'] = np.array(q3coreD_by_T)[:,-1]

    np.save(f"npy_files/final_stats_{exp}_{date}.npy",np.array(four_outputs))
    np.save(f"npy_files/N_by_T_{exp}_{date}.npy",N_by_T)
    np.save(f"npy_files/D_by_T_{exp}_{date}.npy",D_by_T)
    np.save(f"npy_files/core_N_by_T_{exp}_{date}.npy",coreN_by_T)
    np.save(f"npy_files/core_D_by_T_{exp}_{date}.npy",coreD_by_T)
    
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
    np.save(f"npy_files/quakes_by_T_{exp}_{date}.npy",quakes_by_T)
    plt.savefig(f"figures/quake_boxplots_{exp}_{date}.pdf")

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
    ax5[0].scatter(prob_ratio,np.array(survival_by_T)[:,-1]/len(seeds),label="survival")
    ax5[0].set_xlabel(r"p$_\mathrm{off,max}$/p$_\mathrm{death}$")
    ax5[0].set_ylabel("Survival fraction")
    ax5[1].scatter(prob_diffs,np.array(survival_by_T)[:,-1]/len(seeds))
    ax5[1].set_xlabel(r"p$_\mathrm{off}$ - p$_\mathrm{death}$")
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
