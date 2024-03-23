from test_classes import Species, State
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/cfebvre/repos/the_model_y3/src/geoTNM/')
import MTE_TPC_combo as Teq

plot_multiple = True # otherwise plot a single run with parameters defined below
num_cols = 9 # number of example seeds to plot

# only used if plot_multiple == False
seed =1112 # 1108 1144 #113
T = 292
T_range = np.r_[254:360] # to use for plotting
maxgens = 10000 #50000
date = 'May_03_23' # 'Feb_16' # test write_freq
extra_folder = 'varypmut' # 'test_write_freq'
experiment = "SteadyT/MTE-env" #"test"
write_freq = 4000 # False #10
poffeco_inState = True

class Experiment:
    def __init__(self,date="",seeds=np.r_[1000:1050],T_range=np.r_[274:320:3],dates=[],experiment="SteadyT/",extra_folder="",L=20,C=100,theta=.25,mu=.1,pmut=0.01,locat='/net/venus/kenes/user/cfebvre/out_venus/TTNM_outputs/',write_freq=4000,poffeco_inState=True):
        self.date = date
        self.seeds = seeds
        self.T_range = T_range
        self.dates = dates
        self.experiment = experiment
        self.extra_folder = extra_folder
        self.L,self.C = L,C
        self.theta, self.mu, self.pmut = theta, mu, pmut
        self.locat = locat
        self.write_freq = write_freq
        if locat.startswith("/net/"):
            self.outpath = locat + f'{experiment}{extra_folder}/{date}/'
        else:
            self.outpath = locat+f'{experiment}/{date}/{extra_folder}/'
        self.fig_locat = self.outpath + "Figures/"
        if not os.path.exists(self.fig_locat):
            os.mkdir(self.fig_locat)
        self.poffeco_inState = poffeco_inState

# possible experiments 
singleTRC = Experiment(seeds=np.r_[2200:2250],date="Feb_24",extra_folder="one-TRC-scaledup",locat='/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/')
oneTRC_varypmut = Experiment(date="Mar_30_23",extra_folder="single-TRC/varypmut")
varTRC_varypmut = Experiment(date="Apr_13_23",extra_folder="var-TRC/varypmut")
MTEenv_varypmut = Experiment(date="Apr_20_23",extra_folder="MTE-env/varypmut")
#MTEenv_varypmut = Experiment(date="Apr_20_23",dates=["Apr_20_23","May_01_23","May_02_23"],extra_folder="MTE-env/varypmut",seeds=np.r_[1000:1300])
MTEenv_varypmut_moreTopt = Experiment(date="May_03_23",dates=["May_03_23"],extra_folder="MTE-env/varypmut",seeds=np.r_[1100:1300])
#MTEenv2_varypmut = Experiment(date="Apr_25_23",extra_folder="MTE-env/varypmut-TRCscale")
MTEenv2_varypmut = Experiment(date="Apr_25_23",dates=["Apr_25_23","Apr_27_23"],extra_folder="MTE-env/varypmut-TRCscale",seeds=np.r_[1000:1100])

# test TRC widths in single TRC (pmut is still varying with T in these experiments)
oneTRC_width13 = Experiment(date="May_03_23",extra_folder="single-TRC/varypmut-width13")
oneTRC_width15 = Experiment(date="May_03_23",extra_folder="single-TRC/varypmut-width15")
varTRC_width13 = Experiment(date="May_05_23",extra_folder="var-TRC/varypmut-width13")
varTRC_width15 = Experiment(date="May_05_23",extra_folder="var-TRC/varypmut-width15")

experiments = [MTEenv_varypmut]

#outpath = '/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/NewTNM/Feb_10/testTNM/'
#outpath = '/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/NewTNM/Feb_11/test_Species_class/'
#outpath = f'/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/{experiment}/{date}/{extra_folder}/'

def get_output(seed,T,experiment,extra_folder):
    outpath = f'/net/venus/kenes/user/cfebvre/out_venus/TTNM_outputs/{experiment}/{extra_folder}/{date}/'

    if T:
        if T < 1:
            T = f"{T:.2f}"
        else: T = f"{int(T)}"
    else: T = "no"

    all_spcs = np.load(outpath+f'species_objects_seed{seed}_{T}K.npy',allow_pickle=True)
    modelrun = np.load(outpath+f"modelrun_seed{seed}_{T}K.npy",allow_pickle=True)

    count = 0
    list1 = []
    for f in os.listdir(outpath):
        if f.startswith(f"species_objects_seed{seed}_{T}K"):
            count += 1
            list1.append(f)
        elif f.startswith(f"modelrun_seed{seed}_{T}K"):
            list1.append(f)
    if count > 1:
        order1,order2 = [],[]
        filelist_t1,filelist_t2 = [],[]
        filelist_species = []
        filelist_states = []
        for f in list1: # os.listdir(outpath):
            if "gen" in f:
                #print(f)
                idx1 = f.index('seed') + len(f"seed{seed}_{T}K_")
                idx2 = f.index('gen')
                finalgen = int(f[idx1:idx2])
                if f.startswith("species_obj"):
                    order1.append(finalgen)
                    filelist_t1.append(f)
                elif f.startswith("model"):
                    order2.append(finalgen)
                    filelist_t2.append(f)
        order_sorted = np.sort(order1)
        indices = []
        for o in order_sorted:
            indices.append(order1.index(o))
        for i in indices:
            filelist_species.append(filelist_t1[i])
        order_sorted = np.sort(order2)
        indices = []
        for o in order_sorted:
            indices.append(order2.index(o))
        for i in indices:
            filelist_states.append(filelist_t2[i])

    if count > 1:
        # need to parse together species from each file
        all_spcs = list(all_spcs) # convert from np.array
        
        # get list of species IDs in all_spcs
        IDs = []
        for spc in all_spcs:
            IDs.append(spc.ID)
        # and make new State object
        big_state = State(seed)
        big_state.N_timeseries = np.zeros((maxgens,))
        big_state.D_timeseries = np.zeros((maxgens,))
        big_state.coreN_timeseries = np.zeros((maxgens,))
        big_state.coreD_timeseries = np.zeros((maxgens,))
        big_state.inSS_timeseries = np.zeros((maxgens,))
        big_state.poffeco_t = np.zeros((maxgens,))
        
        # find number gens run in last chunk
        gensrun = len(modelrun[0].N_timeseries)
        # print("Gens run in last chunk: ",gensrun)
        if gensrun > 0:
            big_state.N_timeseries[-gensrun:] = modelrun[0].N_timeseries
            big_state.D_timeseries[-gensrun:] = modelrun[0].D_timeseries
            big_state.coreN_timeseries[-gensrun:] = modelrun[0].coreN_timeseries
            big_state.coreD_timeseries[-gensrun:] = modelrun[0].coreD_timeseries
            big_state.inSS_timeseries[-gensrun:] = modelrun[0].inSS_timeseries
            if len(modelrun[0].poffeco_t) > 0:
                big_state.poffeco_t[-len(modelrun[0].poffeco_t):] = modelrun[0].poffeco_t

        for f in filelist_species[-1::-1]: # os.listdir(outpath):
            #if f.startswith(f"species_objects_seed{seed}"):
            these_spcs = np.load(outpath+f,allow_pickle=True)
            for spc in these_spcs:
                if spc.ID in IDs:
                    # print("species ID pre-existing: ",spc.ID)
                    # find all_spc species
                    found = 0
                    for spc2 in all_spcs:
                        if spc2.ID == spc.ID:
                            spc2.times_alive = spc.times_alive + spc2.times_alive
                            spc2.populations = spc.populations + spc2.populations
                            spc2.is_core = spc.is_core + spc2.is_core
                            spc2.f_timeseries = spc.f_timeseries + spc2.f_timeseries
                            found = 1
                            break
                    if not found: print("Error, species not found")
                else:
                    IDs.append(spc.ID)
                    all_spcs.append(spc)
                    #all_spcs.append(Species(spc.ID,spc.pop,spc.Topt))
                    #all_spcs[-1].times_alive = spc.times_alive
                    #all_spcs[-1].populations = spc.populations
                    #all_spcs[-1].is_core = spc.is_core
                        #all_spcs[-1].f_timeseries = spc.f_timeseries

        for f in filelist_states: #elif f.startswith(f"modelrun_seed{seed}"):
            if 'gen' in f: # already added output from this file
                m_list = np.load(outpath+f,allow_pickle=True)
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

def get_ecosystem_stats(spc):
    # poff_core not tested--only partially adopted from other code section!!!
    times_in_core = []
    poff_core = []
    if len(spc.times_alive) > 0:
        poff_i = 1/(1+np.exp(-np.array(spc.f_timeseries)))
        # if species spends more than 10 gens in core
        if sum(spc.is_core) > 10:
            times_in_core = np.array(spc.times_alive)[np.where(np.array(spc.is_core)>0)]
            poff_core = []
            for t in range(len(spc.f_timeseries)):
                if spc.times_alive[t] in times_in_core:
                    poff_core.append(poff_i[t])
    return times_in_core, poff_core

if plot_multiple:
    for this_exp in experiments:
        seeds = this_exp.seeds
        T_range = this_exp.T_range
        dates = this_exp.dates
        date = this_exp.date
        experiment = this_exp.experiment
        extra_folder = this_exp.extra_folder
        L,C,theta,mu,pmut = this_exp.L, this_exp.C, this_exp.theta, this_exp.mu, this_exp.pmut
        locat = this_exp.locat
        write_freq = this_exp.write_freq
        outpath = this_exp.outpath
        if len(dates) > 1:
            outpath_list = []
            for d in dates:
                if locat.startswith("/net/"):
                    outpath_list.append(locat + f'{experiment}{extra_folder}/{d}/')
                else: outpath_list.append(locat+f'{experiment}/{d}/{extra_folder}/')
        else: outpath_list = [outpath]
        poffeco_inState = this_exp.poffeco_inState
        fig_locat = this_exp.fig_locat
        bigfig, bigax = plt.subplots() #len(T_range[::2]),figsize=(4,8))
        bigax.set_xlabel("Time (gen)")
        bigax.set_ylabel(r"Range of core growth$_\mathrm{max,i}$")
        range_maxgrowth_by_T = []
        cmap = plt.get_cmap("inferno")
        colors = cmap(np.linspace(0,.9,len(T_range)))
        i = -1
        for T in T_range:
            i += 1
            range_maxgrowth_this_T = []
            seeds_to_plot = seeds.copy()
            if len(seeds) > num_cols:
                seeds_to_plot = seeds[:num_cols]
            fig,ax = plt.subplots(4,len(seeds_to_plot),figsize=(8,5),sharex=True,sharey='row')
            fig.suptitle(f'T: {T}') 
            col = -1
            for seed in seeds: #_to_plot:
                col += 1
                modelrun, all_spcs = get_output(seed,T,experiment,extra_folder)
                if seed in seeds_to_plot:
                    ax[0,col].plot(modelrun[0].N_timeseries)
                    ax[1,col].plot(modelrun[0].D_timeseries,label="tot.")
                    ax[1,col].plot(modelrun[0].coreD_timeseries,label="core")
                pdeathT = Teq.pdeath(float(T))
                maxgrowth_matrix = []
                max_maxgrowth_this_seed = np.zeros((maxgens+1,))
                min_maxgrowth_this_seed = 500*np.ones((maxgens+1,))
                for spc in all_spcs:
                    times_in_core, poff_core = get_ecosystem_stats(spc)
                    times_in_core = np.array(times_in_core)
                    # find max reproduction rate of this species at this timperature
                    poffiT = Teq.Mscale*Teq.poff_Amarasekare(float(T),spc.Topt)
                    # max growthrate of this species at this T is maxbirth_i(T) - death(T)
                    maxgrowth = poffiT - pdeathT
                    
                    # plot some examples of single runs
                    if seed in seeds_to_plot: 
                        # plot maxgrowth of this species while it's in core (if it ever is)
                        ax[2,col].plot(times_in_core,maxgrowth*np.ones(len(times_in_core)))
                        # plot rscale(T)*poff(i) - pdeath(T) for this species while it's in core
                        ax[3,col].plot(times_in_core,maxgrowth*np.array(poff_core)-pdeathT)
                    if len(times_in_core) > 1:
                        idces = np.array(maxgrowth > max_maxgrowth_this_seed[times_in_core])
                        time_idces = times_in_core[idces]
                        max_maxgrowth_this_seed[time_idces] = maxgrowth
                        idces = np.array(maxgrowth < min_maxgrowth_this_seed[times_in_core])
                        time_idces = times_in_core[idces]
                        min_maxgrowth_this_seed[time_idces] = maxgrowth
                    #for t in times_in_core:
                    #    if maxgrowth > max_maxgrowth_this_seed[t]: max_maxgrowth_this_seed[t] = maxgrowth
                    #    elif min_maxgrowth_this_seed[t] == np.nan:
                    #        min_maxgrowth_this_seed[t] = maxgrowth
                    #    elif maxgrowth < min_maxgrowth_this_seed[t]: 
                    #        min_maxgrowth_this_seed[t] = maxgrowth
                min_maxgrowth_this_seed[min_maxgrowth_this_seed==500] = np.nan
                range_maxgrowth_this_seed = max_maxgrowth_this_seed-min_maxgrowth_this_seed
                # plot range between min and max maxgrowth
                #if seed in seeds_to_plot: 
                #    ax[3,col].plot(range_maxgrowth_this_seed)

                #if T%1 == 0:
                #    bigax.plot(range_maxgrowth_this_seed,color=colors[i],alpha=1,label=f"T:{T}")
                range_maxgrowth_this_T.append(range_maxgrowth_this_seed)

            av_range_this_T = np.nanmean(range_maxgrowth_this_T,axis=0)
            print("shape of av_range_this_T: ",np.shape(av_range_this_T))
            range_maxgrowth_by_T.append(av_range_this_T)

            for a in ax[3,:]:
                a.set_xlabel("Time (gen)")
                a.set_ylim(-1,1)
            ax[0,0].set_ylabel("Abundance, N")
            ax[1,0].set_ylabel("Diversity")
            ax[2,0].set_ylabel("Core max growth")
            ax[2,0].set_ylim(-.5,1)
            #ax[3,0].set_ylabel("Max. growth")
            #ax[3,0].set_ylabel("Core range max growth")
            ax[3,0].set_ylabel("Core growth") #poff(T)*poff(i) - pdeath(T)")
            ax[1,-1].legend()
            ax[3,-1].legend()

        """
        # To smooth the signal artifically:
        from scipy import signal
        y=signal.savgol_filter(y,
                       53, # window size used for filtering
                       3), # order of fitted polynomial
        """
    
        #bigax.plot(range_maxgrowth_by_T[:],color=colors)
        i = len(T_range)
        for av_range in range_maxgrowth_by_T[-1::-1]:
            i -= 1
            bigax.plot(av_range,color=colors[i],alpha=.5,label=f"T:{T_range[i]}")
        bigax.legend()
        bigax.set_xlabel("Time (gen)")
        bigax.set_ylabel("Avg. range in core maxgrowth")

else: 
    modelrun, all_spcs = get_output(seed,T,experiment,extra_folder)
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
    print("Total repeats: ",total_repeats)
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
    f2,a2 = plt.subplots()
    f3,a3 = plt.subplots() # core Topt
    print(f"seed {seed}, T {T}, core species: ")
    for spc in all_spcs:
        if sum(spc.is_core) > 10:
            times_in_core = np.array(spc.times_alive)[np.where(np.array(spc.is_core)>0)]
            f_in_core = []
            Topt_in_core = []
            for i in range(len(spc.f_timeseries)):
                if spc.times_alive[i] in times_in_core:
                    f_in_core.append(spc.f_timeseries[i])
                    Topt_in_core.append(spc.Topt)
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
            a3.plot(times_in_core,Topt_in_core)

            # calculate p_off,i(T) - pdeath(T) for this species
            # maxgrowth = Teq.Mscale*Teq.poff_Amarasekare(float(T),spc.Topt)* - Teq.pdeath(float(T))
            print("Topt: ",spc.Topt)
            poffiT = Teq.Mscale*Teq.poff_Amarasekare(float(T),spc.Topt)
            pdeathT = Teq.pdeath(float(T))
            maxgrowth = poffiT - pdeathT
            print("poffiT, pdeathT : ",poffiT,pdeathT)
            print("max growth of this species: ",maxgrowth)
            print("time spent in core: ",len(times_in_core))
            a2.plot(times_in_core,maxgrowth*np.ones(len(times_in_core)),"--")
    ax[1].set_xlabel("Time (gen)")
    ax[0].set_ylabel(r"$f_i$(t) of core species")
    ax[1].set_ylabel(r"$\sum_j(J_{ij} n_j)$ of core species")
    a2.set_xlabel("Time (gen)")
    a2.set_ylabel("Max growth of core species")
    a2.set_ylim(0,1)
    a3.set_xlabel("Time (gen)")
    a3.set_ylabel("Topt of core species")
    a3.set_ylim(T_range[0]-1,T_range[-1]+1)

    # PLOT: poff,eco and poff,core
    rscale = Teq.poff_T(int(T))
    print(f"rscale at {T}K: ",rscale)
    pdeath = Teq.pdeath(int(T))
    print(f"pdeath at {T}K: ",pdeath)

    fig,ax = plt.subplots(2,sharex=True)
    if poffeco_inState:
        for m in modelrun:
            poffeco = m.poffeco_t
            print("poffeco: ",poffeco)
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


    if "MTE" in extra_folder or "var" in extra_folder:
        fig,ax = plt.subplots()
        ax.set_ylim(T_range[0]-1,T_range[-1]+1)
        for spc in all_spcs:
            if len(spc.times_alive) > 0:
                ax.plot(spc.times_alive,spc.Topt*np.ones(len(spc.times_alive)))
        ax.set_xlabel("Time (gen)")
        ax.set_ylabel(r"$T_\mathrm{opt,i}$ (K)")

plt.show()
