import matplotlib.pyplot as plt
#import proplot as pplt
import numpy as np
import geoTNM as g
import os

# filename = "/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/Test/VariableT/pypy_seed101Jul_04_298K_TNM_L20.dat"
seed = 100 # first seed
num_seeds = 1 # number of seeds in experiment
day = 'Aug_05'
T = 298 # if one only
temperatures = np.r_[274:320:3] # if multiple temperatures
plot_1seed_only = False
maxgens = 100
experiment = "SteadyT"
sub_folder = "MTE_TPC_combo/" # "Variable_Tresponse/" #"Test_variable_response/"
other_dates = [day[:5] + i for i in list(np.array(np.r_[10:11],dtype=str))]
base_path = "/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/" 
# filename = "/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/Test/VariableT/pypy_seed107Jul_07_298K_TNM_L20.dat" # SteadyT/Jul_07/Test_variable_response/pypy_seed100Jul_07_298K_TNM_L20.dat"
filename = base_path+experiment+'/'+day+'/'+sub_folder+f"pypy_seed{seed}{day}_{T}K_TNM_L20.dat"

# INITIALIZE
#rng, Jran1, Jran2, Jran3, encountered, species, populations, Tresponse = g.tangled_nature.init(100,True)

# RUN 1000 GEN  
# rng, Jran1,Jran2,Jran3,encountered,species,populations = g.tangled_nature.main(rng, Jran1, Jran2, Jran3, encountered, species, populations, T=False, dt=1000*3600*24*365, seed=100, experiment="BasicTNM", output_path="/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/Test/", Tresponse=False, verbose=False) 
# OR SKIP THAT AND USE THIS
#species = [432088, 439816, 432064, 165832, 448200, 431818, 300620, 427624, 923208, 448456, 407112, 432074, 439880, 398856, 493512, 399944, 432008, 956232, 301002, 302664, 169544, 431708, 167880, 432073, 434120, 972744, 432076, 427592, 431817, 431560, 301000, 497608, 431048, 431689, 956360, 432104, 431692, 433736, 448072, 427976, 399304, 431704, 300616, 440264, 398920, 169928, 431680, 497224, 431624, 955976, 431690, 431176, 431816, 431944, 430664, 431720, 432072, 431688]
#populations = [1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 1, 2, 3, 1, 2, 2, 4, 1, 4, 6, 3, 5, 3, 3, 5, 2, 3, 6, 2, 3, 4, 4, 3, 4, 3, 2, 4, 16, 11, 4, 8, 200, 422]

# Calculate poff for all species
def calc_poff_list(T):
    poff_control = []
    poff_T = []
    poff_T_variable = []
    for i in species:
        f = g.life_events.fi(i,Jran1,Jran2,Jran3,species,populations)
        poff_control.append(g.life_events.poff(i,Jran1,Jran2,Jran3,species,populations))
        poff_T.append(g.temperature_effects.poff_total(f,T))
        poff_T_variable.append(g.temperature_effects.poff_total(f,T,Tresponse[i]))
    return poff_control,poff_T,poff_T_variable

def calc_poff_from_life(T):
    poff_control = []
    poff_T = []
    poff_T_variable = []
    for i in species:
        f = g.life_events.fi(i,Jran1,Jran2,Jran3,species,populations)
        poff_control.append(g.life_events.poff(i,Jran1,Jran2,Jran3,species,populations))
        poff_T.append(g.life_events.poff(i,Jran1,Jran2,Jran3,species,populations,T))
        poff_T_variable.append(g.life_events.poff(i,Jran1,Jran2,Jran3,species,populations,T,Tresponse))
    return poff_control,poff_T,poff_T_variable

def violin_by_T():
    """
    Plot poff for the control case, with temperature the same for all
    species, and with variable temperature responses
    """
    for T in np.r_[285:320:5]:
        poff,poff_T,poff_T_variable = calc_poff_list(T)
        poff2,poff_T2,poff_T_variable2 = calc_poff_from_life(T)
        print("Methods are the same?")
        print(poff==poff2, poff_T==poff_T2, poff_T_variable==poff_T_variable2)
        plt.violinplot([poff,poff_T,poff_T_variable,poff_T_variable2],positions=[T-1,T,T+1,T+2],widths=[3,3,3,3]) #,showextrema=False)
        plt.xlabel("Temperature,T (K)")
        plt.ylabel("Reproduction prob.s, poff")
        plt.title("Distributions of reprod. prob.s w/ different temp. effects")
    plt.show()

def plot_timeseries(filename):
    # The columns in Arthur data are: 
    #['gen','popu','div', 'enc', 'core popu', 'core div', 'avg Topt', 'F', 'avg Twidth']

    popu_time = []
    div_time = []
    enc_time = []
    core_popu_time = []
    core_div_time = []
    speciation = []
    Topt_time = []
    F_time = []
    Twidth_time = []
    try:
        with open(filename,'r') as f:
            for line in f:
                parts = line.split(" ")
                popu_time.append(int(parts[1]))
                div_time.append(int(parts[2]))
                enc_time.append(int(parts[3]))
                if len(enc_time) > 1:
                    speciation.append(enc_time[-1] - enc_time[-2])
                core_popu_time.append(int(parts[4]))
                core_div_time.append(int(parts[5]))
                Topt_time.append(float(parts[-3]))
                F_time.append(float(parts[-2]))
                Twidth_time.append(float(parts[-1][:-1])) # remove /n from end
    except: return 0,0,0,0,0,0,0,0,0
    if not plot_1seed_only:
        return popu_time,div_time,enc_time,core_popu_time,core_div_time,speciation,Topt_time,F_time,Twidth_time
    else:
        fig,ax = plt.subplots()
        ax.set_title(f"Population and Core Diversity, & New species formation over Time\nseed {seed}, {day}")
        ax.plot(popu_time,label="population,N (indiv)")
        ax2 = plt.twinx(ax)
        ax2.plot(core_div_time,"r--",label="diversity (species)")
        ax.set_ylabel("Population,N (indiv.)")
        ax.set_xlabel("Time,t (gen.)")
        ax2.set_ylabel("Core diversity (species)")
        

        fig,ax = plt.subplots()
        ax.set_title(f"Population, Diversity, & New species formation over Time\nseed {seed}, {day}")
        ax.plot(popu_time,label="population,N (indiv)")
        ax.plot(div_time,label="diversity (species)")
        ax2 = plt.twinx(ax)
        ax2.plot(speciation,color="r",alpha=.5,label="speciation")
        ax2.set_ylabel("New species formation (species)",color='r')
        ax.set_ylabel('Count')
        ax.legend()
        ax.set_xlabel("Time,t (generations)")
        plt.tight_layout()

        fig,ax = plt.subplots()
        ax.set_title(f"Population and Fitness over Time\nseed {seed}, {day}")
        ax.plot(popu_time,label="populations")
        ax2 = plt.twinx(ax)
        ax2.plot(F_time,color="r",alpha=.5,label="total fitness")
        ax2.set_ylabel("Total ecosystem fitness, F",color="red")
        ax.set_ylabel("Population,N (indiv)",color="b")
        ax.set_xlabel("Time,t (generations)")
        plt.tight_layout()

        fig,ax = plt.subplots()
        ax.set_title(f"Average Species Temperature Optimum over Time\nseed {seed}, {day}")
        ax.plot(Topt_time,color="r",alpha=.5,label="average Topt")
        ax.set_ylabel("Average Topt,<Topt_i> (K)",color="red")
        ax2 = plt.twinx(ax)
        ax2.plot(popu_time,label="populations",alpha=.5)
        #ax.legend()
        ax.set_xlabel("Time,t (generations)")
        ax2.set_ylabel("Population,N (indiv)")
        plt.tight_layout()

        fig,ax = plt.subplots()
        ax.set_title(f"Average Species T-Curve Width over Time\nseed {seed}, {day}")
        ax.plot(Twidth_time,color="r",alpha=.5,label="average Twidth")
        ax.set_ylabel("Average T-width,<Twidth_i> (K)",color="red")
        ax.legend()
        ax.set_xlabel("Time,t (generations)")
        ax2 = plt.twinx(ax)
        ax2.plot(popu_time,label="populations")
        ax2.set_ylabel("Population,N (indiv)")
        plt.tight_layout()

        plt.show()

def plot_allseeds_1T(T,plot_this_T=False):
    popu_all_by_t = np.zeros((maxgens,num_seeds))
    Topt_all_by_t = np.zeros((maxgens,num_seeds))
    Twidth_all_by_t = np.zeros((maxgens,num_seeds))
    i = -1
    count = 0
    for s in np.r_[seed:seed+num_seeds]:
        i += 1
        filename = base_path+experiment+'/'+day+'/'+sub_folder+f"pypy_seed{s}{day}_{T}K_TNM_L20.dat" 
        if not os.path.exists(filename):
            for oday in other_dates:        
                filename = base_path+experiment+'/'+day+'/'+sub_folder+f"pypy_seed{s}{oday}_{T}K_TNM_L20.dat" 
                if os.path.exists(filename):
                    break

        if not os.path.exists(filename):
            print("File not found: ",filename)
        popu_time,div_time,enc_time,core_popu_time,core_div_time,speciation,Topt_time,F_time,Twidth_time = plot_timeseries(filename)

        if popu_time == 0: 
            print("file skipped")
            continue
        else: count += 1

        popu_all_by_t[:len(popu_time),i] = popu_time
        Topt_all_by_t[:len(Topt_time),i] = Topt_time
        Twidth_all_by_t[:len(Twidth_time),i] = Twidth_time

    # average Topt and popu over time
    popu_avg = []
    Topt_avg = []
    # cycle through all times and take average of all nonzeros
    for t in range(maxgens):
        popus_now = np.array(popu_all_by_t[t,:])
        non0s = popus_now.nonzero()
        popus_living = popus_now[non0s]
        popu_avg.append(np.average(popus_living))

        Topts_now = np.array(Topt_all_by_t[t,:])
        Topts_living = Topts_now[non0s]
        Topt_avg.append(np.average(Topts_living))
    
    #popu_avg = np.average(popu_all_by_t,1)
    #Topt_avg = np.average(Topt_all_by_t,1)
    #Twidth_avg = np.average(Twidth_all_by_t,1)

    if not plot_this_T:
        return popu_avg, Topt_avg
    fig,ax = plt.subplots() 
    ax.plot(popu_avg)
    ax2 = plt.twinx(ax)
    ax2.plot(Topt_avg,'r')
    ax2.plot(np.ones(len(Topt_avg))*T,"g--")
    fig.text(.8,.5,'T_env',color='g')
    ax2.set_ylabel("Average Topt,<Topt> (K)",color='r')
    ax.set_ylabel("Average population,<N> (indiv.)",color='b')
    ax.set_xlabel("Time,t (gen.)")
    ax.set_title(f"Population and species Topt over time at T={T}K, avg of {count} seeds")

    '''fig,ax = plt.subplots()
    ax.plot(popu_avg)
    ax2 = plt.twinx(ax)
    ax2.plot(Twidth_avg,'r')
    #ax2.plot(np.ones(len(Twidth_avg))*T,"g--")
    #fig.text(.8,.5,'T_env',color='g')
    ax2.set_ylabel("Average Twidth,<Twidth> (K)",color='r')
    ax.set_ylabel("Average population,<N> (indiv.)",color='b')
    ax.set_xlabel("Time,t (gen.)")
    ax.set_title(f"Population and species Twidth over time at T={T}K, avg of {count} seeds")
'''
    plt.show()
    return popu_avg, Topt_avg

if plot_1seed_only:
    plot_timeseries(filename)
else:
    popu_avg_all = np.zeros((maxgens,len(temperatures)))
    Topt_avg_all = np.zeros((maxgens,len(temperatures)))
    i = -1
    for T in temperatures:
        i += 1
        popu_avg, Topt_avg = plot_allseeds_1T(T)
        popu_avg_all[:len(popu_avg),i] = popu_avg
        Topt_avg_all[:len(Topt_avg),i] = Topt_avg
    
    i = -1
    fig,ax = plt.subplots(2,figsize=(10,6),dpi=80)
    cmap = plt.get_cmap('magma')
    colors = cmap(np.linspace(1,.1,len(temperatures)))
    for T in temperatures:
        i += 1
        ax[0].plot(popu_avg_all[:,i],label=f"T={T}",color=colors[i])
        ax[1].plot(Topt_avg_all[:,i],label=f"T={T}",color=colors[i])
        ax[1].scatter(maxgens,T,color=colors[i])
        #hs.append(h0)
    #fig.legend(hs,loc='r') #, label='Temperature (K)', frame=False, loc='r')
 
    ax[0].set_ylabel(r"Avg. abundance, $\bar{N}$ (indiv.)")
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_ylabel(r"Avg. $T_{opt}$ (K)")
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[0].legend(bbox_to_anchor=(1,1),loc="upper left") #,mode="expand")

    plt.subplots_adjust(hspace=-1)
    fig.text(0.4,0.015,"Time,t (log gen.)")
    plt.suptitle(r"Evolution of avg. abundance & $T_{opt}$ for"+f" {num_seeds} experiments ({day})")
    
    plt.tight_layout()
    plt.savefig(base_path+experiment+'/'+day+'/'+sub_folder+f"Evolution_avg_pop_Topt_{day}.png") 

    # PLOT 2: 
    # average Topt at each temperature at end of run
    print("Average Topt at each temperature")
    print(Topt_avg_all[-1,:])
    plt.figure()
    T_C = temperatures-273
    plt.plot(T_C,Topt_avg_all[-1,:]-273,label=r"average $T_{opt}$")
    plt.plot(T_C,T_C,"--",label="1:1 ratio")
    plt.legend()
    plt.xlabel("Environment temperature,T ($^\circ$C)")
    plt.ylabel(r"Average species $T_{opt}$ ($^\circ$C)")
    plt.title(r"Thermal depedence of species average $T_{opt}$")


    plt.show()
