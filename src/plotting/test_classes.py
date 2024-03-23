import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os

# runtime parameters
seed = 101
maxgens = 1000
experiment = "NewTNM/"
extra_folder = "test/" # include final slash

# constants
pmut = .02
l_genome = 20
pdeath = .2
C = 200
theta = .2 # fraction of individuals that interact
Ninit = 10
mu = .1
#poff = .5
    
# functions
def poff(spci,live_IDs,live_pops,J1,J2,J3): #rmax=.5,mu=.02):
    #return rmax - mu*N
    return 1/(1+np.exp(-fi(spci,live_IDs,live_pops,J1,J2,J3)))

def get_bin(ID,L=l_genome):
    ID = int(ID)
    binstr = bin(ID)[2:]
    numzeros = L - len(binstr)
    binstr = numzeros*'0'+binstr
    return binstr

def check_if_core(spc_pop,live_pops):
    if len(live_pops) < 1:
        return False
    maxpop = max(live_pops)
    if spc_pop >= .05*maxpop:
        return True
    else: return False

def check_if_SS(new_core,old_core):
    overlaps = 0
    other_species = 0
    for ID in new_core:
        if ID in new_core: overlaps+=1
        else: other_species += 1
    if overlaps > other_species and overlaps > 1: return True
    else: return False


# TNM "fitness"
def fi(cur_ID,live_IDs,live_pops,J1,J2,J3,verbose=False):
    """Calculate the impact of all other species on the current one 
    and density dependence (mu N). Mathematically, f_i = sum_j(J_{ij} n_j) - mu N .

    IN:
        cur_ID (int), J1-3 (np.arrays), live_IDs (list), live_pops (list), verbose=False
    OUT: 
        f_i
    """
    if len(live_IDs) > 1:
        if J1[0] != 0: 
            #print("Why is J[0] not zero?")
            J1[0] = 0
        interactions = np.array(live_IDs)^cur_ID # np array of bitwise interaction of ID's
        val = sum(J1[interactions]*J2[interactions]*J3[live_IDs]*live_pops) #- const.mu*sum(populations)
        if verbose: # tested this all matches C++ exactly May 16, 2022
            print("calc_HI is ",val)
            i = species.index(current_spc)
            v2 = val*live_pops[i]/sum(live_pops)
            print("calc_HI*Nj/N is ",v2) # this sum_j(Jij*nj)
            my_fi = val/sum(live_pops) - mu*sum(live_pops)
            poff2 = 1/(1+np.exp(-my_fi))
            print("poff2 is: ",poff2)
        return val/sum(live_pops) - mu*sum(live_pops)
    # if no other species exist, fi is zero
    else: return 0

def fi_old(spci,live_IDs,live_pops):
    f = 0
    for j in range(len(live_IDs)):
        if live_IDs[j] != spci.ID:
            f += live_pops[j]*spci.responses[live_IDs[j]]
    return f
        

def choose_indiv(live_IDs,live_pops,all_spcs): 
    spc_ID = np.random.choice(live_IDs,p=np.array(live_pops)/sum(live_pops))
    for spc in all_spcs:
        if spc_ID == spc.ID:
            return spc
    print("error in choose: ID never found match")

def reproduce(spc_ID):
    mom_bin = list(get_bin(spc_ID)) # copy mother's genome
    baby_bin = mom_bin
    for gene in range(l_genome): # give each gene a chance to flip
        if np.random.random() < pmut:
            baby_bin[gene] = str(int(not mom_bin[gene]))
    return int(('').join(baby_bin),2) # convert binary number to integer


def get_core(max_pop,all_spcs):
    core = []
    for spc in all_spcs:
        if spc.pop >= .05*max_pop:
            core.append(spc.ID)
    return core

def init(L=l_genome,theta=theta,Ni=Ninit,C=C):
    # interactions
    J1 = np.random.random(size=2**L) < theta
    J2 = np.random.normal(size=2**L) #
    J3 = np.random.normal(size=2**L)*C

    # species
    ID = np.random.randint(1,2**L)
    all_spcs = [] # all species ever encountered
    all_spcs.append(Species(ID,Ni))
    live_IDs = [all_spcs[0].ID]
    live_pops = [Ni]
    
    # update timeseries
    all_spcs[0].alive(0,Ni,live_pops,all_spcs,live_IDs,J1,J2,J3)

    return all_spcs,live_IDs,live_pops,J1,J2,J3

# classes
class Species:
    def __init__(self,ID,pop,Topt=False):
        self.ID = int(ID)
        self.pop = pop
        self.bin = get_bin(ID)
        # timeseries
        self.times_alive = []
        self.populations = []
        self.is_core = []
        self.f_timeseries = []
        self.responses = {}
        self.Topt = Topt

    def __str__(self):
        return f"species {self.ID}, Ni = {self.pop}"

    def alive(self,t,pop,live_pops,all_spcs,live_IDs,J1,J2,J3):
        self.times_alive.append(t) # time at which current species is alive
        self.populations.append(pop) # population of species now
        self.is_core.append(check_if_core(pop,live_pops))
        self.f_timeseries.append(fi(self.ID,live_IDs,live_pops,J1,J2,J3)) # fitness of current species

    def new_impact(self,spcnew): # spc is a class object
        # new spcnew affects self with new random number
        self.responses[spcnew.ID] = C*np.random.uniform(-1,1)

    def new_species(self,all_spcs):
        # self is new species; it gets impacted by all other species
        for spcj in all_spcs:
            # all other species affect this new species
            self.responses[spcj.ID] = C*np.random.uniform(-1,1)


class State:
    def __init__(self,seed):
        self.seed = seed
        self.species_now = []
        self.pops_now = []
        self.N_timeseries = []
        self.D_timeseries = []
        self.coreN_timeseries = []
        self.coreD_timeseries = []
        self.inSS_timeseries = []
        self.poffeco_t = [] # timeseries of 1/N*sum(poff_i*N_i)
        self.core = [] # two biggest species from last timestep

    def __str__(self):
        return f"Model state (seed {self.seed}): timeseries"

    def timestep(self,live_pops,all_spcs,biggest_spc,poffeco=0):
        self.N_timeseries.append(sum(live_pops))
        self.D_timeseries.append(len(live_pops))
        core_pops = [p for p in live_pops if p > max(live_pops)*.05]
        self.coreN_timeseries.append(sum(core_pops))
        self.coreD_timeseries.append(len(core_pops))
        if poffeco:
            self.poffeco_t.append(poffeco)
        # new_core = get_core(max(live_pops),all_spcs)
        if biggest_spc[0] in self.core and len(biggest_spc) > 1:
            if biggest_spc[1] in self.core:
        # if new_core == self.core: 
        # if check_if_SS(new_core,self.core): # == self.core: 
                self.inSS_timeseries.append(True)
        else: self.inSS_timeseries.append(False)
        self.core = biggest_spc #new_core
        

# initialize model with a new species
all_spcs, live_IDs, live_pops, J1, J2, J3 = init()

t = 0

# create state
modelrun = State(seed)

starttime = time.time()

if __name__ == "__main__":

    # set output path
    out_path = '/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/'+experiment
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    today = datetime.date.today().strftime("%b_%d_%Y")
    if not os.path.exists(out_path+today):
        os.mkdir(out_path+today)
    out_path = out_path+today+'/'+extra_folder
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    np.random.seed(seed)
    for t in range(maxgens):
        # cycle randomly through 20 new individuals
        #for random_spc in np.random.randint(0,20,size=20):
        for _ in range(sum(live_pops)):
            # choose random individual
            spc = choose_indiv(live_IDs,live_pops,all_spcs)
            # see if it reproduces
            if np.random.random() < poff(spc.ID,live_IDs,live_pops,J1,J2,J3): #poff(sum(all_pops)): 
                # reproduce it
                baby = reproduce(spc.ID) # return baby ID
                if baby not in [spc.ID for spc in all_spcs]:
                    #print(random_spc)
                    #print([spc.ID for spc in all_spcs])
                    all_spcs.append(Species(baby,1))
                    # this species responds to all other existing species
                    # all_spcs[-1].new_species(all_spcs)
                    ## allow new species to impact other species
                    #for spc in all_spcs:
                    #    spc.new_impact(all_spcs[-1])

                    live_IDs.append(baby)
                    live_pops.append(1)
                else:
                    for cur_spc in all_spcs:
                        if baby == cur_spc.ID:
                            cur_spc.pop += 1
                            if cur_spc.ID in live_IDs:
                                idx = live_IDs.index(cur_spc.ID)
                                live_pops[idx] += 1
                                # this species already has list of responses
                            # this species is new to this ecosystem
                            else:
                                live_pops.append(1)
                                live_IDs.append(cur_spc.ID)
                            # update species responses 
                            # *** THERE MAY BE A MORE EFFICIENT WAY TO CHECK WHETHER BABY ALREADY HAS RESPONSES TO OTHER SPECIES.... ***
                            # for other in all_spcs:
                            #    if other.ID != baby:
                            #         if baby not in other.responses.keys():
                            #             other.new_impact(cur_spc)
                            #         if other.ID not in cur_spc.responses.keys():
                            #             cur_spc.new_impact(other)
            # death events
            rand_dead = choose_indiv(live_IDs,live_pops,all_spcs) #np.random.randint(0,20)
            if np.random.random() < pdeath:
                if rand_dead.pop > 0:
                    # update population of class object
                    rand_dead.pop -= 1
                    idx = live_IDs.index(rand_dead.ID)
                    live_pops[idx] -= 1
                    # if this eliminated the species, remove it from species ID list
                    if live_pops[idx] <= 0:
                        live_IDs.pop(idx)
                        live_pops.pop(idx)
        # after timestep, update all species populations etc
        for spc in all_spcs:
            spc.alive(t,spc.pop,live_pops,all_spcs,live_IDs,J1,J2,J3)
        biggest_idx = live_pops.index(max(live_pops))
        biggest_spc = [live_IDs[biggest_idx]]
        if len(live_pops) > 1:
            if biggest_idx < len(live_pops):
                second_idx = live_pops.index(max(np.array(live_pops)[np.r_[:biggest_idx,biggest_idx+1:]]))
            else: second_idx = live_pops.index(max(live_pops[:biggest_idx]))
            biggest_spc.append(live_IDs[second_idx])

        modelrun.timestep(live_pops,all_spcs,biggest_spc)


    print("Run complete")
    print("Time required: ",(time.time()-starttime)/60," minutes")

    # save outputs
    np.save(out_path+'species_object_list_seed'+str(seed)+today,all_spcs)
    np.save(out_path+'modelrun_seed'+str(seed)+today,[modelrun])

    # fig 1
    fig,ax = plt.subplots(2,2,sharex=True)
    ax[0,0].plot(modelrun.N_timeseries)
    ax[0,0].set_ylabel("Abundance")
    ax[0,1].plot(modelrun.coreN_timeseries)
    ax[0,1].set_ylabel("Core abundance")

    ax[1,0].plot(modelrun.D_timeseries)
    ax[1,0].set_ylabel("Diversity")
    ax[1,1].plot(modelrun.coreD_timeseries)
    ax[1,1].set_ylabel("Core diversity")
    ax[1,0].set_xlabel("Time")
    ax[1,1].set_xlabel("Time")

    plt.tight_layout()

    # fig 2
    fig,ax = plt.subplots()
    for spc in all_spcs:
        ax.plot(spc.times_alive,spc.populations)
    ax2 = plt.twinx(ax)
    ax2.plot(modelrun.inSS_timeseries,"--k",label="in SS")
    ax2.legend()
    ax.set_title("All species")
    ax.set_ylabel("Species populations")
    ax.set_xlabel("Time")

    plt.show()
        






