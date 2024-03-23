# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:43:53 2022

@author: camil

This is the main routine of the tanlged nature model (TNM).  However, it only contains two functions: initialization, and main().  

FUNCITONS: 
    - init: The initialization creates the things to be passed between the rest of the scripts: species and populations lists, a random number generator, and the J matrix machinery.  
    - main: The main function initializes the run if it hasn't already been done externally, calls the file-naming function, and then starts the loop to go through timesteps and record stats at the end of each generation.  At the end, it calls the function to save the state of the model in npy files.
"""
import time
import numpy as np
import sys 
import os
#from geoTNM 
import MTE_TPC_combo as Tmod, TNM_constants as const, life_events as life, random_generator as ran_gen, file_management as manage
from test_classes import Species, State
from numpy.random import SeedSequence

# classes
#class Species:
#    def __init__(self,ID,pop,Topt=False):
#        self.ID = int(ID)
#        self.pop = pop
#        #self.bin = get_bin(ID)
#        # timeseries
#        self.times_alive = []
#        self.populations = []
#        self.is_core = []
#        self.f_timeseries = []
#        self.responses = {}
#        self.Topt = Topt
#    
#    def __str__(self):
#        return f"species {self.ID}, Ni = {self.pop}"
#
#    def alive(self,t,pop,live_pops,all_spcs,live_IDs,J1,J2,J3):
#        self.times_alive.append(t) # time at which current species is alive
#        self.populations.append(pop) # population of species now
#        self.is_core.append(check_if_core(pop,live_pops))
#        self.f_timeseries.append(fi(self.ID,live_IDs,live_pops,J1,J2,J3)) # fitness of current species

#class State:
#    def __init__(self,seed):
#        self.seed = seed
#        self.species_now = []
#        self.pops_now = []
#        self.N_timeseries = []
#        self.D_timeseries = []
#        self.coreN_timeseries = []
#        self.coreD_timeseries = []
#        self.inSS_timeseries = []
#        self.core = []
#
#    def __str__(self):
#        return f"Model state (seed {self.seed}): timeseries"
#
#    def timestep(self,live_pops,all_spcs,biggest_spc):
#        self.N_timeseries.append(sum(live_pops))
#        self.D_timeseries.append(len(live_pops))
#        core_pops = [p for p in live_pops if p > max(live_pops)*.05]
#        self.coreN_timeseries.append(sum(core_pops))
#        self.coreD_timeseries.append(len(core_pops))
#        if biggest_spc[0] in self.core and len(biggest_spc) > 1:
#            if biggest_spc[1] in self.core:
#        # if new_core == self.core: 
#        # if check_if_SS(new_core,self.core): # == self.core: 
#                self.inSS_timeseries.append(True)
#        else: self.inSS_timeseries.append(False)
#        self.core = biggest_spc #new_core

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

def fi(cur_ID,species,populations,J1,J2,J3,mu=const.mu):
    if len(species) > 1:
        if J1[0] != 0:
            #print("Why is J[0] not zero?")
            J1[0] = 0
        interactions = np.array(species)^cur_ID # np array of bitwise interaction of ID's
        val = sum(J1[interactions]*J2[interactions]*J3[species]*populations) #- const.mu*sum(populations)
        return val/sum(populations) - mu*sum(populations)
    # if no other species exist, fi is zero
    else: return 0


# Initialization: create rng, species, popoulations, Jran1-3
def init(seed,variable_Tresp=False,D_init=const.D_init): # tested May 18, 22 against TNM_all.init
    """
    Initialization.

    IN: 
        seed (int)
    OUT: 
        rng, Jran1-3 (np.arrays), encountered (set), species (list), populations (list) (, Tresponse)
    """
    print("Seed: ",seed)
    encountered = []
    #species = [] # IDs of each species
    #populations = [] # populations of each species
    
    # Use seed sequence for higher qulaity seeds
    ss = SeedSequence(seed)
    if const.options.rng == "default":
        # *** is it necessary to have different generators here?? ***
        generator = np.random.default_rng(ss.spawn(1)[0]) #123*seed+456)
        rng = np.random.default_rng(ss.spawn(1)[0]) # spawn one seed from seed sequence
        # generator = rng
    else: # ***
        print("Still need to test other rng initialization!!")
        rng = ran_gen.rng.seed(ss)
        generator = ran_gen.rng.seed(ss*123+456)
    
    # variable T response
    if variable_Tresp:
        Tref = Tmod.Tref
        width = Tmod.width
        skew = Tmod.skew
        # pre-define different combintations of above three presets
        #Tresponse1 = generator.normal(Tref,10,size=const.N)
        Tresponse1 = generator.uniform(273-10,320+10,size=const.N)
        Tresponse2 = generator.normal(width,1,size=const.N)
        Tresponse3 = generator.normal(skew,.5,size=const.N)
        Tresponse = np.vstack((Tresponse1,Tresponse2,Tresponse3)).transpose()
        # Tresponse[i] = Tref[i], width[i], skew[i]
    
    Jran1 = rng.random(size=const.N) < const.theta
    Jran2 = generator.normal(size=const.N) #
    Jran3 = generator.normal(size=const.N)*const.C
    
    # choose random species
    #species.append(rng.integers(0,const.N))
    species = list(rng.integers(0,const.N,D_init))
    populations = D_init*[int(const.Npop_init/D_init)]
    # make sure initial N = 500
    while sum(populations) < const.Npop_init:
        populations[rng.integers(0,D_init)] += 1
    # add all new species to encountered list
    i = -1
    for ID in species:
        i += 1
        if variable_Tresp: 
            T_arg = Tresponse1[ID] # Topt value of this species
        else: T_arg = False
        encountered.append(Species(ID,populations[i],T_arg))
        encountered[-1].alive(0,populations[0],populations,encountered,species,Jran1,Jran2,Jran3)
        
    if variable_Tresp:
        return rng,Jran1,Jran2,Jran3,encountered,species,populations,Tresponse
    else:
        return rng,Jran1,Jran2,Jran3,encountered,species,populations


def save_files(encountered,modelrun,seed,T=False,output_path='',gen=False):
    print("Output path: ",output_path)
    if T:
        if T < 1:
            T = f"{T:.2f}"
        else:
            T = f"{int(T)}"
    else: T = "no"
    if gen:
        gen = f"_{gen}gen"
    else: gen = ''
    np.save(output_path+'species_objects_seed'+str(seed)+f"_{T}K{gen}",encountered)
    np.save(output_path+'modelrun_seed'+str(seed)+f'_{T}K{gen}',[modelrun])


# %% Main
def main(rng=None,Jran1=None,Jran2=None,Jran3=None,encountered=None,species=None,populations=None,T=False,dt=1,seed=False,experiment=False,output_path=False,Tresponse=False,TRC=False,vary_pmut=False,write_freq=4000,verbose=False):
    """Main TNM routine.  
    Inputs: 
        - rng, Jran1, Jran2, encountered, species, populations: generated by TNM_all.init() (or previous run),
        - T: set temperature or False (for original TNM)
        - dt: how many genrations to run
        - seed: seed
        - experiment: ["step or steady or whatever", T] (requires T in case of step experiment, can be number or string. if driven with probs, T is [poff,pdeath].) 
        (can also set experiment to False to run the thing normally, but I haven't tested that htis works)
        - Tresponse: np.array with 3 columns and 2**N rows corresponding to Topt or Tref of each species (also Twidth and skew, not yet used here).
        - TRC: type of experiment: var-TRC or MTE-env
    
    Other Runtime Options:
        constants.options: attatch geochem, variable T response per species, and SpPops files are possible options
        
    Outputs: 
        - rng,Jran1,Jran2,Jran3,encountered,species,populations
        OR 
        - seven 0's if extinction has occured """
    t0 = time.time() # start time of this experiment
    modelrun = State(seed)
    
    if verbose:
        print("Seed is ",seed)
    
    # name files for outputs
    # could put file naming for save_files here if desired ***
    if experiment[0] in ["LinearT","linear"]:
        pop_file,div_file,restart_file,time_file,other_files = manage.file_naming(seed,experiment,"linear",output_path)
    else:
        pop_file,div_file,restart_file,time_file,other_files = manage.file_naming(seed,experiment,T,output_path)

    # Adjust pkill if T is included
    if T:
        pkill = Tmod.pdeath(T)
    else:
        pkill = const.pkill  
    
    # Initialize if not connecting to geochem (geochem initializes internally)
    if not const.options.attatch_geochem:
        rng,Jran1,Jran2,Jran3,encountered,species,populations = init(seed)
    if None in np.array([rng,Jran1,Jran2,Jran3,encountered,species,populations],dtype=object):
        print("ERROR initializing, must give rng,Jran1,Jran2,Jran3,encountered,species,and populations to main() if attatching geochem")

    # time and generation counters
    t = 0
    tgen = 0
    lgen = sum(populations)/pkill # pkill adjusted according to T (or not) above
    print("TNM: generation length: ",lgen," iterations")
    if const.options.attatch_geochem:
        gens2run = int(dt/(const.year_s)) # convert dt from seconds to years, and run 1 gen per year
    else: gens2run = const.max_gens
    
    # run TNM once per year.
    print(f"TNM: Running {gens2run} generations")
    while tgen<gens2run:#const.max_gens:
        #print("type Tresponse: ",type(Tresponse))
        species,populations,encountered = life.one_timestep(species,populations,encountered,rng,T,Jran1,Jran2,Jran3,Tresponse,TRC,vary_pmut,verbose)

        # if there was a mass extinction, record it in print_stats
        if sum(populations) <= 0: 
            print("MASS EXTINCTION at ",t," iterations and ",tgen," generations")
            print(f"TNM: seed {seed}, T {T} took ",time.time()-t0," seconds for {tgen} generations")
            
            ## update encountered with all species populations etc
            #poffeco=0
            #for spc in encountered:
            #    spc.alive(tgen,spc.pop,populations,encountered,species,Jran1,Jran2,Jran3)
            #    poffeco += 1/(1+np.exp(-spc.f_timeseries[-1]))*spc.pop
            #poffeco = poffeco/sum(populations)
            # update model state
            big_spc = [0]
            modelrun.timestep(populations,encountered,big_spc,poffeco=0)

            save_files(encountered,modelrun,seed,experiment[1],output_path)
            return rng,Jran1,Jran2,Jran3,encountered,[],[]
        
        # add any new species into list of encountered species
        # encountered.update(species)
                
        # increase time counter
        t += 1
        if verbose:
            print("--------------generation: ",t,"-------------")
            print("--------------population: ",sum(populations),"-------------")
        # if enough timesteps have been taken, increase tgen by one and reset t
        if t >= lgen: 
            t = 0
            tgen += 1

            # calculate ecosystem reproduction probability
            # and update encountered with species populations of living species etc
            poffeco = 0
            for spc in encountered:
                update_spc = False
                if spc.pop < 1:
                    if len(spc.populations) > 0:
                        if spc.populations[-1] > 0:
                            update_spc = True
                    else:
                        update_spc = True
                else:
                    update_spc = True
                if update_spc:
                    spc.alive(tgen,spc.pop,populations,encountered,species,Jran1,Jran2,Jran3)
                    # add the reproduction probability of each species
                    poffeco += 1/(1+np.exp(-spc.f_timeseries[-1]))*spc.pop
            if abs(poffeco) > 1e-9:
                poffeco = poffeco/sum(populations)

            # update model state
            # find biggest two species
            big_idx = populations.index(max(populations))
            big_spc = [species[big_idx]]
            # second biggest species
            if len(populations) > 1:
                if big_idx < len(populations):
                    big_idx = populations.index(max(np.array(populations)[np.r_[:big_idx,big_idx+1:]]))
                else: big_idx = populations.index(max(populations[:big_idx]))
                big_spc.append(species[big_idx])
            # update State    
            modelrun.timestep(populations,encountered,big_spc,poffeco=poffeco)

            # write out files every 4000 generations
            if write_freq:
                if tgen/write_freq - np.floor(tgen/write_freq) < 1e-9:
                    save_files(encountered,modelrun,seed,experiment[1],output_path,gen=tgen) # add gen as kwarg to save_files
                    # empty and refill encountered and modelrun 
                    encountered = []
                    for i in range(len(species)):
                        ID = species[i]
                        pop = populations[i]
                        if type(Tresponse) != bool:
                            Topt = Tresponse[ID,0]
                        else: Topt = False
                        encountered.append(Species(ID,pop,Topt)) # VERIFY ***
                    modelrun = State(seed)
        
            # if attatch_geochem:
            #     # more generations per year in better climate
            #     lgen = sum(populations)/const.pkill*T_eq.biorate(T)
            # else:
            lgen = sum(populations)/const.pkill
            if verbose:
                print("-------------------")
                print("END OF GENERATION ",tgen)
                print("-------------------")
                
    
    # After maxgens or dt, calculate runtime 
    print("End of "+str(tgen)+" generations")
    
    # Save state of TNM to files
    save_files(encountered,modelrun,seed,experiment[1],output_path)

    manage.save_at_end(t0,tgen,seed,T,experiment,output_path,time_file,restart_file,rng,species,populations,encountered,Jran1,Jran2,Jran3,Tresponse)

    print("TNM: Run took ",time.time()-t0," seconds")
    return rng,Jran1,Jran2,Jran3,encountered,species,populations
            
