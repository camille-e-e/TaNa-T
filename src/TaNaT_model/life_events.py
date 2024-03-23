# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:43:56 2022

@author: camil

This module contains functions for different life happenings in the TNM.  It also has functions to calculate species fitness.

FUNCTIONS:
    - fi: calculate sum_j (J_{ij} n_j ) - mu N
    - poff: calculate 1/(1+exp(-fi)) and multiply by B(T) if temperature is input
    - kill: choose a random individual of any species and threaten it with death
    - one_time_step: this function calls the other three functions to take all the steps in one TNM timestep.  A random individual is killed if it's unlucky.  A random individual reproduces if it's lucky (calling on the offspring module, there's also a chance of new species formation).

"""

#from geoTNM 
import TNM_constants as const 
import numpy as np
#from geoTNM import offspring, temperature_effects as Tmod #T_dependence as Tmod
#from geoTNM 
import offspring, MTE_TPC_combo as Tmod #T_dependence as Tmod

# TNM "fitness"
def fi(current_spc,Jran1,Jran2,Jran3,species,populations,verbose=False):
    """Calculate the impact of all other species on the current one 
    and density dependence (mu N). Mathematically, f_i = sum_j(J_{ij} n_j) - mu N .

    IN:
        current_spc (int), Jran1-3 (np.arrays), species (list), populations (list), verbose=False
    OUT: 
        f_i
    """
    if len(species) > 1:
        if Jran1[0] != 0: 
            print("Why is Jran[0] not zero?")
            Jran1[0] = 0
        interactions = np.array(species)^current_spc # np array of bitwise interaction of ID's
        val = sum(Jran1[interactions]*Jran2[interactions]*Jran3[species]*populations) #- const.mu*sum(populations)
        if verbose: # tested this all matches C++ exactly May 16, 2022
            print("calc_HI is ",val)
            i = species.index(current_spc)
            v2 = val*populations[i]/sum(populations)
            print("calc_HI*Nj/N is ",v2) # this sum_j(Jij*nj)
            my_fi = val/sum(populations) - const.mu*sum(populations)
            poff2 = 1/(1+np.exp(-my_fi))
            print("poff2 is: ",poff2)
        return val/sum(populations) - const.mu*sum(populations)
    # if no other species exist, fi is zero
    else: return 0
    
# poff
def poff(current_spc,Jran1,Jran2,Jran3,species,populations,T=False,Tresponse=False,TRC=False,verbose=False):
    """probability of reproduction depends on f (which depends on J matrix and 
    populations of extant species) and on T IF T is not False.  Mathematically, poff = 1/(1+exp(-fi)) or if T is an input, poff = B(T)/(1+exp(-fi)) 
    
    IN: 
        current_spc (int), Jran1-3 (np.arrays), species (list), populations (list), T=False, verbose=False
    OUT: 
        poff (float)
    """
    f = fi(current_spc,Jran1,Jran2,Jran3,species,populations,verbose)
    if type(T) != bool:
        if type(Tresponse) != bool:
            Tresp_i = Tresponse[current_spc]
        else: Tresp_i = False
        return Tmod.poff_total(f,T,Tresp_i,TRC)
    else: 
        return 1/(1+np.exp(-f)) 
    
# %%
# LIFE and DEATH
# choose an individual and threaten with death
def kill(species,populations,encountered,rng,T=False,verbose=False): # tested May 17, 2022
    """species and populations must be lists, rng is a random number generator,
    T can be int or float or False.
    
    Compare a random number to calculated p_death; if the number is smaller, choose
    a species and reduce its population by one.
    
    IN: 
        species (list), populations (list), rng, T=False, verbose=False
    OUT: 
        species, populations.""" 
    
    if verbose:
        print("KILL: There are ",len(species)," species")
    if T:
        if verbose: 
            print("Temperature: ",T," K")
        # if const.variable_Tresp: kill_prob = .2
        # else: kill_prob = 1- T_eq.sig2(T-const.K_to_C)
        kill_prob = Tmod.pdeath(T) # input T in Kelvin
    else: kill_prob = const.pkill 
    # an individual dies if random number less than pdeath
    rand_num = rng.random()
    if verbose: print(f"kill_prob: {kill_prob}, rand num: {rand_num}")
    if rand_num < kill_prob: #kill an individual
        if verbose: print("death")
        # choose which individual to kill
        probs = np.array(populations)/sum(populations)
        # choose index of species of killed individual
        chosen = rng.choice(range(len(species)),p=probs)
        # identify class object of killed species
        found = 0
        for spc in encountered:
            if spc.ID == species[chosen]:
                killed_spc = spc
                if killed_spc.pop > 0:
                    killed_spc.pop -= 1
                found = 1
                break
        if not found:
            print("Error! Killed species not found!",chosen)

        if verbose: print(f"Species {species[chosen]} population decreased to {populations[chosen]-1}")

        # decrease the population of the killed individual by one
        populations[chosen] -= 1
        # if that was the only member of the species, remove the species from lists
        if populations[chosen] <= 0:
            # species went extinct
            populations.pop(chosen)
            species.pop(chosen)
            if verbose:
                print(f"Species {chosen} extinct") 
            if len(species) <= 0:
                # a mass extinction has occurred
                print("Mass extinction (line 580)")
        return species,populations,encountered # individual killed
    else: # no individual dies
        if verbose:
            print("No species killed")
        return species,populations,encountered # no species killed

     
def one_timestep(species,populations,encountered,rng,T,Jran1,Jran2,Jran3,Tresponse=False,TRC=False,vary_pmut=False,verbose=False):
    """
    All the steps involved in one TNM timestep.  A random individual is killed if a random number is smaller than pdeath, then a random individual reproduces if a random numbers is smaller than poff.  If there is a reproduction event, a new species is created if a random number is smaller than pmut.

    IN: 
        species (list), popultaions (list), rng, T, Jran1-3 (np.arrays), verbose=False
    OUT:
        species, populations
    """    

    # threaten an individual with death
    species,populations,encountered = kill(species,populations,encountered,rng,T,verbose)
    # if this causes a mass extinction, record it in print_stats
    if sum(populations) <= 0: 
        return [],[],encountered
        
    # choose a species
    probs = np.array(populations)/sum(populations)
    idx = rng.choice(range(len(populations)),p = probs)
    if verbose:
        print("Random species ",species[idx]," chosen")
        
    # Possibly reproduce an individual
    reprod = poff(species[idx],Jran1,Jran2,Jran3,species,populations,T,Tresponse,TRC,verbose=False) # variable T response per species and SpPops files are possible options                
    if verbose: print("Reproduction probabilitiy: ",reprod)
    # collect Topt, Twidth, skew of this species
    if type(Tresponse) != bool: Topt_list = Tresponse[:,0]
    else: Topt_list = False
    # give T if vary_pmut
    if vary_pmut: T_pmut = T
    else: T_pmut = False
    # if poff > 1, produce one baby and get a chance for another
    two_chances = False
    if reprod > 1:
        # reproduce
        two_chances = True
        # print(f"chance for two babies! T: {T}, poff: {reprod}")
        species,populations,encountered = offspring.reproduction(species[idx],species,populations,encountered,rng,T_pmut,Topt_list,verbose)
        if verbose: print("One reproduction")
        # get a chance for a second
        reprod -= 1
    if rng.random() < reprod: #,attatch_geochem):
        if verbose:
            if two_chances: print("Two babies got born")
            print("Asexual reproduction")
        species,populations,encountered = offspring.reproduction(species[idx],species,populations,encountered,rng,T_pmut,Topt_list,verbose)
    else:
        if verbose:
            print("No reproduction")
                
    return species, populations, encountered
    
    
    
    
