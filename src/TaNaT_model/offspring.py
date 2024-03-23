# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:02:28 2022

@author: camil

This module contains functions for species reproduction and mutation events.  If a species reproduces (determined in life_events module), the reproduction function of this module is called.  Each gene of the baby has a chance to mutate.  If no genes flip, then the populaion of the selected species increases by one.  If, however, there are any gene flips, then a new species ID is calculated (by converting the new binary genome to an integer) and the population of this new (or possibly pre-existing) species is increased by one.

FUNCTIONS: 
    reproduction: do all the above things.
"""
#from geoTNM 
import TNM_constants as const
import numpy as np
from test_classes import Species
import MTE_TPC_combo as Tmod

# offspring of one species
def reproduction(current_spc,species,populations,encountered,rng,T_pmut=False,Topt_list=False,verbose=False):
    """
    Reproduce an individual of the given species.  If there is a mutation event, the baby is part of a new species (whose ID is calucalted from its binary genome)

    IN: 
        state variables: current_spc (int), species (list), populations (list), rng, 
        options: pmut_T (False or current T), Topt_list (list of Topt vals for all species), verbose
    OUT:
        species, populations
    """
    # *** IF running a different thing, rng should deal with it. ***
    # list of true or false for each gene that could flip
    # print("offspring: vary_pmut?",T_pmut)
    if T_pmut:
        mutations = np.ones(const.genes)*rng.random(size=const.genes)<Tmod.pmut(T_pmut)
    else: mutations = np.ones(const.genes)*rng.random(size=const.genes)<const.pmut

    if sum(mutations) == 0:
        # no muations
        idx = species.index(current_spc)
        # increase population of current species by 1
        populations[idx] += 1
        # find encountered object and increase its population
        for spc in encountered:
            if spc.ID == current_spc:
                spc.pop += 1

        if verbose:
            print("No mutation, population of species ",species[idx]," increased to ", populations[idx])
        return species, populations, encountered
    else: # mutation occurred
        bin_new = bin(current_spc)[2:] # convert current species ID to binary
        if verbose:
            print(f"mother {current_spc} has a genome of {bin_new}")
        
        # make sure genome is right length for all chances of gene flips
        if len(bin_new) < const.genes:
            bin_new = '0'*(const.genes-len(bin_new)) + bin_new
            if verbose:
                print("Mother genome is ",bin_new)
        bin_new = list(bin_new) # convert string to list of 0's and 1's for each gene
        
        # flip corresponding genes
        while True in mutations:
            # mutations is np.array, but need list to use index()
            mut = list(mutations).index(True) # find first mutated gene
            # mutate gene
            bin_new[mut] = str(int(not int(bin_new[mut]))) # flip that gene for baby
            mutations[mut] = False # adjust list of mutation locations
            
        str_new = ('').join(bin_new) # convert list back to string
        if verbose:
            print("Baby genome is ",str_new)
        
        # convert string back to integer
        int_new = int(str_new,2)
        # check if this species already exists
        if int_new not in species:
            species.insert(0,int_new)
            populations.insert(0,1)
            # check if int_new is in encountered
            if int_new in [spc.ID for spc in encountered]:
                # update pop of spc
                for spc in encountered:
                    if spc.ID == int_new:
                        spc.pop += 1
                        break
            else:
                if type(Topt_list) == bool:
                    encountered.append(Species(int_new,1))
                else: encountered.append(Species(int_new,1,Topt_list[int_new]))

            if verbose:
                print("New species formed: ",int_new)
        else: # mutant matches existing species
            idx = species.index(int_new)
            populations[idx] += 1
            for spc in encountered:
                if spc.ID == int_new:
                    spc.pop += 1
                    break
            if verbose:
                print("Mutant matches species ",int_new,", population increased to ",populations[idx])
    return species,populations,encountered
