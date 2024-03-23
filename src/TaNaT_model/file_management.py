# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:26:57 2022

@author: camil

This script has three functions: file_naming, print_stats, and save_at_end.  All the functions have to do with writing outputs.  

file_naming (called before main TNM flow): 
    - based on the date and experiment type, make a new ouptut folder if necessary.  
    - name the output files with desired inputs in names.

print_stats (called at the end of each generation):
    - write current state of TNM to output file.

save_at_end (called after completion of experiment):
    - save species, populations, encounered, rng and J matrix as npy files that can be used to restart the experiment later.  the J matrix can also be used for analysis.

"""
#from geoTNM 
import TNM_constants as const
from datetime import date, timedelta
import time
import os
import numpy as np

if const.options.SpPops_file:
    import json

# %% Logistics
def file_naming(seed,experiment,T,output_final_location):
    """
    If the output_final_location is already defined, just use this for all file paths.

    IN: 
        seed
        experiment = ["experiment",T]
        T
        output_final_location = path/to/all/outputfiles

    OUT:
        pop_file,div_file,restart_file,time_file
        other_files = SpPops_file  

    """
    # Tag the output files
    today = date.today()
    tag = today.strftime("%b_%d_")
    # make sure that the tag matches the date in which the experiment started if this run is taking more than one day
    if tag[:-1] not in output_final_location and not const.options.pickup_old_run:
        print("Matching file date to folder date")
        i = 1
        while tag[:-1] not in output_final_location: # if today's date is different from folder date, use folder date instead
            yesterday = today - timedelta(days = i)
            tag = yesterday.strftime("%b_%d")
            i += 1
            if i == 50: # if after 50 days, still haven't found date, it probably didn't work, so just use today's date in filename
                print("Can't find start date! ERRORRR in file_management.file_naming()")
                print("tag: ",tag," ,filename: ",output_final_location)
                tag = today.strftime("%b_%d")
        tag += "_"
    else: print("Using tag: ",tag)
    name_tag = "_TNM"
    # put all the parameters in the filename
    #name_tag += "_C"+str(const.C) + "_A"+str(const.A) + "_mu"+ str(const.mu)
    #name_tag += "_theta"+str(const.theta) + "_mutheta"+str(const.mu_theta) + "_pmut"+str(const.pmut)
    name_tag += "_L" + str(const.genes)
    name_tag +=".dat"

    # file name for population output and diversity output files
    pop_file = output_final_location + "pypy_seed" +str(seed) + tag + str(int(experiment[1])) + "K" + name_tag
    div_file = output_final_location + "diversity_seed" +str(seed) + tag + str(int(experiment[1])) + "K" + name_tag
    if const.options.core_ID_file:
        core_ID_file = output_final_location + "core_IDs"+str(seed)+tag+str(int(experiment[1]))+"K"+name_tag
        other_files = core_ID_file
    elif const.options.SpPops_file:
        SpPops_file = output_final_location + "SpPops" +str(seed) + tag + str(int(experiment[1])) + "K" + name_tag[:-3] + "npy" #"json"
        print("SpPops_file: ",SpPops_file)
        other_files = SpPops_file
    else: other_files = 0
    restart_file = output_final_location + "RESTART_seed" +str(seed) + tag + str(int(experiment[1])) + "K" + name_tag
    time_file = output_final_location+"Run_times"+tag+experiment[0]+name_tag
    
    return pop_file,div_file,restart_file,time_file,other_files  

# if output_path is not final output path, make whole directory structure
def file_naming_all(seed,experiment,T,output_path,verbose=False):
    """
    INPUTS:
        seed - an int
        experiment - in the form ["experiment-type",forcing]
        T - an int or float for the temperature in Kelvin
        output_path - path/to/output/folder
        verbose - print lines about functioning

    OUTPUTS: 
        experiment_folder - name of experiment folder, ie "SpunupT"
        restart_folder - folder containing npy's at end of experiment
        output_final_location - output_path + folders for date and experiment 
        pop_file,div_file,restart_file,time_file - output file names
        other_files - False, or the only other output file that's setup is SpPops_file, which I would like to remove as an option

        """
    # Tag the output files     
    today = date.today()
    tag = today.strftime("%b_%d_")
    name_tag = "_TNM" 
    # put all the parameters in the filename
    #name_tag += "_C"+str(const.C) + "_A"+str(const.A) + "_mu"+ str(const.mu) 
    #name_tag += "_theta"+str(const.theta) + "_mutheta"+str(const.mu_theta) + "_pmut"+str(const.pmut)
    name_tag += "_L" + str(const.genes)
    name_tag +=".dat"
    
    # Make a folder for this experiment
    if experiment:
        # Make folder for experiment type (ie steady T or variable T)
        if experiment[0] in np.array(["steady","SteadyT"]):
            experiment_folder = "SteadyT" 
        elif experiment[0] in np.array(["TempStep","step"]):
            experiment_folder = "TempStep"
        elif experiment[0] in np.array(["LinearT", "linear"]):
            experiment_folder = "LinearT"
        elif experiment[0] in np.array(["ProbabilityTest","prob_test"]):
            experiment_folder = "ProbabilityTest"
            T = False
        elif experiment[0] == "T+spinup" or experiment[0] == "SpunupT":
            experiment_folder = "SpunupT"
        elif experiment[0] == "basic" or experiment[0] == "BasicTNM":
            experiment_folder = "BasicTNM"
        elif experiment[0] == "Test" or experiment[0] == "test":
            experiment_folder = "Test"
        else: # If experiment type not defined yet, print error message
            print("Experiment: ", experiment)
            print("Error in TNM.main(): experiment not defined")
            experiment_folder = "UnknownExperiment"
        if not experiment_folder.endswith("/"):
            experiment_folder=experiment_folder+"/"

        # Make a folder for the results of this experiment if one doesn't already exist
        if experiment_folder[:-1] not in os.listdir(output_path):
        # print(os.listdir(output_path))
            time.sleep(1) # Wait a second and check again
            if experiment_folder[:-1] not in os.listdir(output_path):
                try: os.mkdir(output_path+experiment_folder)
                except: print("Tried to make new file but couldn't")
        
        restart_folder = tag[:-1]+"/"
        # check if this folder exists, and make it if not
        if restart_folder[:-1] not in os.listdir(output_path+experiment_folder):
            # Check again
            if restart_folder[:-1] not in os.listdir(output_path+experiment_folder):
                try:
                    os.mkdir(output_path+experiment_folder+restart_folder)
                except:
                    print("ERROR: restart_folder not found: ",restart_folder)
                    print("contents of output_path+experiment_folder: ",os.listdir(output_path+experiment_folder))

        # location of all output files (.dat, .npy, and .txt)
        output_final_location = output_path+experiment_folder+restart_folder
        print("output location: ",output_final_location)
            
        # file name for population output and diversity output files
        pop_file = output_final_location + "pypy_seed" +str(seed) + tag + str(experiment[1]) + "K" + name_tag
        div_file = output_final_location + "diversity_seed" +str(seed) + tag + str(experiment[1]) + "K" + name_tag
        if const.options.SpPops_file:
            SpPops_file = output_final_location + "SpPops" +str(seed) + tag + str(experiment[1]) + "K" + name_tag[:-3] + "npy" #"json"
            print("SpPops_file: ",SpPops_file)
            other_files = SpPops_file
        elif const.options.core_ID_file:
            core_ID_file = output_final_location + "core_IDs"+str(seed)+tag+str(experiment[1])+"K"+name_tag
            other_files = core_ID_file
        else: other_files = 0
        restart_file = output_final_location + "RESTART_seed" +str(seed) + tag + str(experiment[1]) + "K" + name_tag
        time_file = output_path+"Run_times"+tag+experiment[0]+name_tag
    
    else: # Experiment_folder not given in main inputs
        output_final_location = output_path
        
        # file name for population output and diversity output files
        pop_file = output_path + "pypy_seed" +str(seed) + tag + str(T) + "K" + name_tag
        div_file = output_path + "diversity_seed" +str(seed) + tag + str(T) + "K" + name_tag
        if const.options.SpPops_file:
            SpPops_file = output_path + "SpPops" +str(seed) + tag + str(experiment[1]) + "K" + name_tag[:-3] + "npy" # "json"
            print("SpPops_file: ",SpPops_file)
            other_files = SpPops_file
        else: other_files = 0
        restart_file = output_path + "RESTART_seed" +str(seed) + tag + "K" + name_tag
        time_file = output_path+"Run_times"+name_tag

    if verbose:
        print("Printing to: ", pop_file,div_file,restart_file)

    return experiment_folder,restart_folder,output_final_location,pop_file,div_file,restart_file,time_file,other_files  

def get_interactions(cur_spc,Jran1,Jran2,Jran3,species,populations):
    """Calculate the total effects of all other species times their populations on this species.
    
    INPUTS:
        current_spc, Jran1-3,species,populations,verbose
    OUTPUS: 
        sum_j(J_ij N_j) """
    if len(species) < 1:
        return 0
    else:
        Jran1[0] = 0
        bitwise = np.array(species)^cur_spc
        # sum_j(J_ij N_j)
        interactions = sum(Jran1[bitwise]*Jran2[bitwise]*Jran3[species]*populations)
        return interactions

def print_stats(tgen,species,populations,encountered,Jran1,Jran2,Jran3,pop_file,div_file,years2run=const.max_gens,other_files=False,Tresponse=False,percent=0.05):
    """
    print status of TNM at the end of a generation.

    INPUTS: 
        tgen - TNM generation just completed
        species, populations, encountered, Jran1-3
        pop_file, div_file - filenames previously determined by file_naming above
        years2run - total number of years in experiment, for SpPops file, otherwise can be removed later
        other_files - SpPops_file if that is being made
        Tresp - if not False, 3 lists of Tresponses by species: T_ref, width, skew
        percent - percent of maximum population for a species to be counted in core

    NO OUTPUTS.
    SAVES FILES: 
        pypy.... : t popu div enc core_pop core_div F Jij_avg, J_core_avg, avg_Tref # 0 F 0
        diversity.... : t sp1 pop1 sp2 pop2 ....spn popn
    """
    N = sum(populations)
    core_pop = 0
    core_div = 0
    F = 0 # sum_i (f_i*N_i) = sum_i(N_i(-mu*N + sum_j(J_ij*nj)))
    F_core = 0 # " for core-core interactions
    Jij_avg = 0 # sum_i N_i/N * sum_j(J_ij N_j/(N-N_i))
    J_core_avg = 0 # 1/N_core * sum_ci N_ci*(sum_cj(J_ciji n_cj)) where ci and cj are indices of core species

    # find largest population
    if len(populations) > 0:
        maxP = max(populations)
    else: 
        maxP = 0

    # find species with large populations
    core_TF = np.array(populations)>percent*maxP # array of T/F
    # array of only large populations
    core_popus = np.array(populations)[core_TF]
    # population of all dominant species
    core_pop = sum(core_popus)
    # number of dominant species
    core_div = len(core_popus)        

    # indices of dominant species
    core_indices = list(range(1,1+len(core_TF))*core_TF)
    while 0 in core_indices:
        core_indices.remove(0)
    core_indices = np.array(core_indices,dtype=int)-1 #np.ones(len(core_indices),dtype=int)
    # print(core_indices, type(core_indices))

    # find total ecosystem fitness (F), average interactions of ecosystem (Jij_avg), and avg core interactions (J_core_avg)
    for spc in (np.array(species)):
        spc_idx = species.index(int(spc))
        spc_popu = populations[spc_idx]
        fitness_i = spc_popu*(get_interactions(spc,Jran1,Jran2,Jran3,species,populations)/N - const.mu*N) # spc_popu*calc_f(spc,Jran1,Jran2,Jran3,species,populations)
        interac_i = spc_popu/N*get_interactions(spc,Jran1,Jran2,Jran3,species,populations)/(N-spc_popu) # spc_popu/N*calc_J(spc,Jran1,Jran2,Jran3,species,populations)
        F += fitness_i
        Jij_avg += interac_i
        if spc_popu > percent*maxP: # this is a core species
            F_core += fitness_i
            J_core_avg += interac_i # spc_popu/sum(np.array(populations)[core_indices]) * calc_J(spc,Jran1,Jran2,Jran3,np.array(species)[core_indices],np.array(populations)[core_indices])

    # if species have different Tresponse, record average Tresponses
    #print(type(Tresponse),": Tresponse type")
    if type(Tresponse) != bool:
        avg_Tref = 0
        avg_Twidth = 0
        for spc in species:
            avg_Tref+=Tresponse[spc][0]*populations[species.index(spc)]/sum(populations)
            avg_Twidth += Tresponse[spc][2]*populations[species.index(spc)]/sum(populations)
    else: avg_Tref,avg_Twidth = 0,0

    # now print stats into pop file
    with open(pop_file,'a') as f: # set E and mu to 0
        # tgen sum(populations) diversity len(encountered) core_pop core_div F Ji_avg J_core_avg)
        f.write("%d %d %d %d %d %d %f %f %f %f\n"%(tgen,sum(populations),len(species),len(encountered),core_pop,core_div,F,Jij_avg,J_core_avg,avg_Tref))
    
    # diversity file has: tgen sp1 pop1 sp2 pop2 ... spn popn
    with open(div_file,'a') as d:
        str_sp = str(tgen)+' '
        for i in range(len(species)):
            str_sp += str(species[i])+" "+str(populations[i])+" "
        str_sp += "\n"
        d.write(str_sp)
    
    # this process doesn't work well!
    # save SpPops file while TNM is running
    if other_files:
        if other_files: # this must be core_species file
            core_species = np.array(species)[core_indices]
            core_pops = np.array(populations)[core_indices]
            with open(other_files,'a') as f:
                str_sp = str(tgen)+' '
                for i in range(len(core_species)):
                    str_sp += str(core_species[i])+" "+str(core_pops[i])+" "
                str_sp += "\n"
                f.write(str_sp)
        elif other_files == "SpPop_file":
            SpPop_file = other_files
            if os.path.exists(SpPop_file):
                if SpPop_file.endswith("json"): 
                    with open(SpPop_file,'r+') as f: #'rb') as f:
                    #SpPops = pickle.load(f) #SpPop_file,allow_pickle=True)
                    #try:
                        print("SpPops file: ",SpPop_file)
                        SpPops = json.load(f)
                        print("SpPops load successful")
                    #except:
                     #   print("ERRORRR LOADING JSON FILE!!!")
                elif SpPop_file.endswith("npy"):
                    SpPops = (np.load(SpPop_file,allow_pickle=True)).item()
                    #print("SpPops numpy file opened")
                else: print("ERROR: SpPop file wrongly defined???: ",SpPop_file)
            else:
                SpPops = {}
                print("New SpPops dict created")
            i = -1
            for ID in species:
                i += 1
                # print("ID, pop: ",ID,populations[i])
                if str(ID) not in SpPops.keys():
                    #try:
                    SpPops[str(ID)] = [] # np.zeros(maxgens+1)
                    #except:
                    #    print("Couldn't assign zeros to spc ",spc," ... maxgens is ",maxgens)
                    #    SpPops[str(spc)] = 12
                    #    print("Saved this instead: ",SpPops[str(spc)])
                pop_now = populations[i]
                SpPops[str(ID)].append([int(tgen),pop_now])
                #SpPops[str(ID)][tgen] = populations[i]
                #try:
                # SpPops[str(spc)][int(tgen)] = pop_now
                #except:
                 #   print(f"couldn't assign anything to {str(spc)} at {int(tgen)}")
                # print("SpPops of this ID and tgen: ",SpPops[str(spc)][int(tgen)])
            # print("At this point, SpPops is a ",type(SpPops)," and it is ",SpPops)
            if SpPop_file.endswith("json"):
                with open(SpPop_file,'w') as f: # b') as f:
                    print("SpPop_file is open")
                    #try: 
                    print("SpPops file: ",SpPop_file)
                    json.dump(SpPops,f)
                    print("SpPops dump successful")
                    #except: 
                    #print("ERRORRRR DUMPING JSON FILE!!!")
                #pickle.dump(SpPops,f)
            else:
                np.save(SpPop_file,SpPops)    

def save_at_end(t0,tgen,seed,T,experiment,output_final_location,time_file,restart_file,rng,species,populations,encountered,Jran1,Jran2,Jran3,variable_Tresp=False):
    """
    After experiment has completed, save npy files so experiment can be picked up again later.

    INPUTS:
        t0: computer time at start of experiment
        tgen: last generation in experiment
        seed, T: obvious
        experiment: ["experiment-type",T] 
        # NO FOLDER NAMES OR PATHS SHOULD END WITH SLASH
        experiment_folder: "folder-name"
        output_path: '/path/to/experiment/folder'
        output_final_location: output_path + experiment_folder (+ restart_folder) # don't end with /
        time_file: path/and/time_file.dat
        restart_file: path/and/restart_file.dat
        # if picked up old run, put new results in old path but new folder therewithin
        restart_folder: 'date_today' (if different from date of initial run)
        rng,species,populations,encountered,Jran1,Jran2,Jran3: as usual
        variable_Tresp: if False, do nothing.  Otherwise save these values.
    
    NO OUTPUTS.
    """
    
    # a test...
    #if output_path+restart_folder+experiment_folder == output_final_location:
    #    print("no need for output_path, restart_folder, and experiment_folder")
    #else: 
    #    print("output_final_location: ",output_final_location)
    #    print("output_path: ",output_path,"/nrestart_folder",restart_folder)
    #    print("experiment_folder",experiment_folder)
    
    # Tag the output files     
    today = date.today()
    tag = today.strftime("%b_%d_")
    
    with open(time_file,"a") as f:
        f.write(f"\n{(time.time()-t0)/3600} hours, {tgen} generations, seed {seed}")
    
    with open(restart_file,"a") as f:
        f.write(f"\nJran1: {Jran1}")
        f.write(f"\nJran2: {Jran2}")
        f.write(f"\nJran3: {Jran3}")
        f.write(f"\nencountered: {encountered}")
        f.write(f"\nspecies: {species}")
        f.write(f"\npopulations: {populations}")
        if str(type(rng)) == "<class 'numpy.random.generator.Generator'>": # "numpy.random._generator.Generator" or str(type(rng)) == 'np.random._generator.Generator':
            f.write("rng state: {rng.bit_generator.state}")
        else:
            try:
                f.write(f"rng state: {rng.get_state()}")
            except:
                print("Can't figure out rng type: ",type(rng))
        
    np.save(output_final_location+"/Js_combined"+str(seed)+tag+str(experiment[1])[:4]+"K"+".npy",[Jran1,Jran2,Jran3])
    np.save(output_final_location+"/encountered"+str(seed)+tag+str(experiment[1])[:4]+"K"+".npy",encountered)
    np.save(output_final_location+"/species"+str(seed)+tag+str(experiment[1])[:4]+"K"+".npy",species)
    np.save(output_final_location+"/populations"+str(seed)+tag+str(experiment[1])[:4]+"K"+".npy", populations)
    if type(variable_Tresp) != bool:
        np.save(output_final_location+"/variable_Tresp"+str(seed)+tag+str(experiment[1])[:4]+"K"+".npy", variable_Tresp)
    if str(type(rng)) == "<class 'numpy.random.generator.Generator'>":#'np.random._generator.Generator' or str(type(rng)) == 'numpy.random._generator.Generator':
        print("skipping saving rng to avoid memory issue, line 947 of TNM")
        #        np.save(output_final_location+"/rng"+str(seed)+tag+str(experiment[1])[:4]+"K"+".npy",[rng.bit_generator.state],allow_pickle=True)
    else:
        print("Can't figure out rng type: ",type(rng))
        #try:
        #    np.save(output_path+restart_folder+experiment_folder+"/rng"+str(seed)+tag+str(experiment[1])+"K"+".npy",[rng.get_state()])
        #except:
        #    print("Can't figure out rng type: ",type(rng))
