"""
Purpose: run the TNM+T
Required arguments: seed
Recommended arguments: 
    - gens2run (how many generations to run the model)
    - output_path (path to which to save model outputs)
    - experiment ()
    - Tresponse (False for single-TRC, 'var-TRC' for various-TRC, or 'MTE-env' for MTE-envelope)
    - temp1, tempn, interval (temperatures at which to run experiments, from temp1 to tempn (not included) at intervals of interval)

Author: Camille Febvre
Last modified: Mar 15, 2024
"""

import numpy as np
import TNM_constants as const, tangled_nature as TNM
import sys
import time
import os
from datetime import date, datetime, timedelta
import argparse


# collect inputs from bash
if len(sys.argv) > 3:
    parser = argparse.ArgumentParser()
    var_ints = ['seed','gens2run','spinuptime']
    var_strs = ['experiment','output_path','out_dir','Tresponse','vary_pmut']
    var_floats = ['temp1','tempn','poff1','poffn','pdeath1','pdeathn','interval']
    for v in var_ints: 
        parser.add_argument("--"+v,type=int)
    for v in var_strs:
        parser.add_argument("--"+v,type=str)
    for v in var_floats:
        parser.add_argument("--"+v,type=float)
    args = parser.parse_args()

    # important variables
    seed = args.seed
    if args.output_path:
        output_path = args.output_path
    else: output_path = False
    if args.gens2run:
        print("Input gens2run: ",args.gens2run,type(args.gens2run))
        gens2run = args.gens2run
    else:
        print("Running 10 gens")
        gens2run = 10

    # if there's a Tresponse, remember its kind in TRC
    if args.Tresponse: 
        print("Tresponse input: ",args.Tresponse)
        if args.Tresponse == "False":
            Tresp_TF = False
            TRC = False
        else:
            Tresp_TF = True
            TRC = args.Tresponse
    else:
        Tresp_TF = False
        TRC = False

    # variable Tresponse?
    if args.vary_pmut:
        vary_pmut = args.vary_pmut
    else: vary_pmut = False

    # make temperature array
    if args.temp1:
        if args.tempn < args.temp1:
            interval = -args.interval
            temps2run = np.r_[args.temp1:args.tempn:interval]
        elif args.tempn == args.temp1:
            temps2run = [args.temp1]
        else: 
            temps2run = np.r_[args.temp1:args.tempn:args.interval]
        print("temps2run: ", temps2run)

    if args.poff1:
        if args.poffn < args.poff1:
            interval = -args.interval
            poff_range = np.r_[args.poff1:args.poffn:interval]
        elif args.poffn == args.poff1:
            poff_range = [args.poff1]
        else: 
            poff_range = np.r_[args.poff1:args.poffn:args.interval]
    if args.pdeath1:
        if args.pdeathn < args.pdeath1:
            interval = -args.interval
            pdeath_range = np.r_[args.pdeath1:args.pdeathn:interval]
        elif args.pdeathn == args.pdeath1:
            pdeath_range = [args.pdeath1]
        else: 
            pdeath_range = np.r_[args.pdeath1:args.pdeathn:args.interval]

    # make forcings list
    if args.experiment in np.array(["Test","test","basic","BasicTNM","LinearT","linear"]):
        forcings = [False]
    elif "prob" in args.experiment:
        forcings = []
        for po in poff_range:
            for pd in pdeath_range:
                forcings.append([po,pd])
    else: forcings = temps2run

else: print("Insufficient inputs detected (seed1,num_seeds,temp1,tempn,num_temps,experiment,output_path,gens2run(,spinuptime)")

def launch_experiment(seed,forcing=False,gens2run=10,experiment="BasicTNM",output_path=False,Tresp_TF=False,TRC=False,vary_pmut=False):
    t0 = time.time()

    experiment_package = [experiment,forcing]

    if not output_path:
        print("output_path not defined, using cwd:")
        output_path = os.getcwd()+"/"
        print(output_path,"\n-*-*-*-*-*-")

    # OPTION TO PICKUP OLD RUN
    # ****

    if Tresp_TF:
        print("Variable Tresponse experiment being launched")
        rng,Jran1,Jran2,Jran3,encountered,species,populations,Tresponse = TNM.init(seed,Tresp_TF)
    else: 
        rng,Jran1,Jran2,Jran3,encountered,species,populations = TNM.init(seed)
        Tresponse=False
        print("Tresponse: ",Tresponse)

    # OPTION FOR SPINUP or LINEART
    # ****

    steps = 1
    s2run = gens2run*const.year_s
    for step in range(steps):
        print("run_TNM: vary_pmut? ",vary_pmut)
        rng,Jran1,Jran2,Jran3,encountered,species,populations = TNM.main(rng,Jran1,Jran2,Jran3,encountered,species,populations,forcing,s2run,seed,experiment_package,output_path,Tresponse=Tresponse,TRC=TRC,vary_pmut=vary_pmut)

        if type(populations) == int:
            print("Run ",seed," took ",time.time()-t0," seconds to run.")
            break
        elif sum(populations) <= 1:
            print("Run ",seed," took ",time.time()-t0," seconds to run.")
            break

    print("Run ",seed," took ",time.time()-t0," seconds to run.")

if __name__ == "__main__":
    print(f"\n-----\nExperiment: {args.experiment},\nForcings: {forcings},\nseed {seed}")
    t1 = time.time()

    print("variable Tresponse between species?",Tresp_TF)
    for force in forcings:
        launch_experiment(seed,force,gens2run,args.experiment,output_path,Tresp_TF,TRC,vary_pmut)

    print("Total runtime: ",(time.time()-t1)/60," minutes")

