#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N out_vary_pmut
#PBS -j oe
#PBS -l nodes=venus04

# user inputs:
# -----------
seed=1249 # seed 
gens2run=10000 # number of generations to run the model
temp1=274 # first temperature at which to run an experiment
temp_end=320 # final temperature at which to run experiment
interval=3 # interval between temperatures
Tresponse=var-TRC # False (single-TRC), var-TRC, or MTE-env
vary_pmut=True 

day=Dec_14_23 # today's date
out_path=/net/venus/kenes/user/cfebvre/out_venus/TTNM_outputs/ # path to all model outputs
extra_folder=var-TRC/varypmut/ # folder of this experiment (in out_path)
out_dir=/net/venus/kenes/user/cfebvre/out_venus/TTNM_outputs/SteadyT/var-TRC/varypmut/Dec_14_23/ # = out_path/out_dir
git_location=/net/anahim/nazko/home/cfebvre/repos/the_model_y3/src/geoTNM/ # location of git repo containing original model

# do not modify:
# --------------
experiment=SteadyT # hold temperature constant throughout model run
spinuptime=1000 # not used in SteadyT
modify_params=False # indpendently vary model parameters

# print some useful information
# -----------------
# print last commit
cd ${git_location}
git show --summary
echo "this is the last commit in the repo, which will be copied and saved to ${curdir}"

# print python version
python3 --version

# RUN MODEL: 
# -----------
# cd to output dir
cd ${out_dir}
echo "running from current directory: "
pwd

if [ $experiment == "prob_test" ]; then
	echo "Running prob_test experiment";
	python3 ${out_dir}run_TNM.py --seed $seed --poff1 $poff1 --poffn $poffn --pdeath1 $pdeath1 --pdeathn $pdeathn --interval $interval --mu1 $mu1 --mun $mun --mu_interval $mu_interval --pmut1 $pmut1 --pmutn $pmutn --pmut_interval $pmut_interval --experiment $experiment --output_path $out_dir --gens2run $gens2run --spinuptime $spinuptime --Tresponse $Tresponse
elif [ $modify_params == "True" ]; then
	echo "Modifying parameters";
	python3 ${out_dir}run_TNM.py --seed $seed --temp1 $temp1 --tempn $temp_end --interval $interval --mu1 $mu1 --mun $mun --mu_interval $mu_interval --pmut1 $pmut1 --pmutn $pmutn --pmut_interval $pmut_interval --experiment $experiment --output_path $out_dir --gens2run $gens2run --spinuptime $spinuptime --Tresponse $Tresponse
else 
	echo "$experiment run with vary_pmut is ${vary_pmut}";
	python3 ${out_dir}run_TNM.py --seed $seed --temp1 $temp1 --tempn $temp_end --interval $interval --experiment $experiment --output_path $out_dir --gens2run $gens2run --spinuptime $spinuptime --Tresponse $Tresponse --vary_pmut $vary_pmut
fi

echo "${PBS_JOBNAME}"

exit
