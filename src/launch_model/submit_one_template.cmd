#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N out_vary_pmut
#PBS -j oe
### PBS -l nodes=venus04

# This script is copied and modified by launch_1by1.sh to include user inputs, 
# and it is then submitted to PBS queuing system

# inputs: (copied from launch_1by1.sh)

hostname
#conda activate my-env

git_location=/net/anahim/nazko/home/cfebvre/repos/the_model_y3/src/geoTNM/

# print last commit
cd ${git_location}
git show --summary
echo "this is the last commit in the repo, which will be copied and saved to ${curdir}"

# print python version
python3 --version

# cd to output dir, where all .py files have been copied, and run model from there
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
