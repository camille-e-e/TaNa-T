# THIS IS A BASH SCRIPT TO LAUNCH THE TNM.  DEFINE YOUR INPUTS HERE, WHICH WILL BE COPIED TO THE submit_job.cmd FILE AND THEN THAT WILL BE qsub-bed.  

# Define your inputs below.  These will be copied to the submit_template which will then launch the geoTNM.

# EDIT THESE INPUTS
# -----------------
# constants to modify
vary_pmut=False # vary pmut according to T
modify_params=False # fix parameter values in ranges defined below
mu1=.05
mun=.06
mu_interval=.03
pmut1=.01 #.004
pmutn=.016
pmut_interval=.01 #.003

# these will only be used if experiment is NOT "prob_test"
temp1=274
temp_end=320
# these will only be used if experiment == "prob_test"
poff1=.1
poffn=1.1
pdeath1=.1
pdeathn=1
# always used
interval=3 # use for temps or probs

# geoTNM inputs
seed1=1000
num_seeds=50
experiment="SteadyT" # "BasicTNM","SpunupT","SteadyT","LinearT"
extra_folder="var-TRC/constpmut/" #"MTE-TRC-better-output/" #"slope_298_$temp_end/" #"met_theory_simple/" # '' if none, otherwise end in /
gens2run=10000 #50000
spinuptime=1000 #5000
Tresponse="var-TRC" # "MTE-env" # False, "var-TRC" or "MTE-env"
day=$(date +"%b_%d_%y") # "May_03" # today's date

# location of outputs
out_path=/net/venus/kenes/user/cfebvre/out_venus/TTNM_outputs/

# location of model
location=/net/anahim/nazko/home/cfebvre/repos/the_model_y3/src/geoTNM/

# location of launch_1by1.sh and submit_one_template.sh 
home_dir=/net/anahim/nazko/home/cfebvre/repos/the_model_y3/tests/

# DON'T EDIT BEYOND THIS POINT
# -----------------------------
# complete path to outputs
out_dir=$out_path${experiment}/${extra_folder}${day}/

# Make the new directory for outputs
if [ -d "$out_dir" ]; then
	    echo "$out_dir already exists."
    else 
	    if [ -d "$out_path$experiment/$extra_folder" ]; then
		    echo "$out_path$experiment/$extra_folder already exists"
		    echo "creating $out_dir"
		    mkdir $out_dir
	    else
		    echo "$out_path$experiment/$extra_folder doesn't exist, making two new folders"
		    mkdir $out_path$experiment/$extra_folder
		    mkdir $out_dir
	    fi
fi

# and copies the scripts to that output folder
# copy all python files to output folder
cp ${location}*.py ${out_dir}

## and then qsubs submit_run.
#cd $out_dir # change dir to output dir to try to get error file over there
#echo "currently in "
#pwd

# submit each seed
seedn=($seed1+$num_seeds)
nn=0
for (( seed=$seed1; seed<$seedn; seed++ ))
do
	#nn++
	#if (( $nn % $njobs = 0 )); then
	cd ${home_dir}
	# Copy template to new file
	job_file=submit_one_$day.cmd
	cp ./submit_one_template.cmd ./$job_file # submit_job_$day.cmd
	
	# add inputs into this launch scipt
	if [ $experiment == "prob_test" ]; then # vary params in TNM
		echo "launcher detected a prob test";
		sed -ie "/^# inputs:/a # beginning of insertion \nseed=$seed \npoff1=$poff1 \npoffn=$poffn \npdeath1=$pdeath1 \npdeathn=$pdeathn \ninterval=$interval \nmu1=$mu1 \nmun=$mun \nmu_interval=$mu_interval \npmut1=$pmut1 \npmutn=$pmutn \npmut_interval=$pmut_interval \nexperiment=$experiment \nout_path=$out_path \ngens2run=$gens2run \nspinuptime=$spinuptime \nday=$day \nextra_folder=$extra_folder \nout_dir=$out_dir \nTresponse=$Tresponse \n#end of insertion" $job_file # submit_job_$day.cmd
	elif [ $modify_params == "True" ]; then # vary params in T TNM
		echo "launcher did not detect prob test but still varying parameters";
		sed -ie "/^# inputs:/a # beginning of insertion \nseed=$seed \ntemp1=$temp1 \ntemp_end=$temp_end \ninterval=$interval \nmu1=$mu1 \nmun=$mun \nmu_interval=$mu_interval \npmut1=$pmut1 \npmutn=$pmutn \npmut_interval=$pmut_interval \nexperiment=$experiment \nout_path=$out_path \ngens2run=$gens2run \nspinuptime=$spinuptime \nday=$day \nextra_folder=$extra_folder \nout_dir=$out_dir \nTresponse=$Tresponse \nmodify_params=$modify_params \nvary_pmut=$vary_pmut \n#end of insertion" $job_file # submit_job_$day.cmd
	else
		echo "launcher detected T+TNM run without parameter modification";
		sed -ie "/^# inputs:/a # beginning of insertion \nseed=$seed \ntemp1=$temp1 \ntemp_end=$temp_end \ninterval=$interval \nexperiment=$experiment \nout_path=$out_path \ngens2run=$gens2run \nspinuptime=$spinuptime \nday=$day \nextra_folder=$extra_folder \nout_dir=$out_dir \nTresponse=$Tresponse \nvary_pmut=$vary_pmut \nmodify_params=$modify_params \n#end of insertion" $job_file # submit_job_$day.cmd
	fi
	
	# mv $job_file $out_dir
	mv ${job_file} ${out_dir}
	# change dir to output dir to try to get error file over there
	cd $out_dir #
	# submit
	qsub ${out_dir}${job_file}
      		
	#else
	#	echo "......."
    	#fi
	
done





#echo "about to qsub from " >> ./out_111.txt
#pwd >> ./out_111.txt



