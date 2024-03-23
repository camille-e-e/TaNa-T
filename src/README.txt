This is the Tangled Nature + Temperature (TaNa+T) Model, created by Camille Febvre for 
'Thermal response of ecosystems: modelling how physiological responses to temperature 
scale up in communities", submitted to Journal of Theoretical Biology October 11, 2023.

The model is in the /TaNaT_model folder, and the scripts used to launch the model are contained in the /launch_model folder.

To run the model in the terminal, use 
	>>> python3 path/to/TaNaT_model/run_TNM.py --seed $seed --temp1 $temp1 --tempn $temp_end --interval $interval --experiment $experiment --output_path $out_dir --gens2run $gens2run --spinuptime $spinuptime --Tresponse $Tresponse --vary_pmut $vary_pmut

replacing "$..." with desired values and strings (no quotations necessary).

This command is also inside submit_one_template.cmd, which can be copied to a new file such as submit_one_example.sh,
modified with the variable names, and then run with bash from the terminal:
	>>> bash path/to/launch_model/submit_one_example.cmd
This can also be submit to a queuing system, such as PBS, using
	>>> qsub path/to/launch_model/submit_one_example.cmd

To submit the model to a PBS queuing system, use launch_1by1.sh, in which you can modify the 
desired inputs.  launch_1by1.sh will then use sed to insert the desired inputs into a copy of
submit_one_template.cmd and submit it to the queue.
	>>> bash path/to/launch_model/launch_1by1.sh

For any questions, please contact Camille Febvre at cfebvre@uvic.ca, or Dr. Colin Goldblatt at czg@uvic.ca.
