DOCUMENTATION
-------------
May 25, 2022
Camille Febvre
cfebvre@uvic.ca camillejfebvre@gmail.com

This repo contains the mechanics of the geoTNM, a python version of the Tangled Nature Model (TNM) (adapted from Christensen et al, 2010, and Arthur's opensource C++ TNM) which is modified by temperature according to metabolic theory of biologic rates.  

The complete module is called geoTNM, which is a folder located in SRC.

USAGE:
-----
To run this module, navigate to /tests and then modify set_inputs_and_launch_TNM.sh with desired temperature, seed, and experiment.  (Other inputs, such as output path, can also be edited here.)  Then type
	bash set_inputs_and_launch_TNM.sh
This will copy the current versions of relevant TNM scripts and inputs to the output directory and launch the job from there.

CONTENTS:
---------
SRC contains the package geoTNM, which contains these scripts:
* tangled_nature.py: initialization and main routine of the TNM.
* setup_constants.py: constants and runtime options for the TNM.
* file_management.py: a script to create output folders and file names depending on the date and experiment type, and print outputs every gerneration and at the end of the experiment.
* life_events.py: functions for birth, death, and fitness calculations.
* offspring.py: functions for reproduction and mutations (new species formation)
* run_geoTNM.py: launch tangled_nature
* submit_job.cmd: define inputs and submit run_geoTNM to the PBS scheduler
* __init__.py: makes the geoTNM folder a module
* analysis/: contains script to analyze interactions between species after a model has completed.

tests contains scripts to launch experiments:
* set_inpupts_and_launch_TNM.sh: set inputs and launch experiment
* submit_template.cmd: template script which set_inputs...sh modifies and copies to output path and submits to the scheduler

notebooks: jupyter notebooks


