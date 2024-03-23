DOCUMENTATION
-------------
May 25, 2022
Camille Febvre
cfebvre@uvic.ca camillejfebvre@gmail.com

This repo contains the Tangled Nature+Temperature (TaNa+T) model, a python version of the Tangled Nature (TaNa) Model (adapted from Christensen et al, 2010, and Arthur's opensource C++ TaNa model called the TNM) which we have modified to respond to temperature according to metabolic theory of biologic rates.  

The complete module is called geoTNM, which is a folder located in /src.

USAGE:
-----
To run this module, navigate to /src/launch_model and then modify launch_1by1.sh with desired temperature, seed, and experiment.  (Other inputs, such as output path, can also be edited here.)  Then type
	>>> bash launch_1by1.sh
This will copy the current versions of relevant TaNa+T scripts and inputs to the output directory and launch the job from there.  More details are included in the README.txt file in the src/TaNaT_model/.

This model produces output files, which are saved to the output location specified in /src/launch_model/launch_1by1.sh.

To plot the model output, navigate to /plotting and use the DIRECTORY.txt and README.txt files to determine which plots to make.

CONTENTS of src:
---------
/TaNaT_model/ contains the TaNa_model, which contains these scripts:
* tangled_nature.py: initialization and main routine of the TNM.
* TNM_constants.py: constants and runtime options for the TNM.
* file_management.py: a script to create output folders and file names depending on the date and experiment type, and print outputs every gerneration and at the end of the experiment.
* life_events.py: functions for birth, death, and fitness calculations.
* offspring.py: functions for reproduction and mutations (new species formation)
* run_TNM.py: launch tangled_nature
* submit_job.cmd: define inputs and submit run_geoTNM to the PBS scheduler
* __init__.py: makes the geoTNM folder a module
* analysis/: contains script to analyze interactions between species after a model has completed.

/launch_model/ contains scripts to launch experiments:
* launch_1by1.sh: set inputs and launch experiment
* submit_one_template.cmd: template script which launch_1by1.sh modifies and copies to output path and submits to the scheduler

/plotting/ contains scripts to plot model outputs:


