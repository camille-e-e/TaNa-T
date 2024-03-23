
# Tangled Nature (TaNa) Model + Temperature (T)
==============================================


This package includes modules to run the temperature-modified version of the Tangeld Nature (TaNa) model (Christensen & Jensen, 2002).

DOCUMENTATION
-------------
created May 25, 2022 by Camille Febvre
cfebvre@uvic.ca 
last modified March, 2024 by Camille Febvre

This repo contains the mechanics of the geoTNM, a python version of the Tangled Nature Model (TNM) (adapted from Christensen et al, 2010, and Arthur's opensource C++ TNM) which is modified by temperature according to metabolic theory of biologic rates.

The complete module is called geoTNM, which is a folder located in SRC.

USAGE:
-----
To run this module, navigate to /tests and then modify set\_inputs\_and\_launch\_TNM.sh with desired temperature, seed, and experiment.  (Other inputs, such as output path, can also be edited here.)  Then type
        bash set\_inputs\_and\_launch\_TNM.sh
This will copy the current versions of relevant TNM scripts and inputs to the output directory and launch the job from there.

CONTENTS:
---------
SRC contains the package TaNa+T, which contains these scripts:
* tangled\_nature.py: initialization and main routine of the TNM.
* TNM\_constants.py: constants and runtime options for the TNM.
* file\_management.py: a script to create output folders and file names depending on the date and experiment type, and print outputs every gerneration and at the end of the experiment.
* life\_events.py: functions for birth, death, and fitness calculations.
* offspring.py: functions for reproduction and mutations (new species formation)
* run\_TNM.py: launch tangled\_nature
* submit\_job.cmd: define inputs and submit run\_geoTNM to the PBS scheduler
* \_\_init\_\_.py: makes the geoTNM folder a module
* analysis/: contains script to analyze interactions between species after a model has completed.

tests contains scripts to launch experiments:
* set\_inpupts\_and\_launch\_TNM.sh: set inputs and launch experiment
* submit\_template.cmd: template script which set\_inputs...sh modifies and copies to output path and submits to the scheduler

plotting contains scipts to plot TaNa+T outputs

notebooks: jupyter notebooks

