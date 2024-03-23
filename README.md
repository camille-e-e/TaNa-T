
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
To submit an experiment to a PBS queue, navigate to /tests and then modify launch_1by1.sh with desired temperature, seed, and experiment.  (Other inputs, such as output path, can also be edited here.)  Then type

`        bash launch_1by1.sh`

This will copy the current versions of relevant TNM scripts and inputs to the output directory and launch the job from there.

For more detailed instructions, including instructions on how to run the model without PBS, refer to the README.txt file in `/src/`.

CONTENTS:
---------
See README_contents.txt.

