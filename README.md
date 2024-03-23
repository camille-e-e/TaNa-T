
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
See README_contents.txt.

