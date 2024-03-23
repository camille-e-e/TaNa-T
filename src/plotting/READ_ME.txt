# To plot the following thermal responses, follow the instructions below:

# Survival
----------
1. class_plots.py?
2. final_grid_plots: among other things, interpolates survival from grid for constant pmut
3. compare_survival.py: loads npy files for single_TRC & var_TRC and plots survival


# Abundance and diversity
--------------------------
1. run class_plots.py to produce npy files, e.g.:
	- final_stats_single-TRC_Jun_21_23.npy 
	- final_stats_var-TRC_Apr_13_23.npy
2. run final_ND_all_experiments.py to produce multi-panel figure with final abundance and diversity of core and ecosystem

# Arrhenius plots
-----------------
1. run by_spc_plots.py to produce npy files
    - N_by_T_{exp}_{date}.npy
    - D_by_T_{exp}_{date}.npy
2. run scatter_SAD_stats.py to produce Arrhenius plots of abundance and diversity

# SAD plots
-----------
1.a) run by_spc_plots.py to produce npy files (very long runtime (20min?))
	- skewness_by_T_{date}.npy
    	- SAD_by_T_{exp}_{date}.npy
1.b) also run class_plots.py to get 
	- final_stats_{exp}_{date}.npy
	- final_N_{exp}_{date}.npy
	- final_D_{exp}_{date}.npy
2. run scatter_SAD_stats.py to produce scatter plots of SAD skewness for single and var TRC
	- PDF and CDF of SADs for core and ecosystem in single and var-TRC
	- scatter plots of skewness vs. T
	- Arrhenius plots for N and D vs. T

# Interactions
----------------
0. class_plots.py also makes some interaction plots by T
1. run plot_interactions.py in the same directory in which model output is.  This produces npy files:
	- interactions_by_T.npy
	- core_int_by_T.npy
	- interaction_types_by_T.npy
	- core_interaction_types_by_T.npy
	- connectance_by_T.npy
2. run make_final_interaction_plot.py, either in model output folder or in repos/the_model_y3/plotting/ to produce final figure
	- mean and median Jij
	- mean and median |Jij|
	- fraction of interaction types
1. run plot_Topt_poff_fi_etc.py to plot Topt and fi distributions, in addition to calculated poff_T, poff_T/pdeath = poff_i, and J_ij_mean_over_j

# Quakes
--------  
1. class_plots:
	- boxplots of n. of quakes per experiment, quakes vs. survival
	- many other figures, not much sense though

# Individual Runs
----------------
analyse_classes_copy.py
	- lots of cool plots about interaction strength, quake duration etc

