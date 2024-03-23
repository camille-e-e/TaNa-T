#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 11:00:16 2021

@author: cfebvre

This module contains all constants used by the other scripts, and largely doesn't need to be modified.  Things that do get modified are in the options class: attatch_geochem, rng type, variable T response, SpPops file.  However, genome length and other parameters may also want to be edited here depending on the setup.
"""
import numpy as np
import os
#from geoTNM 
import defaults

# %% OPTIONS
class options:
    attatch_geochem = True # TNM driven by external source
    rng = "default" # choose rng if you don't want the normal one
    # "default", "low-numbers", "read-in"
    old_date = "Jun_17"
    old_location = "/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/BasicTNM/"+old_date+"/" # False # "Feb_21/SpunupT/" # location of experiment you want to restart (after experiments/)
    pickup_old_run = False #old_location # False or old_location
    # variable_Tresp = True # differnt T response for each species
    core_ID_file = True # save core_ID core_pop of each core species in each timestep
    SpPops_file = False # save SpPops_file as TNM is running
    #TPC = "var-TPC" # "MTE-env", "var-TPC", or "one-TPC"
    
# set output path
output_path = defaults.get_path()

# %% CONSTANTS FOR TNM

mu = 0.1 # scale of impact of species on environment
A = 0
C=100 # scale of interactions between species
max_gens = 100


#start population of a random 
Npop_init=500
D_init = 60

#probability of an indivdual to die 
pkill=0.2

#proportion of species that interact at all
theta=0.25
mu_theta = 0

# Genome length--allows for 2**genes possible different species
genes=20
N = 2**genes

#probability of mutation for random species
pmut=0.01 #*genes

# Temperature response varies by species
variable_Tresp = False
T_opt = 20

#modify probabilities with random number centered at 1 with half max at this number
# uncertainty = .05                       


# %% CONSTANTS for GEOCHEM MODEL
day_s = 24*3600 # seconds/day
year_s = day_s*365 # seconds/year
# Earth and ocean properties
g = 9.81 # m/s^2
p_std = 1.01325E+5 # (Pa) standard pressure of Earth
A_earth = 5.10067420e14 #m^2
A_ocean = 3.61e14 # m^2
c_p = 1.005E+3 # (J kg-1 K-1) standard heat capacity of atmosphere
c_water = 4184 # J⋅kg−1⋅K−1 # standard heat capacity of water
m_earth = 5.97237e24 #kg
m_ocean = 1.4e21 # kg 1.4e18 tonnes in Earth's hydrosphere
m_surfocean = A_ocean*100*1000 # m^3 * kg/m^3 = kg mass of top 100m of ocean
m_atmos = 5.1e18 # kg (Nasa https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html)
m_CO2 = 44/1000 # kg/mol
m_calc = 100/1000 # kg/mol molecular weight of calcite and aragonite 
S0=1.36E+3 # (W/m^2) solar constant
sigma=5.67E-08 # (W⋅m−2⋅K−4) Stefan constant
# weathering equation, see Penman et al 2020
alpha = .4 # exponent for weathering rate
R = 8.314 # J/K/mol universal gas const
beta = -50*1000/R # K (activatn NRG ~ 50kJ/mol)/(universal gas const)
# initial conditions
T_const = A_earth/c_water/m_surfocean # (m^2 K J^-1) rate of heating ocean
# T_const = g/(c_p*p_std) # rate of heating atmosphere
halflife_volc = 1.2e9*year_s # half life of potassium determines volc flux
K_to_C = 273.05





# day_s = 24*3600 # seconds/day
# year_s = day_s*365 # seconds/year

# # Earth and ocean properties
# g = 9.81 # m/s^2
# p_std = 1.01325E+5 # (Pa) standard pressure of Earth
# A_earth = 5.10067420e14 #m^2
# A_ocean = 3.61e14 # m^2
# c_p = 1.005E+3 # (J kg-1 K-1) standard heat capacity of atmosphere
# c_water = 4184 # J⋅kg−1⋅K−1 # standard heat capacity of water
# m_earth = 5.97237e24 #kg
# m_ocean = 1.4e21 # kg 1.4e18 tonnes in Earth's hydrosphere
# m_atmos = 5.1e18 # kg (Nasa https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html)
# m_CO2 = 44/1000 # kg/mol
# m_calc = 100/1000 # kg/mol molecular weight of calcite and aragonite 
# S0=1.36E+3 # (W/m^2) solar constant
# sigma=5.67E-08 # (W⋅m−2⋅K−4) Stefan constant
# # weathering equation, see Penman et al 2020
# alpha = .4 # exponent for weathering rate
# R = 8.314 # J/K/mol universal gas const
# beta = -50*1000/R # K (activatn NRG ~ 50kJ/mol)/(universal gas const)
# # initial conditions
# T_const = A_earth/c_water/m_ocean # (m^2 K J^-1) rate of heating ocean
# # T_const = g/(c_p*p_std) # rate of heating atmosphere
# F_v0 = 5e13/365/24/3600 #3e10 #6.4e14 # 5e21/(3600*24*365) # 5e12 mol C/yr (flux of C into atmos from volcanoes) Colburn et al 2015
# halflife_volc = 1.2e9*year_s # half life of potassium determines volc flux
# # flux_volc_kg = flux_volc*m_CO2/1000 # kg/s
# # flux_volc_Pa = flux_volc_kg*g/A_earth # Pa/s

# # pCO2_0 = 5e12/(3600*24*365) # mol C/s Colbourn et al 2015 # Pa
# # T_0 = # K
# # F_02 = # weathering flux

def print_vars():
    for i in vars():
        if len(i) < 8:
            print(i,vars()[i])
    print("halflife_volc ",halflife_volc)
