#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 13:04:31 2023

@author: cfebvre
"""

import numpy as np
import matplotlib.pyplot as plt
from geoTNM import tangled_nature as TNM
import os
from os.path import exists

# Package to estimate probability density funcitons
from fastkde import fastKDE
# Interpolation package in order to interpolate on a given grid
from scipy.interpolate import griddata

seed = 2000
day = "Jan_09"
T = 298
#Temperature array 
#T = np.arange(273,330)
maxgens = 10_000
filename = f"diversity_seed{str(seed)}{day}_{str(T)}K_TNM_L20.dat"
locat = f"/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/SteadyT/{day}/MTE-env/"
filename = locat+filename
if exists(locat):
    pass #print(os.listdir(locat))
else: 
    print("directory not found: ",locat)

# Number of experiments with stable states
# after 10000 generations
no_exp = np.random.randint(low=10, high=100, size=maxgens, dtype=int)

# NOTE:
# In order to make a contour plot the data need to be interpolated on 
# a common grid. Such a grid needs adjustment to the actual data set 
# in order to produce beautiful plots.

# axis points of Tref values
int_grid = np.arange(0,400,4) 

# initialize final data array
data = np.zeros((maxgens,len(int_grid)), float)
weighteddata = np.zeros((maxgens,len(int_grid)), float)


#for t in range(maxgens):
    
rng,Jran1,Jran2,Jran3,encountered,species,populations,Tresponse = TNM.init(seed,True)
#print("Tresponse: \n",Tresponse)

t = -1    
with open(filename,'r') as f:
    # load t'th row
    for row in f:
        if t >= maxgens: 
            break
        t += 1
        #print("t: ",t)
        # convert to list of species and their populations
        elements = row.split(' ')
        if len(elements) < 2:
            print("extinction?")
            print(elements)
            break
        # gen = elements[0]
        species_now = elements[1:-1:2]
        pops_now = elements[2::2]
    
        # get Topt for each species
        Topts_species = [] # np.zeros((len(species_now),))
        Topts_indivs = []
        for i in range(len(species_now)):
            spc = int(species_now[i])
            pop = int(pops_now[i])
#            print("spc: ",spc,type(spc))
            Topt_i = Tresponse[int(spc)]
#            print("Topt: ",Topt_i[0])
            Topts_species.append(float(Topt_i[0]))
            for _ in range(pop):
                Topts_indivs.append(float(Topt_i[0]))
        #print("Topts_species: ",Topts_species)
        #print("Topts_indivs: ",Topts_indivs)

        #print("nans in Topts_now: ",sum(np.isnan(Topts_now)))
        while sum(np.isnan(Topts_species))>0:
            print("removing NaNs from ",Topts_species)
            Topts_species.remove(np.nan)
        Topts_species = np.array(Topts_species)
        while sum(np.isnan(Topts_indivs))>0:
            print("removing NaNs from ",Topts_indivs)
            Topts_indivs.remove(np.nan)
        Topts_indivs = np.array(Topts_indivs)

        # unweighted
        try: 
            mypdf,axis = fastKDE.pdf(Topts_species)
            # Bring data on a common grid to plot on contour
            # This grid needs to be adjusted to the range and 
            # distribution of data
            mypdf_int = griddata(axis,mypdf,int_grid,method='cubic')
            
            #set values where no data is available to 0
            #this is consistent with probability theory
            mypdf_int[np.isnan(mypdf_int)]=0 

            #Normalize the probability density functions such that 
            # the integral is one on the common grid
            mypdf_int = mypdf_int / np.trapz(mypdf_int,int_grid)
            
            # save the data to the final array 
            data [t,:] = mypdf_int
        except: 
            print("Topts_species can't be used: ",Topts_species)
            pass

        # weighted
        try: 
            pdf_weighted,axis_weighted = fastKDE.pdf(Topts_indivs)
            # Bring data on a common grid to plot on contour
            # This grid needs to be adjusted to the range and 
            # distribution of data
            weightedpdf_int = griddata(axis_weighted,pdf_weighted,int_grid,method='cubic')

            #set values where no data is available to 0
            #this is consistent with probability theory
            weightedpdf_int[np.isnan(weightedpdf_int)]=0 

            #Normalize the probability density functions such that 
            # the integral is one on the common grid
            weightedpdf_int = weightedpdf_int / np.trapz(weightedpdf_int,int_grid)

            # save the data to the final array 
            weighteddata [t,:] = weightedpdf_int
        except:
            print("Topts_indivs can't be used ") #,Topts_indivs)
            pass
        

# Plot the array
plt.contourf(range(maxgens),int_grid,data.transpose(),100,cmap='GnBu')
#plt.legend(r"$T_{env}$")
plt.xlabel('time (gens)')
plt.ylabel('Tref')

plt.colorbar(label='PDF')
#plt.contour(range(maxgens),int_grid,data.transpose(),levels=[T,],color="red",linestyle="--")
plt.tight_layout()

fig,ax = plt.subplots()
im = ax.pcolormesh(range(maxgens),int_grid,data.transpose(), cmap='GnBu',vmin=0,vmax=2.5, shading='auto')
ax.plot([0,maxgens],[T,T],linestyle="--",color='red',label=r"$T_{env}$")
fig.colorbar(im, ax=ax, label="PDF") #, extend='both')
plt.legend()
ax.set_xlabel("Time (gen)")
ax.set_ylabel(r"$T_{ref} (^oC)$")

fig,ax = plt.subplots()
im = ax.pcolormesh(range(maxgens),int_grid,weighteddata.transpose(),vmin=0,vmax=2.5, cmap='GnBu', shading='auto')
ax.plot([0,maxgens],[T,T],linestyle="--",color='red',label=r"$T_{env}$")
fig.colorbar(im, ax=ax, label="PDF (weighted)") #, extend='both')
plt.legend()
ax.set_xlabel("Time (gen)")
ax.set_ylabel(r"$T_{ref} (^oC)$")



plt.show()

