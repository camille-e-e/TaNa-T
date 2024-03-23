"""
Purpose of script is to load survival vectors from three experiments and plot on the same axes.

Created by Camille Febvre.
~approximately Oct 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("/home/cfebvre/repos/the_model_y3/src/geoTNM")
from MTE_TPC_combo import *

# figure sizes
sin_width = 3.54 # single column width (inches)
onehalf_width = 5.51 # 1.5x columnu
doub_width = 7.48 # full width

singleTRC_50 = np.load("npy_files/survival_single-TRC_Jun_21_23.npy",allow_pickle=True)
singleTRC_200 = np.load("npy_files/survival_single-TRC_Dec_14_23.npy",allow_pickle=True)
varTRC_50 = np.load("npy_files/survival_var-TRC_Apr_13_23.npy",allow_pickle=True)
varTRC_200 = np.load("npy_files/survival_var-TRC_Dec_14_23.npy",allow_pickle=True)
# this experiment had a different Tref so doesn't really apply here
singleTRC_constmut = np.load("npy_files/survival_single-TRC_Feb_04_24.npy",allow_pickle=True)
varTRC_constmut = np.load("npy_files/survival_var-TRC_Feb_05_24.npy",allow_pickle=True)
#interpolated = np.load("predicted_survival_prob_testFeb_09D_init_60.npy",allow_pickle=True)

# sample sizes
sample_sin = 250
sample_var = 250
sample_constmut = 50

T_range = np.r_[274:320:3]
all_temps = np.r_[274:320:.1]

# combine survival stats
singleTRC = singleTRC_50 + singleTRC_200
varTRC = varTRC_50+varTRC_200


fig,ax = plt.subplots(figsize=(doub_width,.7*doub_width))
f2,a2 = plt.subplots(2,sharex=True,figsize=(doub_width,doub_width))

for a in [ax,a2[0],a2[1]]:
    if a == ax: labels = ["Survival: single-TRC",r"Survival: p$_\mathrm{mut}$=0.01",r"p$_\mathrm{off,T}$: Single TRC","Survival: Various TRC"]
    else: labels = ["Survival",r"Surv.: p$_\mathrm{mut}$=0.01",r"p$_\mathrm{off,T}$","Survival"]
    if a != a2[1]:
        #ax.plot(T_range,constmut,label=r'p$_\mathrm{mut}$=0.01',color="k")
        l_sin = a.plot(T_range,singleTRC/sample_sin,"k",label=labels[0]) # label='Single TRC')
        #a.plot(all_temps,interpolated,"r--",label=labels[1])
        l_mut = a.plot(T_range,singleTRC_constmut/sample_constmut,"r",label=labels[1])
        # plot input TRC
        l_poff = a.plot(all_temps,poff_T(all_temps),"k-.",alpha=.2,label=labels[2])
    else:
        if a == ax: alpha = .5
        else: alpha = 1
        l_var = a.plot(T_range,varTRC/sample_var,"k",alpha=alpha,label=labels[3]) # "k:",label='Various TRC')
        l_var_mut = a.plot(T_range,varTRC_constmut/sample_constmut,"r",label=labels[1])
    if a == a2[0]:
        atwin = a.twinx()
        l_thresh = atwin.plot(all_temps,-np.log(poff_T(all_temps)/pdeath(all_temps)-1),"b--",alpha=.5,label=r"min. f$_i$")
        #atwin.plot(T_range,poff_T(T_range)-pdeath(T_range),"b",alpha=.5,label=r"$p_\mathrm{off,T} - p_\mathrm{death}$")
        atwin.set_ylim(2.6,-1.55) #1,4) #5.5)
        #atwin.set_ylabel(r"Prob. reproduction:death ratio",color="b")
        atwin.set_ylabel("Minimum fitness",color="b") #r"log($\frac{p_\mathrm{off,T}}{p_\mathrm{death}}$ - 1)",color="b")
        atwin.tick_params(axis='y', labelcolor="b")
        #atwin.legend(loc="upper right")
    elif a == a2[1]:
        atwin = a.twinx()
        for Topt in np.r_[274:320:3]:
            params = [Topt,11,-3]
            if Topt == 274: # label one line
                l_var_log = atwin.plot(all_temps,-np.log(poff_T(all_temps,params)/pdeath(all_temps)-1),"b--",alpha=.5)
            else: atwin.plot(all_temps,-np.log(poff_T(all_temps,params)/pdeath(all_temps)-1),"b--",alpha=.5)
        atwin.set_ylim(-.5,-3.2)
        atwin.set_ylabel("Minimum fitness",color="b") #r"log($\frac{p_\mathrm{off,T}}{p_\mathrm{death}}$ - 1)",color="b")
        atwin.tick_params(axis='y', labelcolor="b")
    # plot input TRC
    l_death = a.plot(all_temps,pdeath(all_temps),"k:",alpha=.2,label=r"p$_\mathrm{death}$")

    a.set_ylabel("Fraction")
    a.set_ylim(0,1)
    a.set_xlim(T_range[0],all_temps[-1])
    if a == ax:
        a.legend(loc="upper right")
        a.set_xlabel("Temperature, T (K)")
    elif a == a2[1]:
        lines = l_var+l_var_mut+l_poff+l_death+l_thresh
        labs = [l.get_label() for l in lines]
        a.legend(lines,labs,loc="upper right")
#    elif a == a2[0]:
#        a.legend(loc="upper left")
#    elif a == a2[1]:
#        a.legend(loc="upper right")
#        a.set_xlabel("Temperature, T (K)")

fig.savefig(f"figures/survival_all_experiments_{sample_var}seeds.pdf")

# predict that surival is proportional to poff/pdeath
def prediction(T_range):
    ratio = poff_T(T_range,Tresp)/pdeath(T_range)
    for i in range(len(ratio)):
        if ratio[i] > 3.5:
            ratio[i] = 3.5
    return ratio

plot_prediction = True  #False
if plot_prediction:
    Tresp = [303, 11, -3]
    ax2 = plt.twinx(ax)
    ax2.plot(T_range,np.log(poff_T(T_range,Tresp)/pdeath(T_range)-1),"b--",label=r"log($p_\mathrm{off,T}/p_\mathrm{death}-1)$")
    ax2.set_ylabel(r"log($p_\mathrm{off,T}/p_\mathrm{death}-1)$",color="b")
    #ax2.set_ylim(0) #,1.7) #,1.7) #1,4)
    ax2.set_xlim(T_range[0],T_range[-1])
    ax2.legend(loc="lower right")

    ax2.tick_params(color="b",labelcolor="b")

    fig2,ax2 = plt.subplots()
    ax2.plot(T_range,(singleTRC/50)/(poff_T(T_range,Tresp)/pdeath(T_range)))
    ax2.plot(T_range,np.ones((len(T_range),)),"k:")
    ax2.set_ylabel(r"survival / ($p_\mathrm{death}/p_mathrm{off,T}$)")
    ax2.set_xlabel("Temperature, T (K)")

    # show % viable species for each T
    # show integral over Topt of rmax for each T
    Topt_range = np.r_[265:330:.1] # Topts drawn from this range
    T_range = np.r_[274:320:.3] # experiments run at these Tenv
    pdeath_varies = True # what is your experimental setup?
    skew = -3 
    width = 11
    Tctrl = 303

    #fig,ax = plt.subplots(2,sharex=True)
    i = 0
    exp = 'var-TRC'
    plot_percent_viable = False
    if plot_percent_viable:
    #for exp in ["var-TRC"]:
        i+=1
        a = a2[i].twinx()

        n_viable = []
        ratio_po_pd_sum = []
        for T in T_range: # for each Tenv
            count = 0 # count number of viable species
            ratio_po_pd_tot = 0
            if pdeath_varies:
                death = pdeath(T)
            else: death = 0.2
            for Topt in Topt_range:
                # print(poff_T(T,[Topt,0,0]))
                if exp == "one-TRC":
                    po = poff_T(T,[Tctrl,width,skew]) 
                else: po = poff_T(T,[Topt,width,skew])
                if po > death:
                    count += 1
                    ratio_po_pd_tot += po/death #ff_T(T,[Topt,width,skew])/death
            n_viable.append(count)
            ratio_po_pd_sum.append(ratio_po_pd_tot)

        # ax[0]: viable species
        #if a == a2[1]:
        a.plot(T_range,np.array(n_viable)/len(Topt_range),alpha=.5,label="Viable",color="b")
        #a.plot(T_range,np.array(n_viable)/len(Topt_range)-.75*pdeath(T_range),alpha=.5,label=r"Viable-.75$p_\mathrm{death}$",color="b",linestyle="--")
        a.set_ylim(.25,.58)
        #a.plot(T_range,1/pdeath(T_range),label=r"$\frac{1}{p_\mathrm{death}}$",color="orange")
        #a.plot(T_range,np.array(n_viable)/len(Topt_range)/pdeath(T_range),label=r"$\frac{p_\mathrm{viable}}{p_\mathrm{death}}$",color="g")
        #a.plot(T_range,np.log(np.array(n_viable)/len(Topt_range)/pdeath(T_range)),"--")
        #a.plot(T_range,poff_T(T_range,[Tctrl,width,skew]))
        #if pdeath_varies:
        #    a.plot(T_range,pdeath(T_range),":",label=r"p$_\mathrm{death}$")
        #else: a.plot(T_range,0.2*np.ones(len(T_range)),":",label=r"p$_\mathrm{death}$")
        #a.set_ylim(0)
        a.set_ylabel("Frac. viable spcs",color="b")
        a.tick_params(color="b",labelcolor="b")
        a.tick_params(axis='y', labelcolor="b")
#        a.set_ticklables(color="b")
        #if i == 0:
        #    a.set_ylim(0,1)
        #else:
        #    a.set_ylim(0)
        a.legend(loc="upper right")

letters = iter(["a)","b)","c)","d)","e)","f)","g)","h)","i)","j)","k)","l)"])
for a in a2.flatten():
    letter = next(letters)
    a.text(0.0,1.05,letter,transform=a.transAxes,horizontalalignment="left",verticalalignment="center")

a2[0].set_title("Single TRC")
a2[1].set_title("Various TRC")
f2.savefig("figures/survival_2panes.pdf")

plt.show()

