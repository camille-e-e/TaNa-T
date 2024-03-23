import sys
import numpy as np
from matplotlib import pyplot as plt
sys.path.append("/home/cfebvre/repos/the_model_y3/src/geoTNM")
from MTE_TPC_combo import *


def find_zeros(args=None):
    def func(temps,args=args):
        return abs(poff_T(temps,args)-pdeath_T(temps))
    res = minimize_scalar(func,args=args)
    if res.success:
        return res.x
    else:
        return "Failed"


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize_scalar
    import string # for labeling a b c d in subplots
    #Tctrl = 307.8
    Tuniv = 303 # Topt of sinlge TRC

    # figure widths
    sin_width = 3.54 # one column in JTB
    doub_width = 7.48 # full width in JTB

    temps = np.r_[274:320]
    sampling_temps = np.r_[274:320:3]
    Topt_range = np.linspace(252,350,20)
    cmap = plt.get_cmap("inferno")
    colors = cmap(np.linspace(0,.9,len(Topt_range)))

    # find Tpeak
    Tresp = [Tuniv,11,-3]
    temps_all = np.r_[264:330:.1]
    TRC = list(poff_T(temps_all,Tresp))
    TRC_max = max(TRC)
    idx = TRC.index(TRC_max)
    T_peak = temps_all[idx]
    print("T_peak: ",T_peak)
    Tctrl = T_peak

    # plot setup of single TRC, var-TRC and MTE-env
    # ------------------------------------
    fig,ax = plt.subplots(3,1,figsize=(sin_width,2.5*sin_width),sharex=True)
    # show sampling temperatures
    for a in ax:
        for T in sampling_temps:
            a.plot([T,T],[0,1],"b:",alpha=.3)

    # single TRC setup in first pane
    ax[0].set_ylim(0,1)
    ax[0].plot(temps,poff_T(temps,Tresp),"k",label=r"p$_\mathrm{off,T}$")
    ax[0].plot(temps,pdeath(temps),"k--",label=r"p$_\mathrm{death}$")
    ax[0].set_ylabel("Probability")
    ax[0].set_xlim(min(temps),max(temps))

    # var TRC in second pane
    i = -1
    for Topt in Topt_range:
        i += 1
        ax[1].plot(temps,poff_T(temps,[Topt,11,-3]),c=colors[i])
    ax[1].plot(temps,pdeath(temps),"k--")
    ax[1].set_ylabel("Probability")
    ax[1].set_ylim(0,1)

    # mutation in last pane
    ax[2].plot(temps,pmut(temps),"k-.",label=r"p$_\mathrm{mut}$")
    ax[2].set_ylabel("Probability")
    ax[2].legend(loc="center left")
    ax[2].set_xlabel("Temperature (K)")
    ax[2].set_ylim(0,max(pmut(temps)))

    # T ctrl
    for a in ax:
        if a == ax[0]:
            a.plot([T_peak,T_peak],[0,1],"k:",label=r"T$_\mathrm{ctrl}$")
            a.legend(loc="center left")
        else: 
            a.plot([T_peak,T_peak],[0,1],"k:")

    # label A B C
    ax[0].text(0.05,0.9,"a)",transform=ax[0].transAxes,horizontalalignment="left",verticalalignment="center")
    ax[1].text(0.05,0.9,"b)",transform=ax[1].transAxes,horizontalalignment="left",verticalalignment="center")
    ax[2].text(0.05,0.9,"c)",transform=ax[2].transAxes,horizontalalignment="left",verticalalignment="center")

    plt.tight_layout()
    plt.savefig("figures/single-var-TRC-setup.pdf")

    # another version of single-TRC and var-TRC setup
    # -------------------------------------------------
    fig,ax = plt.subplots(1,2,sharey=True,sharex=True,figsize=(doub_width,.7*doub_width))

    # show sampling temperatures
    # for a in ax:
    #     for T in sampling_temps:
    #         a.plot([T,T],[0,1],"b:",alpha=.3)

    # single TRC setup in first pane
    ax[0].set_ylim(0,1)
    ax[0].plot(temps,poff_T(temps,Tresp),"k",label=r"p$_\mathrm{off,T}$")
    ax[0].plot(temps,pdeath(temps),"k--",label=r"p$_\mathrm{death}$")
    ax[0].set_ylabel("Probability")
    ax[0].set_xlabel("Temperature, T (K)")
    ax[0].set_xlim(min(temps),max(temps))

    # var TRC in second pane
    i = -1
    for Topt in Topt_range:
        i += 1
        ax[1].plot(temps,poff_T(temps,[Topt,11,-3]),c=colors[i])
    ax[1].plot(temps,pdeath(temps),"k--")
    ax[1].set_xlabel("Temperature,T (K)")
    ax[1].set_ylim(0,1)

    # mutation in last pane
    for a in ax:
        a.plot(temps,pmut(temps),"g-.",label=r"p$_\mathrm{mut}$")

    # T ctrl
    for a in ax:
        if a == ax[0]:
            a.plot([T_peak,T_peak],[0,1],"k:",label=r"T$_\mathrm{ctrl}$")
            a.legend(loc="center left")
        else: 
            a.plot([T_peak,T_peak],[0,1],"k:")

    # label A B C
    ax[0].text(0.05,0.9,"a)",transform=ax[0].transAxes,horizontalalignment="left",verticalalignment="center")
    ax[1].text(0.05,0.9,"b)",transform=ax[1].transAxes,horizontalalignment="left",verticalalignment="center")

    plt.tight_layout()
    plt.savefig("figures/single-var-TRC-2panes.pdf")

    # find ratio between poff and pdeath in single TRC
    # -------------------------------------------------
    fig,ax = plt.subplots(figsize=(doub_width,.7*doub_width))
    # single TRC setup in first pane
    ax.set_ylim(0,1)
    ax.plot(temps,poff_T(temps,Tresp),"k",label=r"p$_\mathrm{off,T}$")
    ax.plot(temps,pdeath(temps),"k--",label=r"p$_\mathrm{death}$")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Temperature, T (K)")
    ax.set_xlim(min(temps),max(temps))
    ax.legend()
    ax2 = plt.twinx(ax)
    ax2.plot(temps,poff_T(temps,Tresp)/pdeath(temps),"r")
    ax2.set_ylabel(r"$p_\mathrm{off,T}/p_\mathrm{death}$",color="r")
    ax2.set_ylim(1)

    # Find x-intercepts
    # -------------------------------------------------
    """
    plt.figure()
    Topt_range = np.r_[290:310:3]
    Twidth_range = np.r_[8:14]
    i = -1
    cmap = plt.get_cmap('inferno')
    colors = cmap(np.linspace(.1,1,len(Topt_range)))
    Twidth = 11
    for Topt in Topt_range:
        i += 1 
        plt.plot(temps, poff_T(temps,[Topt,Twidth,0]),color=colors[i],label=f"s: {Twidth}")
        # find intercepts
        x = find_zeros(poff_T,args=(temps,[Topt,Twidth,0]))
        if np.shape(x) != (1,):
            for x1 in x:
                plt.scatter(x1,0,"k*")
        else: plt.scatter(x,0,"k*")
    plt.legend()
    plt.xlabel("Environmental Temperature")
    plt.ylabel("Reproduction probability")
    plt.ylim(0)
    plt.title("skewnorm maximum reproduction")
    """

    # Plot roff_AMARASEKARE for a range of Topt
    # ----------------------
    plt.figure()
    i = -1
    cmap = plt.get_cmap('inferno')
    colors = cmap(np.linspace(.1,1,len(np.r_[290:310:3])))
    for Topt in np.r_[290:310:3]:
        i += 1 
        plt.plot(temps, roff_Amarasekare(temps,Topt),color=colors[i],label=f"Topt: {Topt}K")
    plt.plot(temps,MTE(temps),"--",color="black",label="MTE")
    plt.plot(temps,pdeath(temps),":k",label=r"$p_\mathrm{death}$")
    plt.plot(temps,2*pdeath(temps),":r",label=r"2 $p_\mathrm{death}$")
    plt.legend()
    plt.xlabel("Environmental Temperature (K)")

    plt.ylabel(r"Reproduction Rate,$r_\mathrm{off}$")
    plt.ylim(0,1)
    plt.title("Amarsekare and Savage TPCs for a range of Topts")

    # plot roff_Amarasekare for different widths
    # ------------------------------------------
    plt.figure()
    i = -1
    cmap = plt.get_cmap('inferno')
    colors = cmap(np.linspace(.1,1,len(np.r_[290:310:3])))
    Topt = 300
    for Twidth in np.r_[11:16]:
        i += 1 
        plt.plot(temps, poff_T(temps,[Topt,Twidth,0]),color=colors[i],label=f"s: {Twidth}")
    plt.legend()
    plt.xlabel("Environmental Temperature")
    plt.ylabel("Reproduction probability")
    plt.ylim(0)
    plt.title("skewnorm maximum reproduction")

    # Plot SKEW NORMAL CUVE FOR RANGE OF TREFS
    # -----------------------------------------
    plt.figure()
    i = -1
    for Tref in np.r_[290:310:3]:
        i += 1
        plt.plot(temps, poff_T(temps,[Tref,width,skew]),color=colors[i],label=f"individual Tref: {Tref}K")
        plt.plot(np.ones(2)*Tref,[0,1],color=colors[i])
    plt.xlabel("Environmental Temperature")
    plt.ylabel("Reproduction Rate,r_off (per capita per generation)")
    plt.ylim(0,1)
    plt.title("Skew normal for range of Tref vals")
    plt.legend()

    # PLOT MTE * skew normal
    # ---------------------
    plt.figure()
    envelope = MTE(temps) # ** is this the correct method? **
    i = -1
    for To in np.r_[290:310:3]:
        i += 1
        Tref = To #+5
        plt.plot(temps, envelope*poff_T(temps,[Tref,width,skew]),color=colors[i],label=f"skewnorm Tref: {Tref}K")
        #plt.plot(temps, roff_Amarasekare(temps,Topt),"--",color=colors[i],label=f"Amarasekare Topt: {Topt}K")
    plt.plot(temps, envelope,"--k",label="2.5*B(T)")
    plt.plot(temps, pdeath(temps),":k",label="pdeath")
    plt.legend()
    plt.xlabel("Environmental Temperature")
    plt.ylabel("Reproduction Rate,r_off (per capita per generation)")
    plt.ylim(0,1)
    plt.title("MTE * skew normal curve")

    # show % viable species for each T
    # -------------------------------
    # show integral over Topt of rmax for each T
    Topt_range = np.r_[265:330:.1] # Topts drawn from this range
    T_range = np.r_[274:320:3] # experiments run at these Tenv
    pdeath_varies = True # what is your experimental setup?
    skew = -3 
    width = 11

    # do this for single TRC, var TRC and MTE-env
    for exp in ["one-TRC","var-TRC","MTE-env"]:
        fig,ax = plt.subplots(3,sharex=True)
        n_viable = []
        rmax_sum = []
        ratio_po_pd_sum = []
        for T in T_range: # for each Tenv
            count = 0 # count number of biable species
            rmax_tot = 0
            ratio_po_pd_tot = 0
            if exp == "MTE-env" or pdeath_varies:
                death = pdeath(T)
            else: death = 0.2
            for Topt in Topt_range:
                # print(poff_T(T,[Topt,0,0]))
                if exp == "one-TRC":
                    po = poff_T(T,[Tctrl,width,skew]) 
                else: po = poff_T(T,[Topt,width,skew])
                if po > death:
                    count += 1
                    rmax_tot += po-death #ff_T(T,[Topt,width,skew])-death
                    ratio_po_pd_tot += po/death #ff_T(T,[Topt,width,skew])/death
            n_viable.append(count)
            rmax_sum.append(rmax_tot)
            ratio_po_pd_sum.append(ratio_po_pd_tot)

        # ax[0]: viable species
        ax[0].plot(T_range,np.array(n_viable)/len(Topt_range))
        if exp == "MTE-env" or pdeath_varies:
            ax[0].plot(T_range,pdeath(T_range),":",label=r"p$_\mathrm{death}$")
        else: ax[0].plot(T_range,0.2*np.ones(len(T_range)),":",label=r"p$_\mathrm{death}$")
        ax[0].set_ylabel("Frac. viable spcs",color="b")
        ax[0].set_ylim(0,1)
        ax[0].legend()

        # ax[1]: sum of rmax for all possible specea at each T
        ax[1].plot(T_range,rmax_sum)
        ax[1].set_ylabel("Tot max growth",color="b")
        axtwin = plt.twinx(ax[1])
        axtwin.plot(T_range,np.array(rmax_sum)/np.array(n_viable),"r")
        axtwin.set_ylabel("Average",color="r")
        ax[1].set_ylim(0)
        axtwin.set_ylim(0)
        
        ax[2].plot(T_range,ratio_po_pd_sum)
        ax[2].set_ylabel(r"$\sum$ $p_\mathrm{off}$ / $p_\mathrm{death}$",color="b")
        axtwin = plt.twinx(ax[2])
        axtwin.plot(T_range,np.array(ratio_po_pd_sum)/np.array(n_viable),"r")
        axtwin.set_ylabel("Average",color="r")


        ax[-1].set_xlabel("Temperature (K)")

        ax[0].set_title(exp)

    # PLOT Contour plot of poff(i,T)
    # ------------------------------
    #temps = np.r_[275:320]
    fitnesses = np.r_[-5:5:.2]
    # DEFAULT TRESPONSE (SINGLE TRC)
    Tresp = [Tuniv,11,-3]

    # define ecological fitness from original TNM
    def poff_i(fitness):
        return 1/(1+np.exp(-fitness))

    # make contour plot of poff for a range of fitnesses and temperatures
    poff_all = np.zeros((len(fitnesses),len(temps)))
    poff_pdeath_all = np.zeros((len(fitnesses),len(temps)))
    row = -1
    for f in fitnesses:
        row += 1
        col = -1
        for T in temps:
            col += 1
            # probability of reproducting depending on ecology and evolution
            poff_all[row,col] = poff_T(T,Tresp)*poff_i(f)
            # probability of surviving and reproducing
            poff_pdeath_all[row,col] = poff_T(T,Tresp)*poff_i(f) - pdeath(T)
    
    # contour plot of poff
    # --------------------
    plt.figure(figsize=(sin_width,1.2*sin_width))
    plt.contourf(temps,fitnesses,poff_all,cmap=plt.get_cmap("Greys"),levels=np.linspace(0,1,11))
    plt.ylabel(r"Fitness, $f_i$")
    plt.xlabel("Temperature,T (K)")
    #plt.title("Reproduction prob., p_off")
    cb = plt.colorbar(label=r"p$_\mathrm{off,total}$",location="bottom")
    cb.ax.plot( [.2,.2],[0,1], 'b--') # my data is between 0 and 1
    plt.contour(temps,fitnesses,poff_all,[.2],colors=["b"],linestyles=["--"])
    plt.plot([Tctrl,Tctrl],[-5,5],"r:",label=r"T$_\mathrm{ctrl}$")
    plt.legend(loc="lower left")
    plt.ylim(-5,4.8)
    plt.tight_layout()
    plt.savefig("figures/poff_contours.pdf")

    # PLOT Am Sav to TTNM translation
    # ------------------------------
    #temps = np.r_[270:320]
    cmap = plt.get_cmap("Greys")
    colors = cmap(np.linspace(.2,1,3))
    fig,ax = plt.subplots(2,2,figsize=(doub_width,.8*doub_width),sharex=True,sharey=True)

    # Top left: Am Sav birth and death
    ax[0,0].plot(temps,roff_Amarasekare(temps),color=colors[2],label=r"r$_\mathrm{birth}$")
    ax[0,0].plot(temps,pdeath(temps),color=colors[1],label=r"r$_\mathrm{death}$") 
    ax[0,0].set_ylabel("Life events")
    # Bottom left: Am Sav rmax
    ax[1,0].plot(temps,roff_Amarasekare(temps)-pdeath(temps,dTr=0.05),color="g",label=r"r$_\mathrm{max}$") #f"Ad: {Ad}")
    ax[1,0].set_ylabel("Population growth") #$r_{max}$") #" (indiv./lifetime)")

    # Top right: TTNM poff & pdeath
    ax[0,1].plot(temps,poff_T(temps,Tresp),"--",color=colors[2],label=r"p$_\mathrm{off,T}$")
    ax[0,1].plot(temps,pdeath(temps,),"--",color=colors[1],label=r"p$_\mathrm{death}$")
    ax[0,1].plot([Tctrl,Tctrl],[0,1],"k:",label=r"T$_\mathrm{ctrl}$")

    # Bottom right: TTNM max growth
    ax[1,1].plot(temps,poff_T(temps,Tresp)-pdeath(temps),"--",color="g",label=r"r$_\mathrm{max,TTNM}$")
    ax[1,1].plot([Tctrl,Tctrl],[0,1],"k:")

    # adjust axes
    ax[0,0].set_ylim(0,1)
    ax[1,0].set_ylim(0,1)

    # label axes
    ax[1,0].set_xlabel("Temperature,T (K)")
    ax[1,1].set_xlabel("Temperature,T (K)")

    for a in ax.flatten():
        a.legend(loc="upper left")

    for n,a in enumerate(ax.flatten()):
        a.text(-0.1, 1.1, string.ascii_uppercase[n], transform=a.transAxes,size=10) 

    plt.tight_layout()
    
    plt.savefig("figures/Am_Sav_setup.pdf")

    # am sav equation, skewnorm approx, and scaled up version
    # -------------------------------------------------------
    fig,ax = plt.subplots()
    # am sav
    ax.plot(temps,roff_Amarasekare(temps),color="k",label=r"$r_\mathrm{off}$ (Am. Sav.)")
    ax.plot(temps,1/1.4*pdeath(temps),color=colors[1],label=r"r$_\mathrm{death}$ (am. Sav.)") 

    # skew norm approx
    Tresponse = [303,11,-3]
    ax.plot(temps,1/3*poff_T(temps,Tresponse),"r--",label=r"$r_\mathrm{off}$ (skewnorm)")

    # scaled up
    ax.plot(temps,poff_T(temps,Tresponse),"k-.",label=r"$p_\mathrm{off,T}$")
    ax.plot(temps,pdeath(temps),"-.",color=colors[1],label=r"$p_\mathrm{death}$")

    # T ctrl
    ax.plot([Tctrl,Tctrl],[0,1],"k:") #,label=r"$T_\mathrm{peack}$")
    ax.plot([temps[0],temps[-1]],[.2,.2],"k:",alpha=.5)

    ax.set_xlabel("Temperature, T (K)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0,1)
    ax.set_xlim(temps[0],temps[-1])
    ax.legend()

    plt.savefig("figures/amsav_approx_scaledup.pdf")

    # Predict survival in single-TRC, var-TRC, and MTE-env
    # ---------------------------------------------------
    fig,ax = plt.subplots(3)
    for a in ax:
        a.set_ylabel(r"Min. f$_i$")
    ax[-1].set_xlabel("Temperature (K)")
    all_temps = np.r_[274:480:.01]
    
    # single TRC
    ax[0].plot(all_temps,-np.log(poff_T(all_temps)/pdeath(all_temps) -1))

    for Topt in np.r_[274:480:3]:
        params = [Topt,11,-3]
        # var TRC
        ax[1].plot(all_temps,-np.log(poff_T(all_temps,params)/pdeath(all_temps)-1)) #,"b--",alpha=.5)
        # MTE env
        ax[2].plot(all_temps,-np.log(roff_Amarasekare(all_temps,Topt)/pdeath(all_temps)-1)) #,"b--",alpha=.5)

    for a in ax:
        a.invert_yaxis()
    ax[1].set_ylim(1)
    ax[2].set_ylim(1)

    # Predict survival in threshold daisy experiments when all species have same death probability
    # ----------------------------------------------
    fig,ax = plt.subplots(2,sharex=True)
    ax[0].set_title("Constant death rate")
    ax[1].set_title("Death varies with T")
    for a in ax:
        a.set_ylabel(r"Min. f$_i$")
    ax[-1].set_xlabel("Temperature (K)")
    all_temps = np.r_[274:320:.01]
    
    for Topt in np.r_[284.3,295.7]: #[274:320:3]:
        if Topt < 290:
            color = "blue"
        else: color = "r"
        params = [Topt,11,-3]
        ax[0].plot(all_temps,-np.log(poff_T(all_temps,params)/pdeath(303)-1),color=color) # pdeath(303) = pdeath_ctrl = 0.2
        ax[1].plot(all_temps,-np.log(poff_T(all_temps,params)/pdeath(all_temps)-1),color=color) # pdeath(303) = pdeath_ctrl = 0.2

    for a in ax:
        a.invert_yaxis()
        a.set_ylim(1)
    fig.tight_layout()

    plt.show()
