import sys
import numpy as np
from matplotlib import pyplot as plt
sys.path.append("/home/cfebvre/repos/the_model_y3/src/geoTNM")
from MTE_TPC_combo import *



if __name__ == "__main__":
    print("This is a test modification")

    # TEST ROFF_AMARASEKARE REAL FAST
    import matplotlib.pyplot as plt
    # 1. PLOT TRC FOR SINGLE AND VARIOUS TRC SETUPS
    temps = np.r_[274:320:.1]
    # find Tpeak
    poff_list = poff_T(temps)
    idx = list(poff_list).index(max(poff_list))
    Tpeak = temps[idx]
    print("Tpeak: ",Tpeak)

    # scale poff and pdeath
    poff_scaler = 1
    death_scaler = 1 #.4
    pmut_scaler = 1# 1.4

    plt.plot(temps,poff_list*poff_scaler,label=r"p$_\mathrm{off}$")
    plt.plot([Tpeak,Tpeak],[0,1],"k:")
    plt.plot([temps[0],temps[-1]],[.2,.2],"k:")
    plt.plot(temps,pdeath(temps)*death_scaler,"k--",label=r"p$_\mathrm{death}$")
    plt.plot(temps,pmut(temps)*pmut_scaler,"k",label=r"p$_\mathrm{mut}$")
    plt.plot([temps[0],temps[-1]],[.01,.01],"k:")
    plt.legend(loc="upper left")

    plt.xlabel("Temperature (K)")
    plt.ylabel("Probability")
    plt.ylim(0,1)


    plt.figure()
    plt.plot(temps,pmut(temps)*pmut_scaler)
    plt.plot([Tpeak,Tpeak],[0,1],"k:")
    plt.plot([temps[0],temps[-1]],[.01,.01],"k:")
    plt.ylim(0,.05)

    Topt_range = np.r_[263:330:5]
    cmap = plt.get_cmap("inferno")
    colors = cmap(np.linspace(0,.9,len(Topt_range)))
    plt.figure()
    i = -1
    for Topt in Topt_range:
        i += 1
        Tresp = [Topt,11,-3]
        if i == len(Topt_range)-1:
            plt.plot(temps,poff_T(temps,Tresp),color=colors[i],label=r"p$_\mathrm{off}$")
        else: plt.plot(temps,poff_T(temps,Tresp),color=colors[i])
    plt.plot(temps,pdeath(temps)*death_scaler,"k--",label=r"p$_\mathrm{death}$")
    plt.plot(temps,pmut(temps)*pmut_scaler,"k",label=r"p$_\mathrm{mut}$")
    plt.legend(loc="upper left")

    plt.xlabel("Temperature (K)")
    plt.ylabel("Probability")
    plt.ylim(0,1)

    # 2. OTHER PLOTS
    temps = np.r_[270:320]
    plt.figure()
    i = -1
    cmap = plt.get_cmap('inferno')
    colors = cmap(np.linspace(1,.1,len(np.r_[277:310:3])))
    for Topt in np.r_[277:310:3]:
        i += 1 
        plt.plot(temps, poff_Amarasekare(temps,Topt),color=colors[i],label=f"Topt: {Topt}K")
    #plt.plot(temps,MTE(temps),"--",color="black")
    plt.plot(temps,pdeath(temps),"r",label="pdeath")
    plt.plot([temps[0],temps[-1]],[.2,.2],"k--")
    plt.plot([Topt,Topt],[0,1],"k--")
    plt.plot([temps[0],temps[-1]],[1,1],"k--")
    plt.legend()
    plt.xlabel("Environmental Temperature")
    plt.ylabel("Reproduction Rate,r_off (per capita per generation)")
    #plt.ylim(0,1)
    plt.title("Amarsekare and Savage TPCs for a range of Topts")

    # optimize MTE-env so it's controlled at T=309
    plt.figure()
    scalers = np.r_[1.9:2.1:.05]
    colors = cmap(np.linspace(1,.1,len(scalers)))
    Topt = 309
    offset = 5
    i = -1
    #for scaler in scalers:
        #i += 1
    i = 3
    scaler = 1.95
    plt.plot(temps, scaler*roff_Amarasekare(temps,Topt-offset),color=colors[i],label=f"{scaler}")
    plt.plot(temps,pdeath(temps),"r")
    plt.plot([temps[0],temps[-1]],[.2,.2],"k--")
    plt.legend()
    plt.plot([Topt,Topt],[0,1],"k--")
    plt.plot([temps[0],temps[-1]],[1,1],"k--")

    # TEST ROFF_AMARASEKARE scaled up a bit
    import matplotlib.pyplot as plt
    temps = np.r_[270:320]
    plt.figure()
    i = -1
    cmap = plt.get_cmap('inferno')
    colors = cmap(np.linspace(1,.1,len(np.r_[290:310:3])))
    for Topt in np.r_[290:310:3]:
        i += 1 
        plt.plot(temps, 2*roff_Amarasekare(temps,Topt),color=colors[i],label=f"individual Topt: {Topt}K")
    plt.plot(temps,MTE(temps),"--",color="black")
    plt.legend()
    plt.xlabel("Environmental Temperature")
    plt.ylabel("Reproduction Rate,r_off (per capita per generation)")
    plt.ylim(0,1)
    plt.title("Amarsekare and Savage TPCs x 2")

    # TEST SKEW NORMAL CUVE FOR RANGE OF TREFS
    plt.figure()
    i = -1
    for Tref in np.r_[290:310:3]:
        i += 1
        plt.plot(temps, poff_T(temps,[Tref,width,skew]),color=colors[i],label=f"individual Tref: {Tref}K")
        plt.plot(np.ones(2)*Tref,[0,1],color=colors[i])
    # plot MTE envelope
    plt.xlabel("Environmental Temperature")
    plt.ylabel("Reproduction Rate,r_off (per capita per generation)")
    plt.ylim(0,1)
    plt.title("Skew normal for range of Tref vals")
    plt.legend()

    # PLOT MTE
    plt.figure()
    plt.plot(temps, MTE(temps))
    plt.xlabel("Environmental Temperature")
    plt.ylabel("Whole organism metabolic rate,B (J/s)")
    plt.title("MTE")


    # PLOT MTE * skew normal
    plt.figure()
    envelope = MTE(temps)
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

    plt.show()
