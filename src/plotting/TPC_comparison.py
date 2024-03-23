import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
Tref = 307.5
scaling = 5.5*1.0134*3 # 4.5
width = 10.5
skew = -3
temps = np.r_[274:320:3]
def TPC(T,Tref=Tref):
    return scaling*skewnorm.pdf(T,skew,loc=Tref,scale=width)
def death(T):
    dTr = 1.6*0.05
    Ad = 6000
    TD = 1/294 - 1/T
    return 1 - np.exp(-dTr*np.exp(Ad*TD))
def findTopt(T,Tref=Tref):
    if len(T) < 2:
        print("error, can only find maximum with multiple temeratures")
    Tmin = min(T)
    Tmax = max(T)
    Trange = np.linspace(Tmin,Tmax,1000)
    Tcurve = TPC(Trange,Tref)
    TPCmax = max(Tcurve)
    Topt_idx = list(Tcurve).index(TPCmax)
    Topt = Trange[Topt_idx]
    return Topt

Tref_avg_by_T = [294.5, 296.7, 297.7, 298.7, 300.4, 300.2, 302.9, 304.6, 306.4, 308.6, 310.2, 312.8, 0 , 0, 0, 0]
Topt_avg_by_T = []
for Tref in Tref_avg_by_T:
    Topt_avg_by_T.append(findTopt(temps,Tref))

# FIG 1
fig,ax = plt.subplots()
ax.plot(temps,temps,label="1:1 ratio")
ax.plot(temps,Topt_avg_by_T,label="<Topt>(T)")
ax.legend()
ax.set_xlabel("Temperature,T (K)")
ax.set_ylabel("Average Topt in surviving ecosystems,<Topt> (K)")
ax.set_title("Thermal dependence of average Topt")

# FIG 2
varTPC_survival_by_T = [9,18,34,42,53,58,69,55,53,31,15,4,0,0,0,0]
uniTPC_survival_by_T = [0,0,0,1,7,48,79,87,92,100,81,23,0,0,0,0]
fig,ax = plt.subplots(figsize=(10,6))
ax.set_ylabel("Number of living ecosystems (out of 100)")
ax2 = plt.twinx(ax)
ax2.set_ylabel("Average TPC (semi-transparent curves)")
ax.set_xlabel("Temperature,T (K)")
ax.plot(temps, uniTPC_survival_by_T, "b",label="one TPC")
ax.plot(temps, varTPC_survival_by_T, "r",label="various TPCs")
ax.legend()
ax2.plot(temps, TPC(temps)-death(temps), "b", alpha = .5)
#ax2.plot(temps,death(temps),"black")
i = -1
cmap = plt.get_cmap('hot')
colors = cmap(np.linspace(.1,1,len(temps)))
for Topt in Topt_avg_by_T:
    i+=1
    ax2.plot(temps, TPC(temps,Topt)-death(temps), color=colors[i], alpha=.5, label=f"Tenv: {temps[i]}, <Topt>: {Topt:.1f}")
ax2.legend(bbox_to_anchor=(1.15,1),loc="upper left")
ax2.set_ylim(0)
ax.set_title("Survival of ecosystems with and without TPC variation")
plt.tight_layout()
plt.show()
