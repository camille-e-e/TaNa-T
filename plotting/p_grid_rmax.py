import numpy as np
import matplotlib.pyplot as plt
from geoTNM import MTE_TPC_combo as MTE

# define probability range
p_range = np.r_[.1:1.1:.1]
T_range = np.r_[260:330]

# define rmax
def poff(T):
    return MTE.poff_T(T)

def pdeath(T):
    return MTE.pdeath(T)

def rmax(T):
    return poff(T) - pdeath(T)

# make grid
grid_rmax = np.zeros((len(p_range),len(p_range)))
grid_T = grid_rmax.copy()

# fill in rmax grid blocks
col = -1
for po in p_range:
    row = len(p_range)
    col += 1
    for pd in p_range:
        row -= 1
        rm = po-pd
        grid_rmax[row,col] = rm

# fill in temperatures of TRC
po_list,pd_list = [],[] # poff,pdeath values of TRC
px_py_list = [] # values of poff*pdeath
for T in T_range:
    po_list.append(30*poff(T))
    pd_list.append(10*pdeath(T))
    rm = rmax(T)
    #print(f"poff: {po}, pdeath: {pd}, T: {T}, rmax:{rm}")
for po in p_range:
    px_py_list.append(po**2)
px_py_list = np.array(px_py_list)
#print(np.shape(px_py_list))
    
    #if pd in p_range and po in p_range:
    #    row = -list(p_range).index(pd)
    #    col = list(p_range).index(po)
        #grid_rmax[row,col] = rm
    #    grid_T[row,col] = T

print("grid_T: ",grid_T)
#print("grid_rmax: ",grid_rmax)

# show grid
cmap = plt.get_cmap('RdPu')
fig,ax = plt.subplots()
fig2,ax2 = plt.subplots()
# this worked before: image = ax.imshow(grid_rmax,cmap=cmap,extent=[0,len(p_range),0,len(p_range)],interpolation='none')
image = ax.imshow(grid_rmax,cmap=cmap,extent=[0,len(p_range),0,len(p_range)],interpolation='none')
#image2 = ax2.imshow(grid_T,cmap=cmap,extent=[0,len(p_range),0,len(p_range)],interpolation='none')
ax.plot(po_list,pd_list,'white',label="TRC")
#ax.plot(10*p_range,10*px_py_list,'white',label="P(surv. & reprod.)")
ax.legend() #loc="lower right")
ax.set_ylim(0,len(p_range)/max(p_range))

# draw gridlines
for a in [ax,ax2]:
    a.set_xlabel(r"$p_{off}$")
    a.set_ylabel(r"$p_{death}$")
    a.grid(which='major',axis='both',linestyle='-',color='k',linewidth=2)

# label ticks
x_labels,y_labels = [],[]
for p in p_range:
    x_labels.append(f"{p:.1f}")
    y_labels.append(f"{p:.1f}")
for a in [ax]: # ,ax2]:
    a.set_xticks(np.r_[1:len(p_range)+1:1])
    a.set_yticks(np.r_[1:len(p_range)+1:1])
    a.set_xticklabels(x_labels)
    a.set_yticklabels(y_labels)

# title plots
ax.set_title(r"r$_{max} = p_{off}-p_{death}$")
ax2.set_title("Temperature values")


cbar = fig.colorbar(image, label=r"$r_{max}$",ax=ax,extend='both')
cbar.minorticks_on()

#cbar = fig2.colorbar(image2, label=r"T (K)",ax=ax2,extend='both')
#cbar.minorticks_on()

fig3,ax3 = plt.subplots()
ax3.plot(np.array(po_list)/10,np.array(pd_list)/10)

plt.show()




