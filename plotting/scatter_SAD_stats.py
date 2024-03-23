import numpy as np
import matplotlib.pyplot as plt

#single_stats = np.load("npy_files/final_stats_single-TRC_Jun_21_23.npy",allow_pickle=True).item()
#var_stats = np.load("npy_files/final_stats_var-TRC_Apr_13_23.npy",allow_pickle=True).item()
single_N_by_T = np.load("npy_files/N_by_T_single-TRC_Jun_21_23.npy",allow_pickle=True)
single_D_by_T = np.load("npy_files/D_by_T_single-TRC_Jun_21_23.npy",allow_pickle=True)
single_coreN_by_T = np.load("npy_files/core_N_by_T_single-TRC_Jun_21_23.npy",allow_pickle=True)
single_coreD_by_T = np.load("npy_files/core_D_by_T_single-TRC_Jun_21_23.npy",allow_pickle=True)
single_survival_by_T = np.load("npy_files/survival_single-TRC_Jun_21_23.npy",allow_pickle=True)
single_SAD_by_T = np.load("npy_files/SAD_by_T_single-TRC_Jun_21_23.npy",allow_pickle=True)
single_skew_by_T = np.load("npy_files/skew_by_T_single-TRC_Jun_21_23.npy",allow_pickle=True) 
var_N_by_T = np.load("npy_files/N_by_T_var-TRC_Apr_13_23.npy",allow_pickle=True)
var_D_by_T = np.load("npy_files/D_by_T_var-TRC_Apr_13_23.npy",allow_pickle=True)
var_coreN_by_T = np.load("npy_files/core_N_by_T_var-TRC_Apr_13_23.npy",allow_pickle=True)
var_coreD_by_T = np.load("npy_files/core_D_by_T_var-TRC_Apr_13_23.npy",allow_pickle=True)
var_survival_by_T = np.load("npy_files/survival_var-TRC_Apr_13_23.npy",allow_pickle=True)
var_SAD_by_T = np.load("npy_files/SAD_by_T_var-TRC_Apr_13_23.npy",allow_pickle=True)
var_skew_by_T = np.load("npy_files/skew_by_T_var-TRC_Apr_13_23.npy",allow_pickle=True) 

T_range = np.r_[274:320:3]

sin_width,doub_width = 3.54,7.48 # standard fig sizes in JTB
gridspec = {'width_ratios':[1,1,1,1,.08,.08]} # ,'height_ratios':[1,1,1]}

# SADs
ff,aa = plt.subplots(2,2,sharey=False,sharex=True,figsize=(doub_width,.8*doub_width))
for a in aa[1,:]:
    a.set_xlabel("Log population")
aa[0,0].set_ylabel("PDF")
aa[1,0].set_ylabel("CDF")
aa[0,0].set_title("Single-TRC")
aa[0,1].set_title("Various-TRC")
letters = iter(["a)","b)","c)","d)","e)","f)","g)","h)","i)","j)","k)","l)"])
for a in aa.flatten():
    letter = next(letters)
    a.text(0.09,0.9,letter,transform=a.transAxes,horizontalalignment="left",verticalalignment="center")

f0,a0 = plt.subplots(2,2,sharey="row",figsize=(doub_width,.8*doub_width))
density = True
if density:
    #for a in a0[1,:]:
    #    a.set_ylim(0,5)
    survival = 1
bins = 20
a0[0,0].set_title("Single-TRC")
a0[0,1].set_title("Various-TRC")
a0[0,0].set_ylabel("CDF of cloud")
a0[1,0].set_ylabel("CDF of core")
a0[1,0].set_xlabel("Log population")
a0[1,1].set_xlabel("Log population")

letters = iter(["a)","b)","c)","d)","e)","f)","g)","h)","i)","j)","k)","l)"])
for a in a0.flatten():
    letter = next(letters)
    a.text(0.05,0.9,letter,transform=a.transAxes,horizontalalignment="left",verticalalignment="center")

# SAD skewness
fig,ax = plt.subplots(2,6,figsize=(1.4*doub_width,.75*doub_width),gridspec_kw=gridspec) # SAD skew plots
ax[0,0].set_title("Single-TRC")
ax[0,1].set_title("Single-TRC")
ax[0,2].set_title("Various-TRC")
ax[0,3].set_title("Various-TRC")

ax[0,0].set_ylabel("Skewness of SAD")
ax[1,0].set_ylabel("Skewness of SAD")

for a in [ax[0,0],ax[0,2]]:
    a.set_xlabel("Abudnance")
    a.set_xlim(0,3000)
    a.set_ylim(-1,9.5)
for a in [ax[0,1],ax[0,3]]:
    a.set_xlabel("Species Richness")
    a.set_xlim(0,120)
    a.set_ylim(-1,9.5)
for a in ax[1,:4]:
    a.set_xlabel("Temperature (K)")
    a.set_xlim(T_range[0]-1,T_range[-1])
    a.set_ylim(-1,9.5)

letters = iter(["a)","b)","c)","d)","e)","f)","g)","h)","i)","j)","k)","l)"])
for a in ax.flatten():
    if a not in [ax[0,4],ax[0,5],ax[1,4],ax[1,5]]:
        letter = next(letters)
        a.text(0.05,0.9,letter,transform=a.transAxes,horizontalalignment="left",verticalalignment="center")
ax[0,5].remove()

f2,a2 = plt.subplots(4,2,sharex='col',sharey='row',figsize=(doub_width,1.2*doub_width)) # arrhenius plots
k = 8.6e-5 #eV
#for a in a2.flatten():
#    a.set_yscale("log")
a2[0,0].set_title("Single TRC")
a2[0,1].set_title("Various TRC")
a2[0,0].set_ylabel("Log(med. abundance)")
a2[1,0].set_ylabel("Log(med. sp. richness)")
a2[2,0].set_ylabel("Log(med. core abundance)")
a2[3,0].set_ylabel("Log(med. core sp. richness)")
for a in a2[3,:]:
    a.set_xlabel(r"1/kT (eV$^{-1}$)")
    a.set_xlim(1/k/T_range[-1],1/k/T_range[0])
for a3 in a2[0,:]:
    a4 = a3.twiny()
    a4.set_xlim(1/k/T_range[-1],1/k/T_range[0])
    T2_ticklabels = np.r_[280:320:10]
    T_tick_loc = 1/k/T2_ticklabels
    a4.set_xticks(T_tick_loc)
    a4.set_xticklabels(T2_ticklabels) #,color="r")
    a4.set_xlabel("Temperature, T (K)") #,color="r")

letters = iter(["a)","b)","c)","d)","e)","f)","g)","h)","i)","j)","k)","l)"])
for a in a2.flatten():
    letter = next(letters)
    a.text(0.05,0.9,letter,transform=a.transAxes,horizontalalignment="left",verticalalignment="center")

cmap = plt.get_cmap("inferno")
colors = cmap(np.linspace(0,.9,len(T_range)))
cmap_D = plt.get_cmap("cool")
cmap_N = plt.get_cmap("viridis")

i = -1
for exp in ["single-TRC","var-TRC"]:
    i+=1
    col = 2*i
    
    if i == 0:
        N_by_T = single_N_by_T
        D_by_T = single_D_by_T
        coreN_by_T = single_coreN_by_T
        coreD_by_T = single_coreD_by_T
        survival_by_T = single_survival_by_T
        SAD_by_T = single_SAD_by_T
        skew_by_T = single_skew_by_T
    else: 
        N_by_T = var_N_by_T
        D_by_T = var_D_by_T
        coreN_by_T = var_coreN_by_T
        coreD_by_T = var_coreD_by_T
        survival_by_T = var_survival_by_T
        SAD_by_T = var_SAD_by_T
        skew_by_T = var_skew_by_T
        #medN_by_T = var_stats['med_N']
        #medD_by_T = var_stats['med_D']

    medN_by_T = []
    medD_by_T = []
    medcoreN_by_T = []
    medcoreD_by_T = []
    ii = -1
    for T in T_range:
        ii += 1
        medN_by_T.append(np.nanquantile(N_by_T[ii],.5,axis=0))
        medD_by_T.append(np.nanquantile(D_by_T[ii],.5,axis=0))
        medcoreN_by_T.append(np.nanquantile(coreN_by_T[ii],.5,axis=0))
        medcoreD_by_T.append(np.nanquantile(coreD_by_T[ii],.5,axis=0))

    #binwidth = 25
    #binmax = 1000
    c = -1
    for T in T_range[:-2]:
        c+=1
        color = colors[c]

        # define colors 
        div_this_T = np.array(D_by_T)[c,:,-2]
        ab_this_T = np.array(N_by_T)[c,:,-2]
        skew_now = skew_by_T[c]

        # remove nans
        # indices where skew_now is not nan
        idx = np.where(np.isnan(ab_this_T)==False)
        skew_now = skew_now[idx] #np.logical_not(np.isnan(skew_now))]
        print("Length of skew_now: ",len(skew_now))
        #N_now = np.array(N_by_T)[c,:,-2]
        N_now = ab_this_T[idx] #np.logical_not(np.isnan(skew_by_T[c]))]
        #D_now = np.array(D_by_T)[c,:,-2]
        D_now = div_this_T[idx] #np.logical_not(np.isnan(skew_by_T[c]))]

        if len(N_now) < 1:
            continue

        # normalize for colors
        div_normed = D_now/max(D_now) # div_this_T/max(div_this_T)
        ab_normed = N_now/max(N_now) #ab_this_T/max(ab_this_T)
        div_colors = cmap_D(1-div_normed)
        ab_colors = cmap_N(ab_normed)
        """if sum(ab_colors) < 1:
            print("error: ab_colors all zero at ",T)
            if max(ab_this_T) > 0:
                print("max(ab_this_T):",max(ab_this_T))
            else: print(ab_this_T)
"""
        
        ## remove nans
        #div_colors = div_colors[idx]
        #ab_colors = ab_colors[idx]

        if not density:
            survival = survival_by_T[c]

        # SADs
        threshold = 33 # population which we consider core
        SAD_now = np.array(SAD_by_T[c])/survival
        print("Temperature: ",T_range[c])
        #print(SAD_now)
        # eliminate nans
        SAD_now = SAD_now[np.logical_not(np.isnan(SAD_now))]
        if np.nan in SAD_now: print("ERROR!! np.nan DETECTED!!")
        # if there are more than 1 value of non-nan species in the SAD:
        print(np.shape(SAD_now))
        if len(SAD_now) > 1: # np.shape(SAD_now)[0] > 0:
            SAD_nonzero = SAD_now[SAD_now > 0]
            SAD_log = np.log(SAD_nonzero)
            aa[0,i].hist(SAD_log,bins=bins,histtype="step",cumulative=0,density=density,color=color,label=f"T={T}K")
            aa[1,i].hist(SAD_log,bins=bins,histtype="step",cumulative=1,density=density,color=color,label=f"T={T}K")

            # sort SAD to break apart cloud and core
            SAD_sorted = np.sort(SAD_now)
            # if there are any core species, collect in own list
            if SAD_sorted[-1] > threshold:
                idx = list(map(lambda i: i> threshold, SAD_sorted)).index(True)
                #print("index: ",idx)
                SAD_cloud = SAD_sorted[:idx]
                SAD_core = SAD_sorted[idx:]
                SAD_core_log = np.log(SAD_core)
                SAD_core_log = SAD_core_log[np.isfinite(SAD_core_log)]
                # core
                #out = aa.hist(SAD_core_log,bins=bins,histtype="step",density=density,color=color,label=f"T={T}K")
                #binwidth = out[1][1] - out[1][0]
                a0[1,i].hist(SAD_core_log,bins=bins,histtype="step",cumulative=True,density=density,color=color,label=f"T={T}K")
                #a0[1,i].plot(out[1][1:]-binwidth/2,out[0],color=color,label=f"T={T}K")
            else: SAD_cloud = SAD_now
            SAD_cloud_log = np.log(SAD_cloud)
            SAD_cloud_log = SAD_cloud_log[np.isfinite(SAD_cloud_log)]
            # cloud
            a0[0,i].hist(SAD_cloud_log,bins=bins,cumulative=True,density=density,histtype='step',color=color,label=f"T={T_range[c]}K")
            #out = aa.hist(SAD_cloud_log,bins=bins,density=density,histtype='step',color=color,label=f"T={T_range[c]}K")
            #binwidth = out[1][1]-out[1][0]
            #a0[0,i].plot(out[1][1:],out[0],color=colors[c],label=f"T={T}")

        # SKEWNESS
        # skew vs N
        #skew_now = skew_now[skew_now != np.nan]
        ax[0,col].scatter(N_now,skew_now,color=color,label=f"T={T_range[i]}K")
        # skew vs D
        ax[0,col+1].scatter(D_now,skew_now,color=color,label=f"T={T_range[i]}K")
        # skew vs T
        ax[1,col+1].scatter(np.ones(len(skew_now))*T,skew_now,color=div_colors,label=f"T={T_range[c]}K")
        ax[1,col].scatter(np.ones(len(skew_now))*T,skew_now,color=ab_colors,label=f"T={T_range[c]}K")

        if T in [289,298, 302]:
            print(f"T: {T} colors: ", ab_colors)
            """
            print(f"Temp: {T}, len(skew_now: {len(skew_now)}")
            print(skew_now)
            print(N_now)
            print(D_now)
            wf,wa = plt.subplots(3)
            wa[0].plot(skew_now)
            wa[0].set_ylabel("skew")
            wa[1].plot(N_now)
            wa[1].set_ylabel("N")
            wa[2].plot(D_now)
            wa[2].set_ylabel("D")
            wa[0].set_title(f"T:{T}")

            wf,wa = plt.subplots(2,2)
            wa[0,0].set_title(f"T:{T}")
            wa[0,0].scatter(N_now,D_now)
            wa[0,0].set_xlabel("N")
            wa[0,0].set_ylabel("D")
            wa[0,1].scatter(N_now,skew_now)
            wa[0,1].set_xlabel("N")
            wa[0,1].set_ylabel("skew")
            wa[1,1].scatter(D_now,skew_now)
            wa[1,1].set_xlabel("D")
            wa[1,1].set_ylabel("skew")
            wa[1,1].scatter(T*np.ones(len(skew_now)),skew_now)
            wa[1,1].set_xlabel("T")
            wa[1,1].set_ylabel("skew")
            """
        """
        # plot each T on its own plot
        plt.figure()
        plt.scatter(np.ones(len(skew_now))*T,skew_now,color=ab_colors,label=f"T={T_range[c]}K")
        plt.title(f"T={T}K")
        """
    # colorbars
    plt.figure()
    ar = np.r_[0:1.01:.01]
    mat = [ar,ar]
    Tmat = [T_range,T_range]
    im1 = plt.imshow(Tmat,cmap="inferno")
    im2 = plt.imshow(mat,cmap="cool")
    im3 = plt.imshow(mat,cmap="viridis")
    cb1 = fig.colorbar(im1,cax=ax[0,4]) #,ticks=[0,1],label="Relative temperature")
    cb2 = fig.colorbar(im2,cax=ax[1,5],ticks=[0,1],label="Relative species richness")
    cb2.ax.invert_yaxis()
    cb3 = fig.colorbar(im3,cax=ax[1,4],ticks=[0,1],label="Relative abundance")
    cb2.ax.set_yticklabels(['1','0'])
    #cb3.ax.set_yticklabels(['min','max'])
    cb1.set_label("Temperature (K)",labelpad=2)
    cb2.set_label("Relative species richness",labelpad=-4)
    cb3.set_label("Relative abundance",labelpad=-4)

    # Arrhenius plots
    k = 8.6e-5 #eV
    arr_T = 1/k/T_range
    cmap2 = plt.get_cmap("Greys")
    colors2 = cmap2(np.linspace(.2,1,5))

    """
    # test plot
    tf,ta = plt.subplots(3)
    ta[0].plot(arr_T)
    ta[1].plot(np.array(medN_by_T)[:,-1])
    ta[2].plot(np.mean(np.array(medN_by_T)[:,-10:-1],axis=1))
    a2[0,0].plot([1,2,3])
    plt.show()"""
    
    j = -1
    # make an Arrhenius line at each of these times:
    for t in [1,10,100,1000,9900]:
        j += 1
        # N
        a2[0,i].plot(arr_T,np.log(np.mean(np.array(medN_by_T)[:,t-1:t+10],axis=1)),color=colors2[j],label=f"t={t}")
        # D
        a2[1,i].plot(arr_T,np.log(np.mean(np.array(medD_by_T)[:,t-1:t+10],axis=1)),color=colors2[j],label=f"t={t}")
        # core N
        a2[2,i].plot(arr_T,np.log(np.mean(np.array(medcoreN_by_T)[:,t-1:t+10],axis=1)),color=colors2[j],label=f"t={t}")
        # core D
        a2[3,i].plot(arr_T,np.log(np.mean(np.array(medcoreD_by_T)[:,t-1:t+10],axis=1)),color=colors2[j],label=f"t={t}")

a2[0,0].legend()

def expectation(x_range=np.r_[-36,42],m=-0.6,b=10):
    return b + m*x_range
x_range = np.r_[36,43]
a2[0,0].plot(x_range,expectation(x_range,b=28),"r--")
a2[0,1].plot(x_range,expectation(x_range,b=28),"r--")
a2[1,0].plot(x_range,expectation(x_range,b=27),"r--")
a2[1,1].plot(x_range,expectation(x_range,b=27),"r--")
a2[2,0].plot(x_range,expectation(x_range,b=28),"r--")
a2[2,1].plot(x_range,expectation(x_range,b=28),"r--")
a2[3,0].plot(x_range,expectation(x_range,b=25),"r--")
a2[3,1].plot(x_range,expectation(x_range,b=25),"r--")

aa[0,1].legend(bbox_to_anchor=(1,.85))
ff.subplots_adjust(right=0.8)
ff.savefig(f"figures/SAD_PDF_CDF.pdf")

a0[0,1].legend(bbox_to_anchor=(1,.85))
f0.subplots_adjust(right=0.8)
f0.savefig(f"figures/SAD_core_cloud.pdf")
#f0.tight_layout()

f2.tight_layout()
f2.savefig(f"figures/Arrhenius_plots.pdf")

fig.tight_layout()
fig.savefig("figures/skewness_combo_plot.pdf")

plt.show()
