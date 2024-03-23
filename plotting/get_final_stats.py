import numpy as np
from os.path import exists
from datetime import date,datetime,timedelta
import matplotlib.pyplot as plt
from matplotlib import colors


dates = ['Nov_03/'] # ['Oct_11/'] 
other_file_dates = [dates[0][:5] + i for i in list(np.array(np.r_[4:9],dtype=str))]
seeds = np.r_[1000:1200]
temps = np.r_[274:320:3]
experiment = 'SpunupT/'
extra_folder = 'one-TRC'

#out_file = '/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/'

run_length=50000
L = 20 #20
locat = '/home/cfebvre/camille/OUT_TNM/temp_TNM/experiments/'
BasicTNM_locat = locat + 'BasicTNM/Jun_17/'


# assemble filename of input
def find_file(date,seed,T,file_type="pypy"):
    # return locat+f"pypy_seed{str(seed)}{date}_{str(T)}K_TNM_L20.dat"
 #   print("* * * f i n d i n g   f i l e * * * ")
    #print("looking for ",type," file")
    if file_type == "pypy": 
        #print("entered pypy for loop")
        file = f"{locat}{experiment}{date}{extra_folder}/pypy_seed{str(seed)}{date[:-1]}_{str(T)}K_TNM_L{L}.dat"
        date2 = date[:-1]
        #print("date2: ",date2)
        num_tries=40
        if not exists(file):
#            print("trying other file dates")
            for d in np.r_[0:num_tries]: #len(other_dates)]:
#                print("try ",d)
                date1 = datetime.strptime(date[:-1],"%b_%d")
                adjusted_date = date1 + timedelta(days=int(d))
                date2 = adjusted_date.strftime("%b_%d")
                file = f"{locat}{experiment}{date}{extra_folder}/pypy_seed{str(seed)}{date2}_{str(T)}K_TNM_L{L}.dat"
                if exists(file): 
#                    print("file found")
                    break 
                else: #continue
                    #print(file,exists(file))
                    continue
    if exists(file):
        #print("File exists") #: ",file)
        return file, date2
    else: 
        print("Can't find file: ",file)
        return 0,0


def get_pop_div(date,seed,T,type="pypy",filename=None):
    if type == "basic": # find BasicTNM run files
        pypy_file = filename
        print('*******\nBasicTNM file:\n',pypy_file)
        #print(exists(pypy_file))
    else:
        pypy_file,date2 = find_file(date,seed,T,"pypy")
    #print("date, seed, T: ",date,seed,T)
    if pypy_file != 0:
        if exists(pypy_file):
            with open(pypy_file,'r') as pypy:
                for line in pypy:
                    pass
                # line is last line
                elements = line.split(" ")
                # (tgen,sum(populations),len(species),len(encountered),core_pop,core_div,F))
                gens_run = int(elements[0])
                populations = int(elements[1])
                diversities = int(elements[2])
                core_pops = int(elements[4])
                core_divs = int(elements[5])
    #            populations.append(int(elements[1]))
    #            diversitites.append(int(elements[2]))
    else:
        print(f"could not open file with seed {seed} and T = {T}")
#        populations.append(0)
#        diversities.append(0)
        return 0,0,0,0,0
    return gens_run, populations, diversities, core_pops, core_divs


for date in dates:
    print("date: ",date)
    out_file = f"{locat}{experiment}{date}{extra_folder}/stats_{date[:-1]}_{extra_folder}.txt"
    with open(out_file,'a') as f:
        f.write("mean abundance, variance of abundance, sample size\nmean diversity, variance of diversity, sample size\n")

    fig,ax = plt.subplots()
    pops_by_T = []
    divs_by_T = []
    survived_by_T = np.zeros(len(temps))
    i = -1
    for T in temps:
        i += 1
        print("T: ",T)
        with open(out_file,'a') as f:
            f.write(f"T: {T}\n")
        gens_this_T = []
        pops_this_T = []
        divs_this_T = []
        corepops_this_T = []
        coredivs_this_T = []
        for seed in seeds:
            gens_run,populations,diversities,core_pops,core_divs = get_pop_div(date,seed,T)
            gens_this_T.append(gens_run)
            if gens_run >= run_length-10:
                survived_by_T[i] += 1
                pops_this_T.append(populations)
                divs_this_T.append(diversities)
                corepops_this_T.append(core_pops)
                coredivs_this_T.append(core_divs)

        max_pop = 2400
        max_div = 160
        pop_bins = int(max_pop/100)
        div_bins = int(max_div/10)
        pops_by_T.append(100*np.histogram(pops_this_T,pop_bins,(0,max_pop),density=True)[0])
        divs_by_T.append(10*np.histogram(divs_this_T,div_bins,(0,max_div),density=True)[0])

        # get mean, variance, and sample size
        mu_pop = np.mean(pops_this_T)
        var_pop = np.var(pops_this_T)
        n_pop = len(pops_this_T)

        mu_div = np.mean(divs_this_T)
        var_div = np.var(divs_this_T)
        n_div = len(divs_this_T)

        with open(out_file,"a") as f:
            f.write(f"{mu_pop}, {var_pop}, {n_pop} \n") #, {survived_by_T[i]} \n")
            f.write(f"{mu_div}, {var_div}, {n_div} \n")

    
    # plot distribution by T
    cmap = plt.get_cmap('Greys')
    #colors = cmap(np.linspace(1,0,1000))
    
    pos_neg_clipped = ax.imshow(np.array(pops_by_T).transpose(),cmap=cmap) #, vmin=0, vmax=3*np.mean(np.mean(pops_by_T)),interpolation='none')
    # draw gridlines
#    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    #ax.set_xticks(np.arange(1, len(temps), 1));
    #ax.set_yticks(np.arange(1, len(pops_by_T), 1));
    ax.invert_yaxis()
    #ylims = pos_neg_clipped.get_extent()
    ax.set_yticks(np.r_[0:pop_bins])
    ax.set_yticklabels(map(str, np.r_[0:max_pop:100])) #[int(1000/len(pops_by_T)):1000-int(1000/len(pops_by_T)):int(1000/len(pops_by_T))]))
    ax.set_xticks(np.r_[.5:len(temps)-.5])
    ax.set_xticklabels(map(str,temps[:-1]),rotation=90)
    #ax.set_xticklabels(np.arange(1,10),loc=np.arange(1,10))
    ax.set_ylabel("Final abundance")
    ax.set_xlabel(r"Temperature,T ($^o$C)")

    ax.set_title(f"{experiment[:-1]} {extra_folder}")

    cbar = fig.colorbar(pos_neg_clipped,label="Conditional Probability")
    cbar.minorticks_on()

    fig_file = f"{locat}{experiment}{date}{extra_folder}/abund_distribution_{date[:-1]}_{extra_folder}.pdf"
    plt.savefig(fig_file)

    # Diversity distribution
    fig,ax = plt.subplots()
    
    pos_neg_clipped = ax.imshow(np.array(divs_by_T).transpose(),cmap=cmap) #, vmin=0, vmax=3*np.mean(np.mean(pops_by_T)),interpolation='none')
    # draw gridlines
#    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    #ax.set_xticks(np.arange(1, len(temps), 1));
    #ax.set_yticks(np.arange(1, len(pops_by_T), 1));
    ax.invert_yaxis()
    #ylims = pos_neg_clipped.get_extent()
    ax.set_yticks(np.r_[0:div_bins])
    ax.set_yticklabels(map(str, np.r_[0:max_div:10])) #[int(1000/len(pops_by_T)):1000-int(1000/len(pops_by_T)):int(1000/len(pops_by_T))]))
    ax.set_xticks(np.r_[.5:len(temps)-.5])
    ax.set_xticklabels(map(str,temps[:-1]),rotation=90)
    #ax.set_xticklabels(np.arange(1,10),loc=np.arange(1,10))
    ax.set_ylabel("Final diversity")
    ax.set_xlabel(r"Temperature,T ($^o$C)")

    ax.set_title(f"{experiment[:-1]} {extra_folder}")

    cbar = fig.colorbar(pos_neg_clipped,label="Conditional Probability")
    cbar.minorticks_on()

    fig_file = f"{locat}{experiment}{date}{extra_folder}/div_distribution_{date[:-1]}_{extra_folder}.pdf"


    plt.show()
    



