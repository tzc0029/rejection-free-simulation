'''
Author: Tianshi Che
Date: Nov 16th 2021
'''

'''
Version Note: direct method rejection free
'''
from math import e
import matplotlib
import numpy as np, argparse, sys, random, os
from os import path
from utils_backup import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cayenne.simulation import Simulation
import time
from tqdm import tqdm
import numpy as np
import seaborn as sns

def main():
    start_time = time.time()
  ###############################################
    CO_Oxidation_no_vape = """
        const compartment comp1;
        comp1 = 1; # volume of compartment
        r1: CO_g + E => CO_s; k1;
        r3: O2_g + 2E => 2O_s; k3;
        r5: CO_s + O_s => CO2_g + 2E; k5;
        k1 = 5.78e5;
        k3 = 1.62e5;
        k5 = 1.71e2;
        CO_g = 0;
        CO_s = 1;
        E = 1;
        O2_g = 0;
        O_s = 1;
        CO2_g = 0;
        chem_flag = true;
    """
    CO_Oxidation = """
        const compartment comp1;
        comp1 = 1; # volume of compartment
        r1: CO_g + E => CO_s; k1;
        r2: CO_s => CO_g + E; k2;
        r3: O2_g + 2E => 2O_s; k3;
        r4: 2O_s => O2_g + 2E; k4;
        r5: CO_s + O_s => CO2_g + 2E; k5;
        k1 = 5.78e5;
        k2 = 1.65e3;
        k3 = 1.62e5;
        k4 = 2.33e11;
        k5 = 1.71e2
        CO_g = 0;
        CO_s = 1;
        E = 1;
        O2_g = 0;
        O_s = 1;
        CO2_g = 0;
        chem_flag = true;
    """

    NH3_Decomposition = """
        const compartment comp1;
        comp1 = 1; # volume of compartment
        r3: NH3_g + E => NH3_s; k3;
        r1: 2H_s => 2E + H2_g; k1;
        r2: 2N_s => 2E + N2_g; k2;
        
        r4: NH3_s => NH3_g + E; k4;
        r5: N_s + H_s => NH_s + E; k5;
        r6: NH_s + E => N_s + H_s; k6;
        r7: NH_s + H_s => NH2_s + E; k7;
        r8: NH2_s + E => NH_s + H_s; k8;
        r9: NH2_s + H_s => NH3_s + E; k9;
        r10: NH3_s + E => NH2_s + H_s; k10;
        k1=3.68e5;
        k2=1.41e1;
        k3=3.94e4;
        k4=9e6;
        k5=0.18;
        k6=1.15e10;
        k7=4.45e5;
        k8=4.27e6;
        k9=1.87e6;
        k10=1.05e7;
        H_s = 1;
        E = 1;
        H2_g = 0;
        N_s = 1;
        N2_g = 0;
        NH3_g = 0;
        NH3_s = 1;
        NH_s = 1;
        NH2_s = 1;  
        chem_flag = true;
    """

    model_str_3 = """
        const compartment comp1;
        comp1 = 1; # volume of compartment
        r1: CO_g + E => CO_s; k1;
        r2: CO_s + CO_g + E; k2;
        k1 = 1.7e4
        chem_flag = true;
    """
    n = 56
    max_iter = int(5000)
    plot_title = 'CO Oxidation'
    
    ###############################################  

    sim = Simulation.load_model(CO_Oxidation_no_vape, "ModelString")
    # Note: the empty space must be named 'E'
    # the index of empty space
    idx_empty = sim.species_names.index("E") 
    # create a lattice surface
    lattice = np.full(shape=(n,n), fill_value=idx_empty, dtype=int)
    
    # list to store recorded data
    data_record = [[0] for _ in range(len(sim.species_names))]
    data_record[idx_empty] = [n * n]
    # iteration records
    x_data = [] 
    # time records
    t_data = [0]
    # cumulated time among iterations 
    cumu_time = 0
    e_0 = np.full(shape=(n,n), fill_value=0, dtype=int)
    E = [copy.deepcopy(e_0) for _ in range(len(sim.rxn_names))]
    for i in range(n):
        for j in range(n):
            neighbours = check_neighbours(i, j, n)
            neighbours_id = position_to_id(neighbours, lattice)
            e = get_react_candidate(sim._react_stoic, sim._init_state, lattice[i, j], neighbours_id, idx_empty)
            for item in e:
                E[item][i,j] = 1
    N_j = [0 for _ in range(len(sim.rxn_names))]
    for j in range(len(sim.rxn_names)):
        x = sum(sum(E[j]))
        N_j[j] = x
    E_gamma = copy.deepcopy(E)
    for j in range(len(sim.rxn_names)):
        E_gamma[j] *= int(sim._k_det[j])
    Gamma_total = sum(sum(sum(E_gamma)))

    
    for i in tqdm(range(max_iter+1)):
        if Gamma_total <= 0:
            print("no avaialable reaction")
            break
        t_data.append(cumu_time)
        row, col, reaction_idx = generate_site(n, sim, E, Gamma_total, N_j)
        r2 = random.uniform(0, 1)

        cumu_time += -np.log(r2)/Gamma_total

        # get all neighbours with predefined order (up, right, down, left)
        neighbours = check_neighbours(row, col, n)
        # also create a random version
        neighbours_random = neighbours.copy()
        neighbours_random = neighbours_random[1:]
        random.shuffle(neighbours_random)
        neighbours_random.insert(0, neighbours[0])
        
        neighbours_id = position_to_id(neighbours, lattice)
        update_records(data_record, i)
        change_log = update_events(sim._react_stoic[:,reaction_idx], sim._prod_stoic[:,reaction_idx], lattice, neighbours_random, sim._init_state, data_record, i)
        
        change_cand = []
        for rowp, colp in change_log:
            neighbours = check_neighbours(rowp, colp, n)
            for item in neighbours:
                if item not in change_cand:
                    change_cand.append(item)

        for rowpp, colpp in change_cand:
            # we let all candidates be 0 first, then decide their E
            for indx in range(len(E)):
                if E[indx][rowpp,colpp] == 1:
                    E[indx][rowpp,colpp] = 0
                    N_j[indx] -= 1
                    Gamma_total -= 1 * sim._k_det[indx]
            neighbours = check_neighbours(rowpp, colpp, n)
            neighbours_id = position_to_id(neighbours, lattice)
            e = get_react_candidate(sim._react_stoic, sim._init_state, lattice[rowpp, colpp], neighbours_id, idx_empty)
            for rx in e:
                if E[rx][rowpp,colpp] != 1:
                    E[rx][rowpp,colpp] = 1
                    N_j[rx] += 1
                    Gamma_total += 1 * sim._k_det[rx]
        
        if i == 2000 or i == 4000:
            filtered_mat = create_filtered_mat(lattice, n, filter_size=5)
            # data = np.random.rand(8, 12)
            ax = sns.heatmap(filtered_mat, annot=True, annot_kws={"fontsize":1}, vmin=0, vmax=1, cmap="YlGnBu")
            cbar = ax.collections[0].colorbar
            cbar.set_ticks([0, .5, 1])
            cbar.set_ticklabels(['0%', '50%', '100%'])
            ax.set_title("CO_s at {} iters, k_CO: {:.2e}, K_O2: {:.2e}".format(i, sim._k_det[0], sim._k_det[1]))
            plt.savefig('heat_{}_k_CO_{:.2e}.pdf'.format(i, sim._k_det[0]))
            plt.clf()
  
    ###################################
    ####   visualization related   ####
    ###################################
    for i in range(len(data_record)):
        # if sim.species_names[i] != "E" and sim._init_state[i] != 0:
        if sim._init_state[i] != 0:
        # if sim.species_names[i] != "O2_g" and sim.species_names[i] != "CO_g":
        # if sim.species_names[i] == "O_s":
        # if sim._init_state[i] != 0:
            quotients = [number / (n*n) for number in data_record[i]]
            plt.plot(t_data, quotients, marker="", label=sim.species_names[i])
            # plt.plot(t_data, data_record[i], marker="", label=sim.species_names[i], linewidth=0.1)
            # plt.plot(t_data, data_record[i], marker="", label=sim.species_names[i])
            # plt.plot(t_data[:500], data_record[i][:500], marker="", label=sim.species_names[i])
    
    # plt.ylim(ymax=0.2)
    plt.title(plot_title + "--- %.2f seconds ---" % (time.time() - start_time))
    plt.xlabel("Time")
    plt.ylabel("Population (%)")
    plt.legend(bbox_to_anchor=(0.75, 0.5), loc='center left')
    ######### log scale
    plt.xscale("log")
    # plt.xlim(xmin=0)
    #########
    plt.savefig('test.png')
    directory = "results/" + plot_title + "_DotLine_View_Size_{}_Iter_{}".format(n, max_iter)

    if not os.path.exists(directory):
        os.mkdir(directory)
    plt.savefig(os.path.join(directory, plot_title + "_Size_{}_Iter_{}.png".format(n, max_iter)))
    print("--- %s seconds ---" % (time.time() - start_time))

    ###################################
    ####    Visualizated Matrix    ####
    ###################################
    # values = np.unique(lattice.ravel())
    # im = plt.imshow(lattice, interpolation='none')
    # colors = [ im.cmap(im.norm(value)) for value in values]
    # lb = [sim.species_names[i] for i in values]
    # patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=lb[i]) ) for i in range(len(values)) ]
    # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    
    # plt.title(plot_title)
    # directory = "results/" + plot_title + "_Matrix_View_Size_{}_Iter_{}".format(n, max_iter)
    # if not os.path.exists(directory):
    #     os.mkdir(directory)
    # plt.savefig(os.path.join(directory, plot_title + "_Size_{}_Iter_{}.png".format(n, max_iter)))
    # # print(lattice)
    # print("Number of empty: {} ({:.2%}), O: {} ({:.2%}), CO: {} ({:.2%})".format(np.count_nonzero(lattice == 0), np.count_nonzero(lattice == 0)/total, np.count_nonzero(lattice == 1), np.count_nonzero(lattice == 1)/total, np.count_nonzero(lattice == 2), np.count_nonzero(lattice == 2)/total))

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-size',		type=int,             default=256,			help='Size of the lattice surface')
    # parser.add_argument('-iter',		type=int,             default=100000,			help='Max iteration')
    # args = parser.parse_args()
    main()
