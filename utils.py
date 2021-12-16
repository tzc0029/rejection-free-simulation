'''
Author: Tianshi Che
Date: Nov 16th 2021
'''
import random
import numpy as np
import copy


# def roulette_selection(react_candidates, k_det):
#     k_sum = 0
#     probability_list = []
#     for rxn in react_candidates:
#         k_sum += k_det[rxn]
#     for rxn in react_candidates:
#         probability_list.append(k_det[rxn]/k_sum)

#     maximum = sum(item for item in probability_list)
#     rand_num = random.uniform(0, maximum)
#     current = 0

#     for i in range(len(probability_list)):
#         current += probability_list[i]
#         if current > rand_num:
#             return react_candidates[i] 


# when we consider n=4, only four immediate neighbours are considered 
def check_neighbours(row, col, n):
    n -= 1
    neighbour = []
    neighbour.append([row, col])
    neighbour.append([row-1, col]) if row > 0 else neighbour.append([n, col])
    neighbour.append([row, col+1]) if col < n else neighbour.append([row, 0])
    neighbour.append([row+1, col]) if row < n else neighbour.append([0, col])
    neighbour.append([row, col-1]) if col > 0 else neighbour.append([row, n])
    return neighbour


def position_to_id(neighbours, lattice):
    result = []
    for row, col in neighbours:
        result.append(lattice[row, col])
    return result

def generate_time(react_candidates, k_det):
    tau_list = []
    probability_list = []

    for item in react_candidates:
        probability_list.append(k_det[item])
    denominator = copy.deepcopy(sum(probability_list))

    for item in react_candidates:
        r = random.uniform(0,1)
        tau = -1/(k_det[item]/denominator)*np.log(r)/1000000
        # tau = -1/k_det[item]*np.log(r)
        tau_list.append(tau)
    return min(tau_list), react_candidates[tau_list.index(min(tau_list))]


def get_react_candidate(react_stoic, init_state, center, neighbours_id, idx_empty):
    candidate = []
    react_stoic_view = copy.deepcopy(react_stoic)
    neighbours_id_view = copy.deepcopy(neighbours_id)
    # remove the center block 
    del neighbours_id_view[0]


    # for each reactions
    for ind1 in range(react_stoic_view.shape[1]):
        # if this reaction involves center, append into candidate for further selection
        if react_stoic_view[center,ind1] > 0:
            candidate.append(ind1)

    deletion_list = copy.deepcopy(candidate)
    # for reactions in candidate set
    for rxn in candidate:
        # we minus the center because it can be fulfilled obviously
        react_stoic_view[center,rxn] -= 1            
        if all(v == 0 for v in react_stoic_view[:, rxn]):
            pass
        # for every species in this reaction
        for ind2 in range(react_stoic_view.shape[0]):
            if react_stoic_view[ind2, rxn] > 0 and neighbours_id_view.count(ind2) < react_stoic_view[ind2, rxn] and init_state[ind2] != 0:
                deletion_list.remove(rxn)
    return deletion_list


'''
Update lattice when reactions are decided
'''
def update_events(sim, reaction_idx, lattice, neighbours_random, data_record, epoch):
    react_vec_view = copy.deepcopy(sim._react_stoic[:,reaction_idx])
    prod_vec_view = copy.deepcopy(sim._prod_stoic[:,reaction_idx])
    init_state_view = copy.deepcopy(sim._init_state)
    
    ######## in order to keep track of CO #####
    if prod_vec_view[5] != 0:
        data_record[5][epoch+1] += 1
    ###########################################

    react = []
    product = []
    while not all(v == 0 for v in react_vec_view):
        for i in range(len(react_vec_view)):
            if react_vec_view[i] > 0:
                react_vec_view[i] -= 1
                if init_state_view[i] != 0:
                    react.append(i)
                    
    while not all(s == 0 for s in prod_vec_view):
        for j in range(len(prod_vec_view)):
            if prod_vec_view[j] > 0:
                prod_vec_view[j] -= 1
                if init_state_view[j] != 0:
                    product.append(j)
    random.shuffle(react)
    random.shuffle(product)
    # our assumption so far
    assert len(product) == len(react)

    change_log = []
    while product != []:
        for row, col in neighbours_random:
            if lattice[row, col] in react:
                change_log.append((row, col))
                react.remove(lattice[row, col])
                r_ind = lattice[row, col] # which mole to react (0,1,2...)
                # append a new value to data_record which is 1 less
                data_record[r_ind][epoch+1] -= 1
                lattice[row, col] = product.pop(0)
                # add one product to the record
                p_ind = lattice[row, col]
                data_record[p_ind][epoch+1] += 1
    return change_log

def update_records(data_record, i):
    for item in data_record:
        item.append(item[i])

            
def generate_site(n, sim, E, Gamma_total, N_j):
    sigma_1 = random.uniform(0, 1)
    gamma_temp = 0
    # for rxn in range(len(sim.rxn_names)):
    #     for ind in range(n*n):
    #         ii, jj = np.unravel_index(ind, (n,n), order='C')
    #         # gamma_temp+= E[rxn][ii, jj]*sim._k_det[rxn]
    #         gamma_temp+= E[rxn][ii, jj]
    #         if sigma_1 <= gamma_temp / Gamma_total:
    #             row = ii
    #             col = jj
    #             return row, col, rxn
    # print("wrong")
    # quit()
    for rxn in range(len(sim.rxn_names)):
        gamma_temp += N_j[rxn] * sim._k_det[rxn]
        if sigma_1 <= gamma_temp / Gamma_total:
            # sigma_2 = random.uniform(0, 1)
            # m = math.floor(sigma_2*N_j[rxn])
            # ii, jj = np.unravel_index(m, (n,n), order='C')
            # return ii, jj, rxn
            ij = np.where(E[rxn] == 1)
            sigma_2 = random.randint(0, ij[0].shape[0]-1)
            return ij[0][sigma_2], ij[1][sigma_2], rxn

    print("wrong")
    quit()

def create_filtered_mat(lattice, n, filter_size):
    filtered_mat_size = n-filter_size+1
    filtered_mat = np.full(shape=(filtered_mat_size, filtered_mat_size), fill_value=0, dtype=float)
    for i in range(filtered_mat_size):
        for j in range(filtered_mat_size):
            # only count O for now
            filtered_mat[i, j] = count_species(lattice, i, j, filter_size) / (filter_size * filter_size)

    return filtered_mat

def count_species(lattice, i, j, filter_size):
    submatrix = lattice[i:i+filter_size, j:j+filter_size]
    count = np.count_nonzero(submatrix == 2)
    return count

