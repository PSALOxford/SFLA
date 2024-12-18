import os
import numpy as np
import pandas as pd
from SUC import solve_SUC
import pandapower as pp
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pd2ppc import _pd2ppc
from WT_error_gen import WT_sce_gen
import itertools
from gurobipy import GRB
from joblib import Parallel, delayed

def solve_one_instance(param, save_path_root, quadra_cost, bigM, thread):
    # this function solves the SUC for a single parameter setting and random seed
    # ------------------
    # parameters
    network_name, load_scaling_factor, epsilon, theta, T, num_gen, N_WDR, gurobi_seed, method = param
    # ------------------
    num_WT = 10  # the number of wind turbines
    N_samples_train = 1000  # the number of wind power scenarios used for training
    N_samples_test = 5000  # the number of wind power scenarios used for testing

    MIPGap = 0.001

    log_file_name = (f'{network_name}_theta{theta}_epsilon{epsilon}_gurobi_seed{gurobi_seed}'
                     f'_num_gen{num_gen}_N_WDR{N_WDR}_load_scaling_factor{load_scaling_factor}_{method}_T{T}{"quadra_cost" if quadra_cost else ""}.txt')
    log_file_name = os.path.join(save_path_root, log_file_name)
    # remove the old log file if any
    if os.path.exists(log_file_name):
        # # skip the current run if the log file already exists
        # return None
        os.remove(log_file_name)

    result_dict_path = (f'result_{network_name}_theta{theta}_epsilon{epsilon}_gurobi_seed{gurobi_seed}'
                        f'_num_gen{num_gen}_N_WDR{N_WDR}_load_scaling_factor{load_scaling_factor}_{method}_T{T}{"quadra_cost" if quadra_cost else ""}.npy')
    result_dict_path = os.path.join(save_path_root, result_dict_path)
    # remove the old file if any
    if os.path.exists(result_dict_path):
        os.remove(result_dict_path)
    # ------------------

    network_dict = {'case118': pp.networks.case118(),
                    'case300': pp.networks.case300(),
                    'case24_ieee_rts': pp.networks.case24_ieee_rts(),
                    'case14': pp.networks.case14(),
                    'case5': pp.networks.case5(),
                    'case4gs': pp.networks.case4gs(),
                    'case_ieee30': pp.networks.case_ieee30()}

    seed = gurobi_seed
    rng = np.random.RandomState(seed)

    rng_fixed = np.random.RandomState(0) # this is to avoid too much randomness that requires too many runs to have stable results

    # Tstart should also be randomly generated
    Tstart = rng.randint(0, 48-T) # rng.randint(0, 48-T)

    # load network model
    network = network_dict[network_name]

    # load network load data
    load_location = os.path.join(os.getcwd(), 'data', 'UK_norm_load_curve_highest.npy')
    network_load = np.load(load_location)
    # the network load is at half-hourly resolution, we need to average the consceutive time steps to get hourly resolution
    network_load = np.mean(np.vstack([network_load[::2],
                                      network_load[1::2]]), axis=0)
    # duplicate the network load to make it two days
    network_load = np.tile(network_load, 2)
    network_load = network_load[Tstart:Tstart+T]

    # -------------------------------------
    pp.rundcpp(network)
    _, ppci = _pd2ppc(network)
    bus_info = ppci['bus']
    branch_info = ppci['branch']
    PTDF = makePTDF(ppci["baseMVA"], bus_info, branch_info,
                    using_sparse_solver=False)

    num_branch = len(branch_info)

    # get load info
    load_bus_size = bus_info[:, 2] * load_scaling_factor

    load_total = np.sum(load_bus_size)
    # we then get the load curves at all buses, using the network_load curve
    load_bus_all = load_bus_size.reshape(1, -1) * network_load.reshape(-1, 1)

    ###### set generator capacity
    gen_cap_total = network.gen['p_mw'].sum()  # the total generation capacity
    gen_cap_individual = gen_cap_total / num_gen  # the individual generation capacity
    # add some randomness when assigning the generation capacity to each generator
    gen_cap_individual = rng_fixed.uniform(0.6, 1.4, num_gen) * gen_cap_individual
    gen_pmin_individual = 0.25 * gen_cap_individual  # the individual minimum generation capacity. This is based on "DISTRIBUTIONALLY ROBUST CHANCE-CONSTRAINED GENERATION EXPANSION PLANNING"

    gen_ramp_up = rng_fixed.uniform(0.8 / 100, 30 / 100,
                                    num_gen) * 60 * gen_cap_individual  # the ramping-up limit (%/min) of gas generators
    gen_ramp_down = gen_ramp_up  # the ramping-down limit of thermal generators
    # the following is based on the Alberta Electric System Operator, which states that the regulation range should be at most 10 times of the ramp rate in (MW/min)
    gen_UR_max = gen_ramp_up / 60 * 10  # the maximum upward reserve of thermal generators
    gen_DR_max = gen_UR_max  # the maximum downward reserve of thermal generators
    # generator cost parameters
    gen_cost = rng.uniform(23.13, 57.03, num_gen)  # the cost of gas generators (USD/MWh)
    gen_cost_quadra = rng.uniform(0.002, 0.008, num_gen)  # the quadratic cost of gas generators (USD/MWh^2)
    gen_cost_fixed = rng.uniform(0, 600, num_gen) * gen_cap_individual / 400  # the fixed cost of gas generators (USD/hr)
    urc = drc = gen_cost / 2  # the cost of reserve (USD/MWh)
    su = rng.uniform(20, 150, num_gen) * gen_cap_individual  # the start-up cost per start-up per MW
    sd = rng.uniform(2, 15, num_gen) * gen_cap_individual  # the shut-down cost per shut-down per MW
    # the minimum up and down time of gas generators
    UT = 2
    DT = 1

    ###### system reserve requirement
    UR_extra_total = 0. * load_total  # the required extra upward reserve. It is set small considering the major mismatch comes from wind
    DR_extra_total = 0. * load_total  # the required extra downward reserve

    # get generator locations
    bus_list = np.arange(bus_info.shape[0])
    gen_bus_list = rng_fixed.choice(bus_list, num_gen, replace=True)
    WT_bus_list = rng_fixed.choice(bus_list, num_WT, replace=True)

    # get line info
    P_line_limit = np.abs(ppci['branch'][:, 5])  # the line flow limit
    # clip on 2 times of the total load to avoid numerical issues
    P_line_limit = np.clip(P_line_limit, 0, 2 * load_total)

    WT_total = 0.6 * load_total
    WT_individual = WT_total / num_WT
    # load the wind power scenarios, which is decomposed into prediction and error scenarios
    WT_pred, WT_error_scenarios, WT_full_scenarios = WT_sce_gen(num_WT, N_samples_train + N_samples_test)
    WT_pred = WT_pred[Tstart:Tstart+T] * WT_individual  # scale
    WT_error_scenarios = WT_error_scenarios[:, Tstart:Tstart+T] * WT_individual  # scale
    WT_full_scenarios = WT_full_scenarios[:, Tstart:Tstart+T] * WT_individual  # scale
    # generate training and testing scenarios
    WT_error_scenarios_train = WT_error_scenarios[:N_samples_train]
    WT_error_scenarios_test = WT_error_scenarios[N_samples_train:]

    # perform SUC
    input_param_dict = {'T': T, 'num_gen': num_gen, 'num_WT': num_WT, 'num_branch': num_branch,
                        'load_bus_all': load_bus_all, 'PTDF': PTDF, 'gen_cap_individual': gen_cap_individual,
                        'gen_pmin_individual': gen_pmin_individual, 'gen_UR_max': gen_UR_max, 'gen_DR_max': gen_DR_max,
                        'UR_extra_total': UR_extra_total, 'DR_extra_total': DR_extra_total, 'gen_ramp_up': gen_ramp_up,
                        'gen_ramp_down': gen_ramp_down, 'UT': UT, 'DT': DT, 'WT_pred': WT_pred,
                        'WT_error_scenarios_train': WT_error_scenarios_train, 'P_line_limit': P_line_limit,
                        'gen_bus_list': gen_bus_list, 'WT_bus_list': WT_bus_list, 'N_WDR': N_WDR, 'epsilon': epsilon, 'thread': thread,
                        'theta': theta, 'method': method, 'MIPGap': MIPGap, 'gen_cost': gen_cost, 'gen_cost_quadra': gen_cost_quadra, 'gen_cost_fixed':gen_cost_fixed,
                        'urc': urc, 'drc': drc, 'su': su, 'sd': sd, 'bigM': bigM, 'gurobi_seed': gurobi_seed, 'log_file_name': log_file_name, 'rng': rng, 'quadra_cost': quadra_cost}
    prob, x, gen_UR_all, gen_DR_all, gen_v_all, \
        gen_power_all, WT_schedule_all, WT_curtail_all, A, B, d = solve_SUC(**input_param_dict)

    # Check the status of the solution
    if (prob.status not in [GRB.Status.OPTIMAL, GRB.Status.TIME_LIMIT]) or (prob.SolCount == 0):
        min_cost = np.nan
        t_solve = np.nan
        reliability_test = np.nan
    else:
        min_cost = prob.objVal
        t_solve = prob.Runtime
        # calculate the out-of-sample JCC satisfaction rate
        random_var_scenarios_test = WT_error_scenarios_test.reshape(WT_error_scenarios_test.shape[0], -1)
        reliability_test = 100*np.mean(np.all(A @ x.X <= random_var_scenarios_test @ B.T + d, axis=1))
        print('------------------------------------')
        print('Optimal value:', prob.objVal)
        print(f'The out-of-sample JCC satisfaction rate is {reliability_test}%')
        print(f'spent {t_solve} seconds for solving the SUC')
        print(f'The method used is {method}')
        print(
            f'theta = {theta}, epsilon = {epsilon}, N_WDR = {N_WDR}, load_scaling_factor = {load_scaling_factor}, network_name = {network_name}')
        print('------------------------------------')

    # save the results
    result_dict = {'min_cost (USD)': min_cost, 'reliability_test (%)': reliability_test, 't_solve (s)': t_solve}
    np.save(result_dict_path, result_dict, allow_pickle=True)
    # load the results
    result_tuple = np.load(result_dict_path, allow_pickle=True).item()


def run_all_param():
    # this function runs the SUC for all the parameter combinations
    bigM = 1e5  # this is only for "exact"
    thread = 4 # the number of threads for Gurobi solver
    n_jobs = 16 # higher than 20 would cause memory issues
    # ---------------------------------------------------------------------------------
    T_list = [24]  # the number of time steps
    theta_list = [5e-1, 1e-1]  # the Wasserstein radius. Bonferroni approximation requires small theta for feasibility
    epsilon_list = [0.05, 0.025]  # [0.05, 0.025] the risk level
    gurobi_seed_list = [i for i in range(10000*0, 10000*150, 10000)]
    num_gen_list = [100]  # the number of thermal generators
    N_WDR_list = [50, 100, 150]  # the number of scenarios for the WDRJCC
    load_scaling_factor_list = [1]  # [1] the scaling factor for the load
    method_list = ['proposed', 'exact']  # proposed, ori, exact, wcvar, bonferroni. the method to reformulate the WDRJCC
    network_name_list = ['case24_ieee_rts']
    quadra_cost = True

    # find the combination of all these parameters
    param_comb = list(itertools.product(network_name_list, load_scaling_factor_list, epsilon_list, theta_list, T_list, num_gen_list, N_WDR_list, gurobi_seed_list, method_list))

    save_path_root = os.path.join(os.getcwd(), f'SUC_results_bigM{int(bigM)}_thread{int(thread)}')
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    # solve the SUC for all the combinations in parallel
    Parallel(n_jobs=n_jobs)(delayed(solve_one_instance)(param, save_path_root, quadra_cost, bigM, thread) for param in param_comb)

if __name__ == '__main__':
    run_all_param()