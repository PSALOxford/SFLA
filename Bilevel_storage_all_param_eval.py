from Bilevel_Storage import calculate_prices, calculate_prices_with_fixed_binary, solve_bilevel
from joblib import Parallel, delayed
import itertools
import os
import numpy as np
from WT_error_gen import WT_sce_gen
import gurobipy as gp
from gurobipy import GRB

def solve_one_instance(param, save_path_root, M, thread):
    # this function solves the bilevel problem for a single parameter setting and random run

    epsilon, theta, N, numerical_focus, IntegralityFocus, T, gurobi_seed, method = param

    log_file_name = f'{method}_theta{theta}_epsilon{epsilon}_seed{gurobi_seed}_N{N}_T{T}_numerical_focus{numerical_focus}_IntegralityFocus{IntegralityFocus}.txt'
    log_file_name = os.path.join(save_path_root, log_file_name)
    # remove the existing one if any
    if os.path.exists(log_file_name):
        os.remove(log_file_name)

    seed = gurobi_seed
    rng = np.random.RandomState(seed)

    # Tstart should also be randomly generated
    Tstart = rng.randint(0, 24-T+1)

    num_gen = 5  # the number of thermal generators
    num_WT = 1  # the number of wind turbines
    num_storage = 1  # the number of storages
    N_samples = 1000  # the number of wind power scenarios used for training

    eta = {s: 0.95 for s in range(num_storage)}  # charge efficiency
    storage_alpha = {s: 0.5 for s in range(num_storage)}  # control stored energy of storage
    E_MAX = {s: 400 / 1000 for s in range(num_storage)}  ### MWh
    E_INI = {s: 0.5 * E_MAX[s] for s in range(num_storage)}  ### MWh
    P_MAX_DIS = {s: 100 / 1000 for s in range(num_storage)}  ### MW
    R_MAX_DIS_UP = R_MAX_DIS_DN = {s: 50 / 1000 for s in range(num_storage)}  ### MW
    P_MAX_CH = {s: 50 / 1000 for s in range(num_storage)}  ### MW
    R_MAX_CH_UP = R_MAX_CH_DN = {s: 100 / 1000 for s in range(num_storage)}  ### MW
    c_ch = {s: 2 for s in range(num_storage)}  ### $/MWh
    c_dis = {s: 12 for s in range(num_storage)}  ### $/MWh

    LOAD = [8768.88888888889,
            8768.88888888889,
            8644.44444444445,
            8217.77777777778,
            7986.66666666667,
            7933.33333333333,
            7755.55555555556,
            7666.66666666667,
            7666.66666666667,
            7702.22222222222,
            7773.33333333333,
            7595.55555555556,
            7595.55555555556,
            7595.55555555556,
            7506.66666666667,
            7720,
            7915.55555555556,
            8111.11111111111,
            7986.66666666667,
            8146.66666666667,
            8288.88888888889,
            8448.88888888889,
            8253.33333333333,
            8093.33333333333
            ]
    P_MAX = [850, 400, 240, 310, 330]

    P_MAX = np.array(P_MAX) * np.max(LOAD) / np.sum(P_MAX) * 1.5
    cost_p_values = np.array([10, 16, 27, 46, 182, ])
    # add +-20% random noise to the bid price
    cost_p_values = cost_p_values * rng.uniform(0.8, 1.2, len(cost_p_values))
    cost_p_values = np.sort(cost_p_values)

    P_MAX = np.array(P_MAX) / 1000
    P_MIN = {g: 0 for g in range(num_gen)}
    R_MAX_UP = {g: 0.5 * P_MAX[g] for g in range(num_gen)}
    R_MAX_DN = {g: 0.5 * P_MAX[g] for g in range(num_gen)}

    LOAD = np.array(LOAD)[Tstart:Tstart+T] / 1000

    R_UP_EX = {t: 0 * LOAD[t] for t in range(len(LOAD))}
    R_DN_EX = {t: 0 * LOAD[t] for t in range(len(LOAD))}

    # Initialize the c[t, g] as a dictionary of dictionaries
    c = {(t, g): cost_p_values[g] for t in range(T) for g in range(num_gen)}
    # print(c)
    c_rs = {(t, g): 0.5 * cost_p_values[g] for t in range(T) for g in range(num_gen)}
    c_cur = {(t, j): 65 for t in range(T) for j in range(num_WT)} # the cost of wind power curtailment, 50 GBP https://como.ceb.cam.ac.uk/media/preprints/c4e-preprint-304.pdf

    WT_total = 0.6 * 8  # 0.6*8
    WT_individual = WT_total / num_WT

    W_FORE, WT_error_scenarios, WT_full_scenarios = WT_sce_gen(num_WT, N_samples * 5)
    # WT_full_scenarios ~(0,1) , and scale
    W_FORE = W_FORE[Tstart:Tstart+T] * WT_individual  # scale
    WT_error_scenarios = WT_error_scenarios[:, Tstart:Tstart+T] * WT_individual  # scale
    WT_full_scenarios = WT_full_scenarios[:, Tstart:Tstart+T] * WT_individual  # scale

    ## j all J sum
    WT_error_scenarios = WT_error_scenarios.sum(axis=-1)
    WT_full_scenarios = WT_full_scenarios.sum(axis=-1)

    # for out of sample test, divide train test sets
    WT_error_scenarios_train = WT_error_scenarios[:N_samples]
    WT_error_scenarios_test = WT_error_scenarios[N_samples:]

    # solve the bilevel optimization problem
    param_dict = dict(T=T, N=N, M=M, theta=theta, epsilon=epsilon, WT_error_scenarios_train=WT_error_scenarios_train,
                      num_storage=num_storage, num_gen=num_gen, num_WT=num_WT, LOAD=LOAD, R_UP_EX=R_UP_EX, R_DN_EX=R_DN_EX,
                      P_MIN=P_MIN, P_MAX=P_MAX, R_MAX_UP=R_MAX_UP, R_MAX_DN=R_MAX_DN, W_FORE=W_FORE, P_MAX_DIS=P_MAX_DIS,
                      R_MAX_DIS_UP=R_MAX_DIS_UP, R_MAX_DIS_DN=R_MAX_DIS_DN, P_MAX_CH=P_MAX_CH, R_MAX_CH_UP=R_MAX_CH_UP,
                      R_MAX_CH_DN=R_MAX_CH_DN, E_MAX=E_MAX, E_INI=E_INI, eta=eta, storage_alpha=storage_alpha, c=c,
                      c_rs=c_rs, c_cur=c_cur, c_ch=c_ch, c_dis=c_dis, method=method, rng = rng,
                      numerical_focus=numerical_focus, IntegralityFocus=IntegralityFocus, thread=thread,
                      gurobi_seed=gurobi_seed, log_file_name=log_file_name)

    (prob, lambda_en, lambda_up, lambda_dn, b_hat_ch, p_hat_ch, b_hat_dis, p_hat_dis,
               b_hat_ch_up, r_ch_up, b_hat_ch_dn, r_ch_dn, b_hat_dis_up, r_dis_up, b_hat_dis_dn, r_dis_dn,
               p_ch, p_dis, r_hat_dis_up, r_hat_dis_dn, r_hat_ch_up, r_hat_ch_dn, e,
               r_wm_up, r_wm_dn, r_up, r_dn, w_cur, p, k, Q_WM_UP, Q_WM_DN, random_var_scenarios) = solve_bilevel(**param_dict)


    ##################################################################################################
    ############################# New Problem For Solving Market Clearing#############################
    ##################################################################################################
    if (prob.status == gp.GRB.OPTIMAL or prob.status == gp.GRB.TIME_LIMIT) and prob.SolCount>0:
        # Assuming r_wm_up and r_wm_dn are Gurobi variables, get their values after optimization
        r_wm_up_values = np.array([var.X for var in r_wm_up])
        r_wm_dn_values = np.array([var.X for var in r_wm_dn])

        random_var_scenarios_test = WT_error_scenarios_test.reshape(WT_error_scenarios_test.shape[0], -1)

        constraints_up_satisfied = r_wm_up_values >= -random_var_scenarios_test
        constraints_dn_satisfied = r_wm_dn_values >= random_var_scenarios_test

        # Combine the two arrays into one, with each constraint's results following the other
        combined_constraints = np.concatenate([constraints_up_satisfied, constraints_dn_satisfied], axis=1)

        # Calculate the reliability of the market modelled in the bilevel problem
        reliability_combined = np.mean(np.all(combined_constraints, axis=1)) * 100

        print(f'The out-of-sample JCC satisfaction rate is {reliability_combined}%')

        b_hat_ch_values = b_hat_ch.X
        b_hat_dis_values = b_hat_dis.X
        b_hat_dis_up_values = b_hat_dis_up.X
        b_hat_ch_up_values = b_hat_ch_up.X
        b_hat_dis_dn_values = b_hat_dis_dn.X
        b_hat_ch_dn_values = b_hat_ch_dn.X

        p_hat_ch_values = p_hat_ch.X
        p_hat_dis_values = p_hat_dis.X
        r_hat_dis_up_values = r_hat_dis_up.X
        r_hat_ch_up_values = r_hat_ch_up.X
        r_hat_dis_dn_values = r_hat_dis_dn.X
        r_hat_ch_dn_values = r_hat_ch_dn.X
        # post-processing the bids and offers before feeding them to the exact market clearing
        for t in range(T):  # assuming time_periods is defined
            for s in range(num_storage):  # assuming scenarios is defined
                # ensure the cleared quantity mathches the desired zero quantity
                p_hat_ch_values[t, s] *= bool(p_ch[t, s].X)
                p_hat_dis_values[t, s] *= bool(p_dis[t, s].X)
                r_hat_ch_up_values[t, s] *= bool(r_ch_up[t, s].X)
                r_hat_dis_up_values[t, s] *= bool(r_dis_up[t, s].X)
                r_hat_ch_dn_values[t, s] *= bool(r_ch_dn[t, s].X)
                r_hat_dis_dn_values[t, s] *= bool(r_dis_dn[t, s].X)

                # ensure the acceptance of bids/offers
                b_hat_ch_values[t, s] += 1e-5
                b_hat_dis_values[t, s] -= 1e-5
                b_hat_dis_up_values[t, s] -= 1e-5
                b_hat_ch_up_values[t, s] -= 1e-5
                b_hat_dis_dn_values[t, s] -= 1e-5
                b_hat_ch_dn_values[t, s] -= 1e-5

        # calculate the market prices by sensitivity analysis
        # the following reliability is the one in the exact market clearing problem
        energy_prices, reserve_up_prices, reserve_down_prices, reliability_combined_exact = calculate_prices(LOAD, R_UP_EX, R_DN_EX, T,
                                                                                 num_storage,
                                                                                 num_gen, num_WT, N, epsilon, theta,
                                                                                 k, M,
                                                                                 random_var_scenarios,
                                                                                 Q_WM_UP, Q_WM_DN, P_MIN, P_MAX,
                                                                                 R_MAX_UP,
                                                                                 R_MAX_DN, W_FORE, c, c_rs, c_cur,
                                                                                 b_hat_ch_values, b_hat_dis_values,
                                                                                 b_hat_dis_up_values,
                                                                                 b_hat_ch_up_values,
                                                                                 b_hat_dis_dn_values,
                                                                                 b_hat_ch_dn_values,
                                                                                 p_hat_ch_values, p_hat_dis_values,
                                                                                 r_hat_dis_up_values,
                                                                                 r_hat_ch_up_values,
                                                                                 r_hat_dis_dn_values,
                                                                                 r_hat_ch_dn_values,
                                                                                 WT_error_scenarios_test
                                                                                 )
        # CALCULATE THE cleared quantity
        p_ch_values, p_dis_values, r_ch_up_values, r_dis_up_values, r_ch_dn_values, r_dis_dn_values = calculate_prices_with_fixed_binary(
            LOAD, R_UP_EX, R_DN_EX, T, num_storage, num_gen, num_WT, N, epsilon, theta, k, M,
            random_var_scenarios, Q_WM_UP, Q_WM_DN, P_MIN, P_MAX, R_MAX_UP, R_MAX_DN, W_FORE,
            c,
            c_rs, c_cur, b_hat_ch_values, b_hat_dis_values, b_hat_dis_up_values,
            b_hat_ch_up_values, b_hat_dis_dn_values, b_hat_ch_dn_values, p_hat_ch_values,
            p_hat_dis_values, r_hat_dis_up_values, r_hat_ch_up_values, r_hat_dis_dn_values,
            r_hat_ch_dn_values, WT_error_scenarios_test, c_ch, c_dis)

        total_profit = 0.0
        for t in range(T):
            for s in range(num_storage):
                # Compute the total profit
                total_profit += (- (energy_prices[t] + c_ch[s]) * p_ch_values[t, s]
                                 + (energy_prices[t] - c_dis[s]) * p_dis_values[t, s]
                                 + reserve_up_prices[t] * (r_ch_up_values[t, s] + r_dis_up_values[t, s])
                                 + reserve_down_prices[t] * (r_ch_dn_values[t, s] + r_dis_dn_values[t, s]))

        print(f"Actual profit: {total_profit}")

    else:
        print("Optimal solution was not found.")
        total_profit = np.nan
        reliability_combined = np.nan
        reliability_combined_exact = np.nan

    # save the results
    result_dict = {'total_profit (kUSD)': total_profit, 'reliability_test (%)': reliability_combined, 'reliability_test_exact (%)': reliability_combined_exact}
    result_dict_path =f'results_{method}_theta{theta}_epsilon{epsilon}_seed{gurobi_seed}_N{N}_T{T}_numerical_focus{numerical_focus}_IntegralityFocus{IntegralityFocus}.npy'
    result_dict_path = os.path.join(save_path_root, result_dict_path)

    np.save(result_dict_path, result_dict, allow_pickle=True)

def main():
    n_jobs = 30
    M = 1e5
    thread = 4
    # ---------------------------------------------------------------------------------
    theta_list = [1e-2, 5e-2]  # [1e-3, 1e-1] the Wasserstein radius. Bonferroni approximation requires small theta for feasibility
    epsilon_list = [0.05, 0.025]  # [0.05, 0.01] the risk level
    gurobi_seed_list = [i for i in range(0, 10000*30, 10000)]
    T_list = [4, 8, 12, 16, 20, 24]
    N_WDR_list = [50]
    numerical_focus_list = [False]
    IntegralityFocus_list = [0]
    method_list = ['proposed', 'linearforN', 'wcvar', 'bonferroni']  # proposed, linearforN, wcvar, bonferroni

    # find the combination of all these parameters
    param_comb = list(itertools.product(epsilon_list, theta_list, N_WDR_list, numerical_focus_list, IntegralityFocus_list, T_list, gurobi_seed_list, method_list))

    save_path_root = os.path.join(os.getcwd(), f'Bilevel_results_bigM{int(M)}_thread{int(thread)}')
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    # solve the SUC for all the combinations in parallel
    Parallel(n_jobs=n_jobs)(delayed(solve_one_instance)(param, save_path_root, M, thread) for param in param_comb)

if __name__ == '__main__':
    main()