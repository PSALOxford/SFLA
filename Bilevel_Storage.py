import numpy as np
import gurobipy as gp
from gurobipy import GRB
from WT_error_gen import WT_sce_gen
import time
from joblib import Parallel, delayed
import numpy as np

def linearize_complementarity(model, alpha, beta, U, M1, M2):
    """
    Linearizes the complementarity condition 0 <= alpha ⊥ beta >= 0
    using big-M method.

    Parameters:
    model: The optimization model (Gurobi or Pyomo model)
    alpha: The variable alpha in the complementarity condition
    beta: The variable beta in the complementarity condition
    U: The binary variable for linearization
    M1: The big-M constant for alpha
    M2: The big-M constant for beta
    """
    # Add the linearized constraints
    model.addConstr(alpha >= 0)
    model.addConstr(beta >= 0)
    model.addConstr(alpha <= M1 * U)
    model.addConstr(beta <= M2 * (1 - U))


def solve_VaR(N, theta, t, T, epsilon, random_var_scenarios, direction):
    '''
    calculate VaR for Bonferroni method at time t
    '''
    # Model setup
    if direction == 'up':
        model_name = 'VaR_up'
        adjust = 1
    else:
        model_name = 'VaR_dn'
        adjust = -1

    prob = gp.Model(model_name)

    # Decision variables
    alpha = prob.addMVar(N, lb=0, name="alpha")  # Lower bound is 0
    beta = prob.addVar(lb=-GRB.INFINITY, name="beta")  # Scalar variable
    m = prob.addMVar(N, lb=0, name="m")  # Lower bound is 0
    eta = prob.addVar(lb=-GRB.INFINITY, name="eta")  # Scalar variable

    # Constraints
    prob.addConstr(theta * beta + (1 / N) * gp.quicksum(alpha) <= epsilon / (2*T), name="epsilon_constraint")
    for i in range(N):
        prob.addConstr(alpha[i] >= 1 - m[i] * (eta + adjust * random_var_scenarios[i, t]), name=f"alpha_constraint_{i}")
        prob.addConstr(beta >= m[i], name=f"beta_constraint_{i}")

    # Objective
    prob.setObjective(eta, GRB.MINIMIZE)

    # Set solver parameters
    prob.setParam('OutputFlag', 0)
    prob.setParam('IntFeasTol', 1e-9)
    prob.setParam('FeasibilityTol', 1e-9)
    prob.setParam('OptimalityTol', 1e-9)

    # prob.write(f"{model_name}_t{t}.lp")
    # Optimize the model
    prob.optimize()

    # Return the objective value
    return prob.objVal

def solve_all_VaR(N, theta, T, epsilon, random_var_scenarios, n_jobs):
    '''
    calculate all VaR for Bonferroni method
    '''
    results_up = Parallel(n_jobs=n_jobs)(delayed(solve_VaR)(N, theta, t, T, epsilon, random_var_scenarios, 'up') for t in range(T))
    results_dn = Parallel(n_jobs=n_jobs)(delayed(solve_VaR)(N, theta, t, T, epsilon, random_var_scenarios, 'dn') for t in range(T))
    return results_up, results_dn

def market_clear_exact_JCC(LOAD, R_UP_EX, R_DN_EX, T,num_storage,num_gen,num_WT, N, epsilon, theta, k, M, random_var_scenarios,
                            Q_WM_UP, Q_WM_DN, P_MIN, P_MAX, R_MAX_UP, R_MAX_DN, W_FORE, c, c_rs, c_cur,
                           b_hat_ch_values, b_hat_dis_values, b_hat_dis_up_values,
                               b_hat_ch_up_values, b_hat_dis_dn_values, b_hat_ch_dn_values, p_hat_ch_values,
                               p_hat_dis_values, r_hat_dis_up_values, r_hat_ch_up_values, r_hat_dis_dn_values,
                               r_hat_ch_dn_values,WT_error_scenarios_test):
    ## clear the market using the exact JCC model

    new_prob = gp.Model('new_problem')

    ### lower-level primal variables ###
    p_ch = new_prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='p_ch')
    p_dis = new_prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='p_dis')
    p = new_prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='p')
    r_dis_up = new_prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_dis_up')
    r_ch_up = new_prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_ch_up')
    r_dis_dn = new_prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_dis_dn')
    r_ch_dn = new_prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_ch_dn')
    r_up = new_prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_up')
    r_dn = new_prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_dn')
    w_sch = new_prob.addMVar((T, num_WT), lb=-GRB.INFINITY, ub=GRB.INFINITY,
                             name='w_sch')  # the scheduled power output of wind turbines
    w_cur = new_prob.addMVar((T, num_WT), lb=-GRB.INFINITY, ub=GRB.INFINITY,
                             name='w_cur')  # the curtailment of wind turbines
    r_wm_up = new_prob.addMVar(T, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_wm_up')
    r_wm_dn = new_prob.addMVar(T, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_wm_dn')

    z = new_prob.addMVar(N, vtype=GRB.BINARY, name='z')
    u = new_prob.addMVar(1, lb=0, ub=GRB.INFINITY, name='u')
    v = new_prob.addMVar(N, lb=0, ub=GRB.INFINITY, name='v')

    new_prob.addConstr(epsilon * N * u - gp.quicksum(v) >= theta * N)
    new_prob.addConstr(gp.quicksum(z) <= k)

    for i in range(N):
        new_prob.addConstr(M * (1 - z[i]) >= u - v[i])
    for t in range(T):
        N_P_UP = np.where(random_var_scenarios[:, t] < Q_WM_UP[t])[0]
        N_P_DN = np.where(-random_var_scenarios[:, t] < Q_WM_DN[t])[0]
        for i in N_P_UP:
            new_prob.addConstr(
                r_wm_up[t] + random_var_scenarios[i, t] + (- random_var_scenarios[i, t] + Q_WM_UP[t]) * z[i] >= u
                - v[i])
        for i in N_P_DN:
            new_prob.addConstr(
                r_wm_dn[t] - random_var_scenarios[i, t] + (random_var_scenarios[i, t] + Q_WM_DN[t]) * z[i] >= u
                - v[i])

    for t in range(T):
        new_prob.addConstr(Q_WM_UP[t] - (u - r_wm_up[t]) >= 0)
        new_prob.addConstr(Q_WM_DN[t] - (u - r_wm_dn[t]) >= 0)

    for t in range(T):
        # power balance equation
        new_prob.addConstr(-(p[t, :].sum() + p_dis[t, :].sum() - p_ch[t, :].sum() + w_sch[t, :].sum()) == -(LOAD[t]),
                           name=f'power_balance_{t}')

        new_prob.addConstr(-(r_ch_up[t, :].sum() + r_dis_up[t, :].sum() + r_up[t, :].sum() - r_wm_up[t] -
                             R_UP_EX[t]) <= 0, name=f'reserve_up_{t}')

        new_prob.addConstr(- (r_ch_dn[t, :].sum() + r_dis_dn[t, :].sum() + r_dn[t, :].sum() - r_wm_dn[t] -
                              R_DN_EX[t]) <= 0, name=f'reserve_down_{t}')
        for s in range(num_storage):
            new_prob.addConstr(p_ch[t, s] >= 0)
            new_prob.addConstr(p_ch[t, s] <= p_hat_ch_values[t, s])

            new_prob.addConstr(p_dis[t, s] >= 0)
            new_prob.addConstr(p_dis[t, s] <= p_hat_dis_values[t, s])

            new_prob.addConstr(r_ch_up[t, s] >= 0)
            new_prob.addConstr(r_ch_up[t, s] <= r_hat_ch_up_values[t, s])

            new_prob.addConstr(r_ch_dn[t, s] >= 0)
            new_prob.addConstr(r_ch_dn[t, s] <= r_hat_ch_dn_values[t, s])

            new_prob.addConstr(r_dis_up[t, s] >= 0)
            new_prob.addConstr(r_dis_up[t, s] <= r_hat_dis_up_values[t, s])

            new_prob.addConstr(r_dis_dn[t, s] >= 0)
            new_prob.addConstr(r_dis_dn[t, s] <= r_hat_dis_dn_values[t, s])

            new_prob.addConstr(r_ch_up[t, s] - p_ch[t, s] <= 0)
            new_prob.addConstr(r_dis_dn[t, s] - p_dis[t, s] <= 0)

        for g in range(num_gen):
            new_prob.addConstr(p[t, g] >= P_MIN[g])
            new_prob.addConstr(p[t, g] <= P_MAX[g])

            new_prob.addConstr(r_up[t, g] >= 0)
            new_prob.addConstr(r_up[t, g] <= R_MAX_UP[g])

            new_prob.addConstr(r_dn[t, g] >= 0)
            new_prob.addConstr(r_dn[t, g] <= R_MAX_DN[g])

            new_prob.addConstr(p[t, g] + r_up[t, g] <= P_MAX[g])
            new_prob.addConstr(r_dn[t, g] - p[t, g] <= -P_MIN[g])

        for j in range(num_WT):
            # wind balance constraint
            new_prob.addConstr(w_sch[t, j] + w_cur[t, j] == W_FORE[t, j])

            new_prob.addConstr(w_cur[t, j] >= 0)
            new_prob.addConstr(w_cur[t, j] <= W_FORE[t, j])

    new_prob.setObjective(
        gp.quicksum(b_hat_ch_values[t, s] * p_ch[t, s] - b_hat_dis_values[t, s] * p_dis[t, s]
                    - (b_hat_ch_up_values[t, s] * r_ch_up[t, s] + b_hat_ch_dn_values[t, s] * r_ch_dn[t, s]
                       + b_hat_dis_up_values[t, s] * r_dis_up[t, s] + b_hat_dis_dn_values[t, s] *
                       r_dis_dn[t, s]) for t in range(T) for s in range(num_storage))
        - gp.quicksum(
            c[t, g] * p[t, g] + c_rs[t, g] * (r_up[t, g] + r_dn[t, g]) for t in range(T) for g in range(num_gen))
        - gp.quicksum(c_cur[t, j] * w_cur[t, j] for t in range(T) for j in range(num_WT)),
        gp.GRB.MAXIMIZE)

    new_prob.setParam('MIPGap', 1e-9)
    new_prob.setParam('IntFeasTol', 1e-9)
    new_prob.setParam('FeasibilityTol', 1e-9)
    new_prob.setParam('OptimalityTol', 1e-9)
    new_prob.setParam('OutputFlag', 0)
    new_prob.optimize()

    if new_prob.status != GRB.OPTIMAL:
        print('The problem does not have an optimal solution.')
        raise Exception('The problem does not have an optimal solution.')

    r_wm_up_values = np.array([new_prob.getVarByName(f'r_wm_up[{t}]').X for t in range(T)])
    r_wm_dn_values = np.array([new_prob.getVarByName(f'r_wm_dn[{t}]').X for t in range(T)])

    random_var_scenarios_test = WT_error_scenarios_test.reshape(WT_error_scenarios_test.shape[0], -1)

    constraints_up_satisfied = r_wm_up_values >= -random_var_scenarios_test
    constraints_dn_satisfied = r_wm_dn_values >= random_var_scenarios_test

    combined_constraints = np.concatenate([constraints_up_satisfied, constraints_dn_satisfied], axis=1)

    reliability_combined_exact = np.mean(np.all(combined_constraints, axis=1)) * 100

    return new_prob.ObjVal, new_prob, reliability_combined_exact


def calculate_prices(LOAD, R_UP_EX, R_DN_EX, T, num_storage, num_gen, num_WT, N, epsilon, theta, k, M,
                     random_var_scenarios,
                     Q_WM_UP, Q_WM_DN, P_MIN, P_MAX, R_MAX_UP, R_MAX_DN, W_FORE, c, c_rs, c_cur,
                     b_hat_ch_values, b_hat_dis_values, b_hat_dis_up_values, b_hat_ch_up_values,
                     b_hat_dis_dn_values, b_hat_ch_dn_values, p_hat_ch_values, p_hat_dis_values,
                     r_hat_dis_up_values, r_hat_ch_up_values, r_hat_dis_dn_values, r_hat_ch_dn_values,WT_error_scenarios_test):
    ### calculate the market clearing prices using the exact JCC model and sensitivity analysis (marginal price)

    small_increase = 0.001  # Small increment for sensitivity analysis

    energy_prices = []
    reserve_up_prices = []
    reserve_down_prices = []

    # get the original objective value
    obj_val_original, _, reliability_combined = market_clear_exact_JCC(LOAD, R_UP_EX, R_DN_EX, T, num_storage, num_gen, num_WT, N, epsilon,
                                                 theta, k, M, random_var_scenarios,
                                                 Q_WM_UP, Q_WM_DN, P_MIN, P_MAX, R_MAX_UP, R_MAX_DN, W_FORE, c, c_rs,
                                                 c_cur,
                                                 b_hat_ch_values, b_hat_dis_values, b_hat_dis_up_values,
                                                 b_hat_ch_up_values, b_hat_dis_dn_values, b_hat_ch_dn_values,
                                                 p_hat_ch_values, p_hat_dis_values, r_hat_dis_up_values,
                                                 r_hat_ch_up_values, r_hat_dis_dn_values, r_hat_ch_dn_values,
                                                 WT_error_scenarios_test)

    # Energy price sensitivity
    for t in range(T):
        D_ex_increase = np.copy(LOAD)
        D_ex_increase[t] += small_increase

        obj_val_increase, _, __ = market_clear_exact_JCC(D_ex_increase, R_UP_EX, R_DN_EX, T, num_storage, num_gen, num_WT, N,
                                                     epsilon, theta, k, M, random_var_scenarios,
                                                     Q_WM_UP, Q_WM_DN, P_MIN, P_MAX, R_MAX_UP, R_MAX_DN, W_FORE, c, c_rs,
                                                     c_cur,
                                                     b_hat_ch_values, b_hat_dis_values, b_hat_dis_up_values,
                                                     b_hat_ch_up_values, b_hat_dis_dn_values, b_hat_ch_dn_values,
                                                     p_hat_ch_values, p_hat_dis_values, r_hat_dis_up_values,
                                                     r_hat_ch_up_values, r_hat_dis_dn_values, r_hat_ch_dn_values,WT_error_scenarios_test)

        energy_price = -(obj_val_increase - obj_val_original) / small_increase
        energy_prices.append(energy_price)
        print(f"Energy price at time {t}: {energy_price}")

    # Reserve up price sensitivity
    for t in range(T):
        R_ex_up_increase = R_UP_EX.copy()
        R_ex_up_increase[t] += small_increase

        obj_val_increase, _, __ = market_clear_exact_JCC(LOAD, R_ex_up_increase, R_DN_EX, T, num_storage, num_gen, num_WT, N,
                                                     epsilon, theta, k, M, random_var_scenarios,
                                                     Q_WM_UP, Q_WM_DN, P_MIN, P_MAX, R_MAX_UP, R_MAX_DN, W_FORE, c, c_rs,
                                                     c_cur,
                                                     b_hat_ch_values, b_hat_dis_values, b_hat_dis_up_values,
                                                     b_hat_ch_up_values, b_hat_dis_dn_values, b_hat_ch_dn_values,
                                                     p_hat_ch_values, p_hat_dis_values, r_hat_dis_up_values,
                                                     r_hat_ch_up_values, r_hat_dis_dn_values, r_hat_ch_dn_values,WT_error_scenarios_test)

        reserve_up_price = -(obj_val_increase - obj_val_original) / small_increase
        reserve_up_prices.append(reserve_up_price)
        print(f"Reserve up price at time {t}: {reserve_up_price}")

    # Reserve down price sensitivity
    for t in range(T):
        R_ex_dn_increase = R_DN_EX.copy()
        R_ex_dn_increase[t] += small_increase

        obj_val_increase, _, __ = market_clear_exact_JCC(LOAD, R_UP_EX, R_ex_dn_increase, T, num_storage, num_gen, num_WT, N,
                                                     epsilon, theta, k, M, random_var_scenarios,
                                                     Q_WM_UP, Q_WM_DN, P_MIN, P_MAX, R_MAX_UP, R_MAX_DN, W_FORE, c, c_rs,
                                                     c_cur,
                                                     b_hat_ch_values, b_hat_dis_values, b_hat_dis_up_values,
                                                     b_hat_ch_up_values, b_hat_dis_dn_values, b_hat_ch_dn_values,
                                                     p_hat_ch_values, p_hat_dis_values, r_hat_dis_up_values,
                                                     r_hat_ch_up_values, r_hat_dis_dn_values, r_hat_ch_dn_values,WT_error_scenarios_test)

        reserve_down_price = -(obj_val_increase - obj_val_original) / small_increase
        reserve_down_prices.append(reserve_down_price)
        print(f"Reserve down price at time {t}: {reserve_down_price}")

    return energy_prices, reserve_up_prices, reserve_down_prices, reliability_combined



def calculate_prices_with_fixed_binary(LOAD, R_UP_EX, R_DN_EX, T, num_storage, num_gen, num_WT, N, epsilon, theta, k, M,
                                       random_var_scenarios, Q_WM_UP, Q_WM_DN, P_MIN, P_MAX, R_MAX_UP, R_MAX_DN, W_FORE, c,
                                       c_rs, c_cur, b_hat_ch_values, b_hat_dis_values, b_hat_dis_up_values,
                                       b_hat_ch_up_values, b_hat_dis_dn_values, b_hat_ch_dn_values, p_hat_ch_values,
                                       p_hat_dis_values, r_hat_dis_up_values, r_hat_ch_up_values, r_hat_dis_dn_values,
                                       r_hat_ch_dn_values, WT_error_scenarios_test, c_ch, c_dis):

    '''
    This function is used for two folds:
    1. Calculate and return the cleared quantity (energy and reserve) for the market clearing with exact WDRJCC
    2. Calculate the marginal prices by fixing the binaries to the optimal solution. This is not the true marginal prices
    due to non-convexity. We calculate this and then the total profits simply for investigation but did not use it for our case study results.
    '''

    _, new_prob, __ = market_clear_exact_JCC(LOAD, R_UP_EX, R_DN_EX, T, num_storage, num_gen, num_WT, N, epsilon, theta, k, M,
                                         random_var_scenarios, Q_WM_UP, Q_WM_DN, P_MIN, P_MAX, R_MAX_UP, R_MAX_DN, W_FORE, c,
                                         c_rs, c_cur, b_hat_ch_values, b_hat_dis_values, b_hat_dis_up_values,
                                         b_hat_ch_up_values, b_hat_dis_dn_values, b_hat_ch_dn_values, p_hat_ch_values,
                                         p_hat_dis_values, r_hat_dis_up_values, r_hat_ch_up_values, r_hat_dis_dn_values,
                                         r_hat_ch_dn_values, WT_error_scenarios_test)

    fixed_model = new_prob.fixed()
    fixed_model.optimize()

    if fixed_model.status == GRB.OPTIMAL:
        lambda_en = np.array([fixed_model.getConstrByName(f'power_balance_{t}').Pi for t in range(T)])
        lambda_up = np.array([fixed_model.getConstrByName(f'reserve_up_{t}').Pi for t in range(T)])
        lambda_dn = np.array([fixed_model.getConstrByName(f'reserve_down_{t}').Pi for t in range(T)])

        print("Fixed Model Energy Prices (Duals):", lambda_en)
        print("Fixed Model Reserve Up Prices (Duals):", lambda_up)
        print("Fixed Model Reserve Down Prices (Duals):", lambda_dn)

        # Initialize NumPy arrays to store values for each storage unit across all time periods
        p_ch_values = np.zeros((T, num_storage))
        p_dis_values = np.zeros((T, num_storage))
        r_ch_up_values = np.zeros((T, num_storage))
        r_dis_up_values = np.zeros((T, num_storage))
        r_ch_dn_values = np.zeros((T, num_storage))
        r_dis_dn_values = np.zeros((T, num_storage))

        total_profit = 0.0
        for t in range(T):
            for s in range(num_storage):
                # Get the values of variables from new_prob
                p_ch_values[t, s] = new_prob.getVarByName(f'p_ch[{t},{s}]').X
                p_dis_values[t, s] = new_prob.getVarByName(f'p_dis[{t},{s}]').X
                r_ch_up_values[t, s] = new_prob.getVarByName(f'r_ch_up[{t},{s}]').X
                r_dis_up_values[t, s] = new_prob.getVarByName(f'r_dis_up[{t},{s}]').X
                r_ch_dn_values[t, s] = new_prob.getVarByName(f'r_ch_dn[{t},{s}]').X
                r_dis_dn_values[t, s] = new_prob.getVarByName(f'r_dis_dn[{t},{s}]').X

                # Compute the total profit based on the dual variables and these values
                total_profit += (- (lambda_en[t] + c_ch[s]) * p_ch_values[t, s]
                                 + (lambda_en[t] - c_dis[s]) * p_dis_values[t, s]
                                 + lambda_up[t] * (r_ch_up_values[t, s] + r_dis_up_values[t, s])
                                 + lambda_dn[t] * (r_ch_dn_values[t, s] + r_dis_dn_values[t, s]))

        print(f"Profit from MIP Model (quantity) and Fixed Binary Variables Model (price): {total_profit}")
        # the following is simply for verifying fixing the binaries then resolving for getting the duals will not alter the oroginal optimal solutions
        # as mentioned, the profits calculated here are simply for investigation but not used for our case study results.
        total_profit_fixed = 0.0
        for t in range(T):
            for s in range(num_storage):
                # Get the values of variables from fixed_model
                p_ch_value_fixed = fixed_model.getVarByName(f'p_ch[{t},{s}]').X
                p_dis_value_fixed = fixed_model.getVarByName(f'p_dis[{t},{s}]').X
                r_ch_up_value_fixed = fixed_model.getVarByName(f'r_ch_up[{t},{s}]').X
                r_dis_up_value_fixed = fixed_model.getVarByName(f'r_dis_up[{t},{s}]').X
                r_ch_dn_value_fixed = fixed_model.getVarByName(f'r_ch_dn[{t},{s}]').X
                r_dis_dn_value_fixed = fixed_model.getVarByName(f'r_dis_dn[{t},{s}]').X

                # Compute the total profit based on the dual variables and these values
                total_profit_fixed += (- (lambda_en[t] + c_ch[s]) * p_ch_value_fixed
                                 + (lambda_en[t] - c_dis[s]) * p_dis_value_fixed
                                 + lambda_up[t] * (r_ch_up_value_fixed + r_dis_up_value_fixed)
                                 + lambda_dn[t] * (r_ch_dn_value_fixed + r_dis_dn_value_fixed))

        print(f"Profit from Fixed Binary Variables Model (quantity) and Fixed Binary Variables Model (price):  {total_profit_fixed}")

        # Return the stored values
        return p_ch_values, p_dis_values, r_ch_up_values, r_dis_up_values, r_ch_dn_values, r_dis_dn_values

    else:
        print("Fixed model did not solve to optimality.")
        return None, None, None, None, None, None


def solve_bilevel(T, N, M, theta, epsilon, WT_error_scenarios_train, num_storage, num_gen, num_WT, LOAD, R_UP_EX,
                  R_DN_EX, P_MIN, P_MAX, R_MAX_UP, R_MAX_DN, W_FORE, P_MAX_DIS, R_MAX_DIS_UP, R_MAX_DIS_DN, rng, gurobi_seed,
                  P_MAX_CH, R_MAX_CH_UP, R_MAX_CH_DN, E_MAX, E_INI, eta, storage_alpha, c, c_rs, c_cur, c_ch, c_dis,
                  method, numerical_focus = False, log_file_name = None, IntegralityFocus=0, thread=32, n_jobs_bonf_cal_var=1, TimeLimit=3600):
    # this function solves the bilevel optimization problem with JCC

    # the parameter for the proposed method
    k = np.floor(N * epsilon).astype(int)
    kappa = 1 * np.ones(N)

    # the following two are only for wcvar method
    w_cvar_up = np.ones(T) / (2 * T)
    w_cvar_dn = np.ones(T) / (2 * T)

    # only select N scenarios from the training set
    random_var_scenario_index = rng.choice(WT_error_scenarios_train.shape[0], N, replace=False)
    random_var_scenarios = WT_error_scenarios_train[random_var_scenario_index, :]

    Q_WM_UP = np.zeros(T)
    Q_WM_DN = np.zeros(T)

    for t in range(T):
        Q_WM_UP[t] = np.sort(random_var_scenarios[:,t])[k]
        Q_WM_DN[t] = np.sort(-random_var_scenarios[:,t])[k]

    if method == 'bonferroni':
        eta_up, eta_dn = solve_all_VaR(N, theta, T, epsilon, random_var_scenarios, n_jobs_bonf_cal_var)
        print("eta_up Results:", eta_up)
        print("eta_dn Results:", eta_dn)

    M1_mu_min = M
    M1_mu_max = M
    M1_mu_min_ch = M
    M1_mu_max_ch = M
    M1_mu_min_dis = M
    M1_mu_max_dis = M
    M1_mu_min_ch_up = M
    M1_mu_max_ch_up = M
    M1_mu_min_ch_dn = M
    M1_mu_max_ch_dn = M
    M1_mu_min_dis_up = M
    M1_mu_max_dis_up = M
    M1_mu_min_dis_dn = M
    M1_mu_max_dis_dn = M
    M1_mu_ch_up = M
    M1_mu_dis_dn = M
    M1_mu_min_up = M
    M1_mu_max_up = M
    M1_mu_min_dn = M
    M1_mu_max_dn = M
    M1_mu_up = M
    M1_mu_dn = M
    M1_mu_min_w_cur = M
    M1_mu_max_w_cur = M
    M1_mu_min_u = M
    M1_mu_min_v = M
    M1_mu_1 = M
    M1_mu_2 = M
    M1_mu_3 = M
    M1_mu_4 = M
    M1_mu_5 = M
    M1_lambda_up = M
    M1_lambda_dn = M
    M1_mu_min_alpha = M


    M2_mu_min = M
    M2_mu_max = M
    M2_mu_min_ch = M
    M2_mu_max_ch = M
    M2_mu_min_dis = M
    M2_mu_max_dis = M
    M2_mu_min_ch_up = M
    M2_mu_max_ch_up = M
    M2_mu_min_ch_dn = M
    M2_mu_max_ch_dn = M
    M2_mu_min_dis_up = M
    M2_mu_max_dis_up = M
    M2_mu_min_dis_dn = M
    M2_mu_max_dis_dn = M
    M2_mu_ch_up = M
    M2_mu_dis_dn = M
    M2_mu_min_up = M
    M2_mu_max_up = M
    M2_mu_min_dn = M
    M2_mu_max_dn = M
    M2_mu_up = M
    M2_mu_dn = M
    M2_mu_min_w_cur = M
    M2_mu_max_w_cur = M
    M2_mu_min_u = M
    M2_mu_min_v = M
    M2_mu_1 = M
    M2_mu_2 = M
    M2_mu_3 = M
    M2_mu_4 = M
    M2_mu_5 = M
    M2_lambda_up = M
    M2_lambda_dn = M
    M2_mu_min_alpha = M

    # perform bilevel optimization
    prob = gp.Model('bilevel')

    ### upper-level variables ###
    p_hat_ch = prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='p_hat_ch')
    p_hat_dis = prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='p_hat_dis')
    r_hat_dis_up = prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_hat_dis_up')
    r_hat_ch_up = prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_hat_ch_up')
    r_hat_dis_dn = prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_hat_dis_dn')
    r_hat_ch_dn = prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_hat_ch_dn')
    u_ch = prob.addMVar((T, num_storage), vtype=GRB.BINARY, name='u_ch')
    u_dis = prob.addMVar((T, num_storage), vtype=GRB.BINARY, name='u_dis')
    b_hat_ch = prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='b_hat_ch')
    b_hat_dis = prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='b_hat_dis')
    b_hat_dis_up = prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='b_hat_dis_up')
    b_hat_ch_up = prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='b_hat_ch_up')
    b_hat_dis_dn = prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='b_hat_dis_dn')
    b_hat_ch_dn = prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='b_hat_ch_dn')
    e = prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='e')

    ### lower-level primal variables ###
    p_ch = prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='p_ch')
    p_dis = prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='p_dis')
    p = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='p')
    r_dis_up = prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_dis_up')
    r_ch_up = prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_ch_up')
    r_dis_dn = prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_dis_dn')
    r_ch_dn = prob.addMVar((T, num_storage), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_ch_dn')
    r_up = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_up')
    r_dn = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_dn')
    w_sch = prob.addMVar((T, num_WT), lb=-GRB.INFINITY, ub=GRB.INFINITY, name = 'w_sch')  # the scheduled power output of wind turbines
    w_cur = prob.addMVar((T, num_WT), lb=-GRB.INFINITY, ub=GRB.INFINITY, name = 'w_cur')  # the curtailment of wind turbines
    r_wm_up = prob.addMVar(T, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_wm_up')
    r_wm_dn = prob.addMVar(T, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='r_wm_dn')


    ### lower-level dual variables ###
    lambda_en = prob.addMVar(T, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='lambda_en')
    lambda_up = prob.addMVar(T, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='lambda_up')
    lambda_dn = prob.addMVar(T, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='lambda_dn')
    mu_min = prob.addMVar((T, num_gen), lb=0, ub=GRB.INFINITY)
    mu_max = prob.addMVar((T, num_gen), lb=0, ub=GRB.INFINITY)
    mu_min_ch = prob.addMVar((T, num_storage), lb=0, ub=GRB.INFINITY, name='mu_min_ch')
    mu_max_ch = prob.addMVar((T, num_storage), lb=0, ub=GRB.INFINITY, name='mu_max_ch')
    mu_min_dis = prob.addMVar((T, num_storage), lb=0, ub=GRB.INFINITY, name='mu_min_dis')
    mu_max_dis = prob.addMVar((T, num_storage), lb=0, ub=GRB.INFINITY, name='mu_max_dis')
    mu_min_ch_up = prob.addMVar((T, num_storage), lb=0, ub=GRB.INFINITY, name='mu_min_ch_up')
    mu_max_ch_up = prob.addMVar((T, num_storage), lb=0, ub=GRB.INFINITY, name='mu_max_ch_up')
    mu_min_ch_dn = prob.addMVar((T, num_storage), lb=0, ub=GRB.INFINITY, name='mu_min_ch_dn')
    mu_max_ch_dn = prob.addMVar((T, num_storage), lb=0, ub=GRB.INFINITY, name='mu_max_ch_dn')
    mu_min_dis_up = prob.addMVar((T, num_storage), lb=0, ub=GRB.INFINITY, name='mu_min_dis_up')
    mu_max_dis_up = prob.addMVar((T, num_storage), lb=0, ub=GRB.INFINITY, name='mu_max_dis_up')
    mu_min_dis_dn = prob.addMVar((T, num_storage), lb=0, ub=GRB.INFINITY, name='mu_min_dis_dn')
    mu_max_dis_dn = prob.addMVar((T, num_storage), lb=0, ub=GRB.INFINITY, name='mu_max_dis_dn')
    mu_ch_up = prob.addMVar((T, num_storage), lb=0, ub=GRB.INFINITY, name='mu_ch_up')
    mu_dis_dn = prob.addMVar((T, num_storage), lb=0, ub=GRB.INFINITY, name='mu_dis_dn')
    mu_min_up = prob.addMVar((T, num_gen), lb=0, ub=GRB.INFINITY, name='mu_min_up')
    mu_max_up = prob.addMVar((T, num_gen), lb=0, ub=GRB.INFINITY)
    mu_min_dn = prob.addMVar((T, num_gen), lb=0, ub=GRB.INFINITY, name='mu_min_dn')
    mu_max_dn = prob.addMVar((T, num_gen), lb=0, ub=GRB.INFINITY)
    mu_up = prob.addMVar((T, num_gen), lb=0, ub=GRB.INFINITY, name='mu_up')
    mu_dn = prob.addMVar((T, num_gen), lb=0, ub=GRB.INFINITY, name='mu_dn')
    mu_w_sch = prob.addMVar((T, num_WT), lb=0, ub=GRB.INFINITY, name='mu_w_sch')
    mu_min_w_cur = prob.addMVar((T, num_WT), lb=0, ub=GRB.INFINITY, name='mu_min_w_cur')
    mu_max_w_cur = prob.addMVar((T, num_WT), lb=0, ub=GRB.INFINITY, name='mu_max_w_cur')

    if method == 'proposed':
        u = prob.addMVar(1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='u')
        v = prob.addMVar(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='v')
        mu_min_u = prob.addMVar(1, lb=0, ub=GRB.INFINITY, name='mu_min_u')
        mu_min_v = prob.addMVar(N, lb=0, ub=GRB.INFINITY, name='mu_min_v')
        mu_1 = prob.addMVar(1, lb=0, ub=GRB.INFINITY, name='mu_1')
        mu_4 = prob.addMVar(T, lb=0, ub=GRB.INFINITY, name='mu_4')
        mu_5 = prob.addMVar(T, lb=0, ub=GRB.INFINITY, name='mu_5')
        mu_2 = {}  # Initialize mu_2 as an empty dictionary
        mu_3 = {}  # Initialize mu_3 as an empty dictionary
        for t in range(T):
            N_P_UP = np.where(random_var_scenarios[:, t] < Q_WM_UP[t])[0]
            N_P_DN = np.where(-random_var_scenarios[:, t] < Q_WM_DN[t])[0]
            # Define mu_2 and mu_3 only for indices in N_P_UP and N_P_DN, respectively
            for i in N_P_UP:
                mu_2[t, i] = prob.addVar(lb=0, ub=GRB.INFINITY)
            for i in N_P_DN:
                mu_3[t, i] = prob.addVar(lb=0, ub=GRB.INFINITY)
    elif method == 'linearforN':
        u = prob.addMVar(1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='u')
        v = prob.addMVar(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='v')
        mu_min_u = prob.addMVar(1, lb=0, ub=GRB.INFINITY, name='mu_min_u')
        mu_min_v = prob.addMVar(N, lb=0, ub=GRB.INFINITY, name='mu_min_v')
        mu_1 = prob.addMVar(1, lb=0, ub=GRB.INFINITY, name='mu_1')
        mu_2 = prob.addMVar((T, N), lb=0, ub=GRB.INFINITY, name='mu_2')
        mu_3 = prob.addMVar((T, N), lb=0, ub=GRB.INFINITY, name='mu_3')
    elif method == 'wcvar':
        alpha = prob.addMVar(N, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        beta = prob.addMVar(1, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        tau = prob.addMVar(1, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        mu_min_alpha = prob.addMVar(N, lb=0, ub=GRB.INFINITY, name='mu_min_alpha')
        mu_1 = prob.addMVar(1, lb=0, ub=GRB.INFINITY, name='mu_1')
        mu_2 = prob.addMVar((T, N), lb=0, ub=GRB.INFINITY, name='mu_2')
        mu_3 = prob.addMVar((T, N), lb=0, ub=GRB.INFINITY, name='mu_3')
        mu_4 = prob.addMVar(T, lb=0, ub=GRB.INFINITY, name='mu_4')
        mu_5 = prob.addMVar(T, lb=0, ub=GRB.INFINITY, name='mu_5')
    elif method == 'bonferroni':
        mu_1 = prob.addMVar(T, lb=0, ub=GRB.INFINITY, name='mu_1')
        mu_2 = prob.addMVar(T, lb=0, ub=GRB.INFINITY, name='mu_2')



    ### linearize KKT conditions with Big-M -> binary variables ###
    bin_mu_min = prob.addMVar((T, num_gen), vtype=GRB.BINARY)
    bin_mu_max = prob.addMVar((T, num_gen), vtype=GRB.BINARY)
    bin_mu_min_ch = prob.addMVar((T, num_storage), vtype=GRB.BINARY)
    bin_mu_max_ch = prob.addMVar((T, num_storage), vtype=GRB.BINARY)
    bin_mu_min_dis = prob.addMVar((T, num_storage), vtype=GRB.BINARY)
    bin_mu_max_dis = prob.addMVar((T, num_storage), vtype=GRB.BINARY)
    bin_mu_min_ch_up = prob.addMVar((T, num_storage), vtype=GRB.BINARY)
    bin_mu_max_ch_up = prob.addMVar((T, num_storage), vtype=GRB.BINARY)
    bin_mu_min_ch_dn = prob.addMVar((T, num_storage), vtype=GRB.BINARY)
    bin_mu_max_ch_dn = prob.addMVar((T, num_storage), vtype=GRB.BINARY)
    bin_mu_min_dis_up = prob.addMVar((T, num_storage), vtype=GRB.BINARY)
    bin_mu_max_dis_up = prob.addMVar((T, num_storage), vtype=GRB.BINARY)
    bin_mu_min_dis_dn = prob.addMVar((T, num_storage), vtype=GRB.BINARY)
    bin_mu_max_dis_dn = prob.addMVar((T, num_storage), vtype=GRB.BINARY)
    bin_mu_ch_up = prob.addMVar((T, num_storage), vtype=GRB.BINARY)
    bin_mu_dis_dn = prob.addMVar((T, num_storage), vtype=GRB.BINARY)
    bin_mu_min_up = prob.addMVar((T, num_gen), vtype=GRB.BINARY)
    bin_mu_max_up = prob.addMVar((T, num_gen), vtype=GRB.BINARY)
    bin_mu_min_dn = prob.addMVar((T, num_gen), vtype=GRB.BINARY)
    bin_mu_max_dn = prob.addMVar((T, num_gen), vtype=GRB.BINARY)
    bin_mu_up = prob.addMVar((T, num_gen), vtype=GRB.BINARY)
    bin_mu_dn = prob.addMVar((T, num_gen), vtype=GRB.BINARY)
    bin_mu_min_w_cur = prob.addMVar((T, num_WT), vtype=GRB.BINARY)
    bin_mu_max_w_cur = prob.addMVar((T, num_WT), vtype=GRB.BINARY)


    bin_lambda_up = prob.addMVar(T, vtype=GRB.BINARY)
    bin_lambda_dn = prob.addMVar(T, vtype=GRB.BINARY)


    if method == 'proposed':
        bin_mu_min_u = prob.addMVar(1, vtype=GRB.BINARY)
        bin_mu_min_v = prob.addMVar(N, vtype=GRB.BINARY)
        bin_mu_1 = prob.addMVar(1, vtype=GRB.BINARY)
        bin_mu_4 = prob.addMVar(T, vtype=GRB.BINARY)
        bin_mu_5 = prob.addMVar(T, vtype=GRB.BINARY)
        bin_mu_2 = {}
        bin_mu_3 = {}
        for t in range(T):
            N_P_UP = np.where(random_var_scenarios[:, t] < Q_WM_UP[t])[0]
            N_P_DN = np.where(-random_var_scenarios[:, t] < Q_WM_DN[t])[0]

            # Define mu_2 and mu_3 only for indices in N_P_UP and N_P_DN, respectively
            for i in N_P_UP:
                bin_mu_2[t, i] = prob.addVar(vtype=GRB.BINARY)
            for i in N_P_DN:
                bin_mu_3[t, i] = prob.addVar(vtype=GRB.BINARY)
    elif method == 'linearforN':
        bin_mu_min_u = prob.addMVar(1, vtype=GRB.BINARY)
        bin_mu_min_v = prob.addMVar(N, vtype=GRB.BINARY)
        bin_mu_1 = prob.addMVar(1, vtype=GRB.BINARY)
        bin_mu_2 = prob.addMVar((T, N), vtype=GRB.BINARY)
        bin_mu_3 = prob.addMVar((T, N), vtype=GRB.BINARY)
    elif method == 'wcvar':
        bin_mu_min_alpha = prob.addMVar(N, vtype=GRB.BINARY)
        bin_mu_1 = prob.addMVar(1, vtype=GRB.BINARY)
        bin_mu_2 = prob.addMVar((T, N), vtype=GRB.BINARY)
        bin_mu_3 = prob.addMVar((T, N), vtype=GRB.BINARY)
        bin_mu_4 = prob.addMVar(T, vtype=GRB.BINARY)
        bin_mu_5 = prob.addMVar(T, vtype=GRB.BINARY)
    elif method == 'bonferroni':
        bin_mu_1 = prob.addMVar(T, vtype=GRB.BINARY)
        bin_mu_2 = prob.addMVar(T, vtype=GRB.BINARY)



    ### upper-level constraints ###
    for t in range(T):
        for s in range(num_storage):
            prob.addConstr(u_dis[t, s] + u_ch[t, s] <= 1)
            prob.addConstr(p_hat_dis[t, s] >= 0)
            prob.addConstr(p_hat_dis[t, s] <=u_dis[t, s] * P_MAX_DIS[s])
            prob.addConstr(r_hat_dis_up[t, s] >= 0)
            prob.addConstr(r_hat_dis_up[t, s] <= u_dis[t, s] * R_MAX_DIS_UP[s])
            prob.addConstr(r_hat_dis_dn[t, s] >= 0)
            prob.addConstr(r_hat_dis_dn[t, s] <= u_dis[t, s] * R_MAX_DIS_DN[s])
            prob.addConstr(p_hat_dis[t, s] + r_hat_dis_up[t, s] <= u_dis[t, s] * P_MAX_DIS[s])
            prob.addConstr(r_hat_dis_dn[t, s] - p_hat_dis[t, s] <= 0)
            prob.addConstr(p_hat_ch[t, s] >= 0)
            prob.addConstr(p_hat_ch[t, s] <= u_ch[t, s] *P_MAX_CH[s])
            prob.addConstr(r_hat_ch_up[t, s] >= 0)
            prob.addConstr(r_hat_ch_up[t, s] <= u_ch[t, s] * R_MAX_CH_UP[s])
            prob.addConstr(r_hat_ch_dn[t, s] >= 0)
            prob.addConstr(r_hat_ch_dn[t, s] <= u_ch[t, s] * R_MAX_CH_DN[s])
            prob.addConstr(p_hat_ch[t, s] + r_hat_ch_dn[t, s] <= u_ch[t, s] * P_MAX_CH[s])
            prob.addConstr(r_hat_ch_up[t, s] - p_hat_ch[t, s] <= 0)
            prob.addConstr(b_hat_ch[t, s] >= 0)
            prob.addConstr(b_hat_ch_up[t, s] >= 0)
            prob.addConstr(b_hat_ch_dn[t, s] >= 0)
            prob.addConstr(b_hat_dis[t, s] >= 0)
            prob.addConstr(b_hat_dis_up[t, s] >= 0)
            prob.addConstr(b_hat_dis_dn[t, s] >= 0)

            prob.addConstr(E_INI[s] + sum(eta[s] * p_hat_ch[h, s] + eta[s] * r_hat_ch_dn[h, s] for h in range(t + 1)) - sum(
                1 / eta[s] * p_hat_dis[h, s] - 1 / eta[s] * r_hat_dis_dn[h, s] for h in range(t + 1)) <= E_MAX[s])

            prob.addConstr(E_INI[s] + sum(eta[s] * p_hat_ch[h, s] - eta[s] * r_hat_ch_up[h, s] for h in range(t + 1)) - sum(
                1 / eta[s] * p_hat_dis[h, s] + 1 / eta[s] * r_hat_dis_up[h, s] for h in range(t + 1)) >= 0)


    ### lower-level KKT ###
    if method == 'proposed':
        # the stationary condition for v_i, i in N
        for i in range(N):
            prob.addConstr(-mu_min_v[i] + mu_1 - gp.quicksum(
                mu_2[t, i] for t in range(T) if i in np.where(random_var_scenarios[:, t] < Q_WM_UP[t])[0]) - gp.quicksum(
                mu_3[t, i] for t in range(T) if i in np.where(-random_var_scenarios[:, t] < Q_WM_DN[t])[0]) == 0)

        # the stationary condition for u
        prob.addConstr(-mu_min_u - epsilon * N * mu_1 + gp.quicksum(mu_4) + gp.quicksum(mu_5) +
                       gp.quicksum(mu_2[t, i] for t in range(T) for i in np.where(random_var_scenarios[:, t] < Q_WM_UP[t])[0]) +
                       gp.quicksum(mu_3[t, i] for t in range(T) for i in np.where(-random_var_scenarios[:, t] < Q_WM_DN[t])[0])
                       == 0)
    elif method == 'linearforN':
        # the stationary condition for v_i, i in N
        for i in range(N):
            prob.addConstr(-mu_min_v[i] + mu_1 - gp.quicksum(mu_2[t, i] for t in range(T)) -
                           gp.quicksum(mu_3[t, i] for t in range(T)) == 0)
        # the stationary condition for u
        prob.addConstr(-mu_min_u - epsilon * N * mu_1 + gp.quicksum( mu_2[t, i] for t in range(T) for i in range(N)) +
                       gp.quicksum(mu_3[t, i] for t in range(T) for i in range(N)) == 0)
    elif method == 'wcvar':
        # the stationary condition for alpha_i, beta, eta
        for i in range(N):
            prob.addConstr(-mu_min_alpha[i] + mu_1/ (epsilon * N)  - gp.quicksum(mu_2[t, i] for t in range(T)) -
                           gp.quicksum(mu_3[t, i] for t in range(T)) == 0)
        prob.addConstr(mu_1 * theta / epsilon- gp.quicksum(mu_4[t] for t in range(T)) - gp.quicksum(
            mu_5[t] for t in range(T)) == 0)
        prob.addConstr(mu_1 - gp.quicksum(mu_2[t, i] for t in range(T) for i in range(N)) -
                       gp.quicksum(mu_3[t, i] for t in range(T) for i in range(N)) == 0)

    for t in range(T):
        # power balance equation
        prob.addConstr(p[t, :].sum() + p_dis[t, :].sum() - p_ch[t, :].sum() + w_sch[t, :].sum() == LOAD[t])
        for s in range(num_storage):
            # the stationary condition for p_dis, p_ch, r_dis_up, r_dis_dn, r_ch_up, r_ch_dn for s and t
            prob.addConstr(b_hat_dis[t, s] - lambda_en[t] + mu_max_dis[t, s] - mu_min_dis[t, s] - mu_dis_dn[t, s] == 0)
            prob.addConstr(-b_hat_ch[t, s] + lambda_en[t] + mu_max_ch[t, s] - mu_min_ch[t, s] - mu_ch_up[t, s] == 0)
            prob.addConstr(b_hat_dis_up[t, s] - lambda_up[t] + mu_max_dis_up[t, s] - mu_min_dis_up[t, s] == 0)
            prob.addConstr(
                b_hat_dis_dn[t, s] - lambda_dn[t] + mu_max_dis_dn[t, s] - mu_min_dis_dn[t, s] + mu_dis_dn[t, s] == 0)
            prob.addConstr(
                b_hat_ch_up[t, s] - lambda_up[t] + mu_max_ch_up[t, s] - mu_min_ch_up[t, s] + mu_ch_up[t, s] == 0)
            prob.addConstr(b_hat_ch_dn[t, s] - lambda_dn[t] + mu_max_ch_dn[t, s] - mu_min_ch_dn[t, s] == 0)
        for g in range(num_gen):
            # the stationary condition for p, r_ch, r_up for g and t
            prob.addConstr(c[t, g] - lambda_en[t] + mu_max[t, g] - mu_min[t, g] + mu_up[t, g] - mu_dn[t, g] == 0)
            prob.addConstr(c_rs[t, g] - lambda_up[t] + mu_max_up[t, g] - mu_min_up[t, g] + mu_up[t, g] == 0)
            prob.addConstr(c_rs[t, g] - lambda_dn[t] + mu_max_dn[t, g] - mu_min_dn[t, g] + mu_dn[t, g] == 0)


        for j in range(num_WT):
            # wind balance constraint
            prob.addConstr(w_sch[t, j] + w_cur[t, j] == W_FORE[t, j])
            # the stationary condition for w_sch and w_cur for j and t
            prob.addConstr(mu_w_sch[t, j] - lambda_en[t] == 0)
            prob.addConstr(c_cur[t, j] + mu_w_sch[t, j] + mu_max_w_cur[t, j] - mu_min_w_cur[t, j] == 0)
        if method == 'proposed':
            # calculate N_P for every t and every upward and downward JCC constraints, i.e., two Np for up and down
            N_P_UP = np.where(random_var_scenarios[:, t]< Q_WM_UP[t])[0]
            N_P_DN = np.where(-random_var_scenarios[:, t] < Q_WM_DN[t])[0]

            # the stationary condition for r_wm_up, r_wm_dn for t, with N_P_UP and N_P_DN respectively
            prob.addConstr(lambda_up[t] - gp.quicksum(kappa[i] * mu_2[t, i] for i in N_P_UP) - mu_4[t] == 0)
            prob.addConstr(lambda_dn[t] - gp.quicksum(kappa[i] * mu_3[t, i] for i in N_P_DN) - mu_5[t] == 0)
        elif method == 'linearforN':
            prob.addConstr(lambda_up[t] - gp.quicksum(kappa[i] * mu_2[t, i] for i in range(N)) == 0)
            prob.addConstr(lambda_dn[t] - gp.quicksum(kappa[i] * mu_3[t, i] for i in range(N)) == 0)
        elif method == 'wcvar':
            prob.addConstr(lambda_up[t] - w_cvar_up[t] *gp.quicksum(mu_2[t, i] for i in range(N)) == 0)
            prob.addConstr(lambda_dn[t] - w_cvar_dn[t] * gp.quicksum( mu_3[t, i] for i in range(N)) == 0)
        elif method == 'bonferroni':
            prob.addConstr(lambda_up[t] - mu_1[t] == 0)
            prob.addConstr(lambda_dn[t] - mu_2[t] == 0)

    ## linearization ###
    if method == 'proposed':
        for i in range(N):
            linearize_complementarity(prob, v[i], mu_min_v[i], bin_mu_min_v[i], M1_mu_min_v, M2_mu_min_v)
        linearize_complementarity(prob, u, mu_min_u, bin_mu_min_u, M1_mu_min_u, M2_mu_min_u)
        linearize_complementarity(prob, -theta * N + epsilon * N * u - gp.quicksum(v), mu_1, bin_mu_1, M1_mu_1,
                                  M2_mu_1)
    elif method == 'linearforN':
        for i in range(N):
            linearize_complementarity(prob, v[i], mu_min_v[i], bin_mu_min_v[i], M1_mu_min_v, M2_mu_min_v)
        linearize_complementarity(prob, u, mu_min_u, bin_mu_min_u, M1_mu_min_u, M2_mu_min_u)
        linearize_complementarity(prob, -theta * N + epsilon * N * u - gp.quicksum(v), mu_1, bin_mu_1, M1_mu_1,
                                  M2_mu_1)
    elif method == 'wcvar':
        for i in range(N):
            linearize_complementarity(prob, alpha[i], mu_min_alpha[i], bin_mu_min_alpha[i], M1_mu_min_alpha,
                                      M2_mu_min_alpha)
        linearize_complementarity(prob, -tau - 1 / epsilon * (theta * beta + 1 / N * gp.quicksum(alpha)), mu_1,
                                  bin_mu_1, M1_mu_1, M2_mu_1)

    for t in range(T):
        linearize_complementarity(prob,
                                  r_ch_up[t, :].sum() + r_dis_up[t, :].sum() + r_up[t, :].sum() - r_wm_up[t] - R_UP_EX[
                                      t], lambda_up[t], bin_lambda_up[t], M1_lambda_up, M2_lambda_up)
        linearize_complementarity(prob,
                                  r_ch_dn[t, :].sum() + r_dis_dn[t, :].sum() + r_dn[t, :].sum() - r_wm_dn[t] - R_DN_EX[
                                      t], lambda_dn[t], bin_lambda_dn[t], M1_lambda_dn, M2_lambda_dn)
        for g in range(num_gen):
            linearize_complementarity(prob, p[t, g] - P_MIN[g], mu_min[t, g], bin_mu_min[t, g], M1_mu_min, M2_mu_min)
            linearize_complementarity(prob, P_MAX[g] - p[t, g], mu_max[t, g], bin_mu_max[t, g], M1_mu_max, M2_mu_max)
            linearize_complementarity(prob, r_up[t, g], mu_min_up[t, g], bin_mu_min_up[t, g], M1_mu_min_up,
                                      M2_mu_min_up)
            linearize_complementarity(prob, R_MAX_UP[g] - r_up[t, g], mu_max_up[t, g], bin_mu_max_up[t, g],
                                      M1_mu_max_up, M2_mu_max_up)
            linearize_complementarity(prob, r_dn[t, g], mu_min_dn[t, g], bin_mu_min_dn[t, g], M1_mu_min_dn,
                                      M2_mu_min_dn)
            linearize_complementarity(prob, R_MAX_DN[g] - r_dn[t, g], mu_max_dn[t, g], bin_mu_max_dn[t, g],
                                      M1_mu_max_dn, M2_mu_max_dn)
            linearize_complementarity(prob, P_MAX[g] - p[t, g] - r_up[t, g], mu_up[t, g], bin_mu_up[t, g], M1_mu_up,
                                      M2_mu_up)
            linearize_complementarity(prob, p[t, g] - r_dn[t, g] - P_MIN[g], mu_dn[t, g], bin_mu_dn[t, g], M1_mu_dn,
                                      M2_mu_dn)
        for s in range(num_storage):
            linearize_complementarity(prob, p_ch[t, s], mu_min_ch[t, s], bin_mu_min_ch[t, s], M1_mu_min_ch,
                                      M2_mu_min_ch)
            linearize_complementarity(prob, p_hat_ch[t, s] - p_ch[t, s], mu_max_ch[t, s], bin_mu_max_ch[t, s],
                                      M1_mu_max_ch, M2_mu_max_ch)
            linearize_complementarity(prob, p_dis[t, s], mu_min_dis[t, s], bin_mu_min_dis[t, s], M1_mu_min_dis,
                                      M2_mu_min_dis)
            linearize_complementarity(prob, p_hat_dis[t, s] - p_dis[t, s], mu_max_dis[t, s], bin_mu_max_dis[t, s],
                                      M1_mu_max_dis, M2_mu_max_dis)

            linearize_complementarity(prob, r_ch_up[t, s], mu_min_ch_up[t, s], bin_mu_min_ch_up[t, s], M1_mu_min_ch_up,
                                      M2_mu_min_ch_up)
            linearize_complementarity(prob, r_hat_ch_up[t, s] - r_ch_up[t, s], mu_max_ch_up[t, s],bin_mu_max_ch_up[t, s]
                                      ,M1_mu_max_ch_up, M2_mu_max_ch_up)
            linearize_complementarity(prob, r_ch_dn[t, s], mu_min_ch_dn[t, s], bin_mu_min_ch_dn[t, s], M1_mu_min_ch_dn,
                                      M2_mu_min_ch_dn)
            linearize_complementarity(prob, r_hat_ch_dn[t, s] - r_ch_dn[t, s], mu_max_ch_dn[t, s],
                                      bin_mu_max_ch_dn[t, s], M1_mu_max_ch_dn, M2_mu_max_ch_dn)

            linearize_complementarity(prob, r_dis_up[t, s], mu_min_dis_up[t, s], bin_mu_min_dis_up[t, s],
                                      M1_mu_min_dis_up, M2_mu_min_dis_up)
            linearize_complementarity(prob, r_hat_dis_up[t, s] - r_dis_up[t, s], mu_max_dis_up[t, s],
                                      bin_mu_max_dis_up[t, s], M1_mu_max_dis_up, M2_mu_max_dis_up)
            linearize_complementarity(prob, r_dis_dn[t, s], mu_min_dis_dn[t, s], bin_mu_min_dis_dn[t, s],
                                      M1_mu_min_dis_dn, M2_mu_min_dis_dn)
            linearize_complementarity(prob, r_hat_dis_dn[t, s] - r_dis_dn[t, s], mu_max_dis_dn[t, s],
                                      bin_mu_max_dis_dn[t, s], M1_mu_max_dis_dn, M2_mu_max_dis_dn)

            linearize_complementarity(prob, p_ch[t, s] - r_ch_up[t, s], mu_ch_up[t, s], bin_mu_ch_up[t, s], M1_mu_ch_up,
                                      M2_mu_ch_up)
            linearize_complementarity(prob, p_dis[t, s] - r_dis_dn[t, s], mu_dis_dn[t, s], bin_mu_dis_dn[t, s],
                                      M1_mu_dis_dn, M2_mu_dis_dn)
        for j in range(num_WT):
            linearize_complementarity(prob, w_cur[t, j], mu_min_w_cur[t, j], bin_mu_min_w_cur[t, j], M1_mu_min_w_cur,
                                      M2_mu_min_w_cur)
            linearize_complementarity(prob, W_FORE[t, j] - w_cur[t, j], mu_max_w_cur[t, j], bin_mu_max_w_cur[t, j],
                                      M1_mu_max_w_cur, M2_mu_max_w_cur)


        if method == 'proposed':
            linearize_complementarity(prob, Q_WM_UP[t] - (u - r_wm_up[t]), mu_4[t], bin_mu_4[t], M1_mu_4, M2_mu_4)
            linearize_complementarity(prob, Q_WM_DN[t] - (u - r_wm_dn[t]), mu_5[t], bin_mu_5[t], M1_mu_5, M2_mu_5)
            N_P_UP = np.where(random_var_scenarios[:, t] < Q_WM_UP[t])[0]
            N_P_DN = np.where(-random_var_scenarios[:, t] < Q_WM_DN[t])[0]
            for i in N_P_UP:
                linearize_complementarity(prob, kappa[i] * random_var_scenarios[i, t]  - (
                            u - v[i] - kappa[i] * r_wm_up[t]), mu_2[t, i], bin_mu_2[t, i], M1_mu_2, M2_mu_2)
            for i in N_P_DN:
                linearize_complementarity(prob, -kappa[i] * random_var_scenarios[i, t]  - (
                        u - v[i] - kappa[i] * r_wm_dn[t]), mu_3[t, i], bin_mu_3[t, i], M1_mu_3, M2_mu_3)
        elif method == 'linearforN':
            for i in range(N):
                linearize_complementarity(prob, kappa[i] * random_var_scenarios[i, t]  - (
                            u - v[i] - kappa[i] * r_wm_up[t]), mu_2[t, i], bin_mu_2[t, i], M1_mu_2, M2_mu_2)
                linearize_complementarity(prob, -kappa[i] * random_var_scenarios[i, t] - (
                        u - v[i] - kappa[i] * r_wm_dn[t]), mu_3[t, i], bin_mu_3[t, i], M1_mu_3, M2_mu_3)
        elif method == 'wcvar':
            for i in range(N):
                linearize_complementarity(prob, w_cvar_up[t] * random_var_scenarios[i, t]  +
                            tau + alpha[i] + w_cvar_up[t] * r_wm_up[t], mu_2[t, i], bin_mu_2[t, i], M1_mu_2, M2_mu_2)
                linearize_complementarity(prob, - w_cvar_dn[t] * random_var_scenarios[i, t] +
                                          tau + alpha[i] + w_cvar_dn[t] * r_wm_dn[t], mu_3[t, i], bin_mu_3[t, i],
                                          M1_mu_3, M2_mu_3)
            linearize_complementarity(prob, beta - w_cvar_up[t], mu_4[t], bin_mu_4[t], M1_mu_4, M2_mu_4)
            linearize_complementarity(prob, beta - w_cvar_dn[t], mu_5[t], bin_mu_5[t], M1_mu_5, M2_mu_5)
        elif method == 'bonferroni':
            linearize_complementarity(prob, r_wm_up[t] - eta_up[t], mu_1[t], bin_mu_1[t], M1_mu_1, M2_mu_1)
            linearize_complementarity(prob, r_wm_dn[t] - eta_dn[t], mu_2[t], bin_mu_2[t], M1_mu_2, M2_mu_2)

    # add bound for dual variables (prices) to mitigate the numerical issue warning
    max_price = max(c.values()) * 1.5
    prob.addConstr(lambda_en <= max_price)
    prob.addConstr(lambda_up <= max_price)
    prob.addConstr(lambda_dn <= max_price)
    # add bound for storage bids
    prob.addConstr(b_hat_ch <= max_price)
    prob.addConstr(b_hat_dis <= max_price)
    prob.addConstr(b_hat_ch_up <= max_price)
    prob.addConstr(b_hat_ch_dn <= max_price)
    prob.addConstr(b_hat_dis_up <= max_price)
    prob.addConstr(b_hat_dis_dn <= max_price)

    prob.setObjective(gp.quicksum(- (lambda_en[t] + c_ch[s]) * p_ch[t, s] + (lambda_en[t] - c_dis[s]) * p_dis[t, s]
                                  + lambda_up[t] * (r_ch_up[t, s] + r_dis_up[t, s])
                                  + lambda_dn[t] * (r_ch_dn[t, s] + r_dis_dn[t, s])
                                  for t in range(T) for s in range(num_storage)), gp.GRB.MAXIMIZE)

    prob.setParam('MIPGap', 1e-3)
    prob.setParam('IntFeasTol', 1e-9)
    prob.setParam('FeasibilityTol', 1e-9)
    prob.setParam('OptimalityTol', 1e-9)
    prob.setParam('Threads', thread)
    # # fix seed
    prob.setParam('Seed', gurobi_seed)

    # prob.setParam('Seed', 10)
    if numerical_focus:
        prob.setParam('NumericFocus', 3)
    prob.setParam('IntegralityFocus', IntegralityFocus)
    # set time limit
    prob.setParam('TimeLimit', TimeLimit)
    if log_file_name is not None:
        prob.setParam('LogFile', log_file_name)

    prob.optimize()

    # concat all variables to be returned
    results = [prob, lambda_en, lambda_up, lambda_dn, b_hat_ch, p_hat_ch, b_hat_dis, p_hat_dis,
               b_hat_ch_up, r_ch_up, b_hat_ch_dn, r_ch_dn, b_hat_dis_up, r_dis_up, b_hat_dis_dn, r_dis_dn,
               p_ch, p_dis, r_hat_dis_up, r_hat_dis_dn, r_hat_ch_up, r_hat_ch_dn, e,
               r_wm_up, r_wm_dn, r_up, r_dn, w_cur, p, k, Q_WM_UP, Q_WM_DN, random_var_scenarios]
    return results

def main():
    gurobi_seed = 130000
    seed = gurobi_seed
    rng = np.random.RandomState(seed)
    numerical_focus = False
    IntegralityFocus = 0
    thread = 4
    TimeLimit = 3600

    num_gen = 5  # the number of thermal generators
    num_WT = 1  # the number of wind turbines
    num_storage = 1  # the number of storages
    N_samples = 1000  # the number of wind power scenarios used for training

    T = 16  # the number of time steps
    N = 50  # the number of scenarios for the WDRJCC
    theta = 0.01  # 0.01 the Wasserstein radius. Bonferroni approximation requires small theta for feasibility 0.1
    epsilon = 0.05  # the risk level
    method = 'proposed'  # linearforN, wcvar, proposed, bonferroni

    Tstart = rng.randint(0, 24 - T + 1)

    M = 1e5

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

    WT_total = 0.6 * 8
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
                      gurobi_seed=gurobi_seed, log_file_name=None, numerical_focus=numerical_focus, IntegralityFocus=IntegralityFocus, thread=thread, TimeLimit=TimeLimit)

    (prob, lambda_en, lambda_up, lambda_dn, b_hat_ch, p_hat_ch, b_hat_dis, p_hat_dis,
               b_hat_ch_up, r_ch_up, b_hat_ch_dn, r_ch_dn, b_hat_dis_up, r_dis_up, b_hat_dis_dn, r_dis_dn,
               p_ch, p_dis, r_hat_dis_up, r_hat_dis_dn, r_hat_ch_up, r_hat_ch_dn, e,
               r_wm_up, r_wm_dn, r_up, r_dn, w_cur, p, k, Q_WM_UP, Q_WM_DN, random_var_scenarios) = solve_bilevel(**param_dict)

    # all the following prints out variables for analysing the correctness of the whole set-up
    if (prob.status == gp.GRB.OPTIMAL or prob.status == gp.GRB.TIME_LIMIT) and prob.SolCount>0:
        print('====================objective====================')
        print(f'objective = {prob.objVal}')

        total_value = 0

        for t in range(T):
            for s in range(num_storage):
                # Extract the values of the decision variables
                b_hat_ch_value = b_hat_ch[t, s].X
                p_ch_value = p_ch[t, s].X
                b_hat_dis_value = b_hat_dis[t, s].X
                p_dis_value = p_dis[t, s].X
                b_hat_ch_up_value = b_hat_ch_up[t, s].X
                r_ch_up_value = r_ch_up[t, s].X
                b_hat_ch_dn_value = b_hat_ch_dn[t, s].X
                r_ch_dn_value = r_ch_dn[t, s].X
                b_hat_dis_up_value = b_hat_dis_up[t, s].X
                r_dis_up_value = r_dis_up[t, s].X
                b_hat_dis_dn_value = b_hat_dis_dn[t, s].X
                r_dis_dn_value = r_dis_dn[t, s].X

                # Calculate each term using the extracted values
                term_storage = (b_hat_ch_value * p_ch_value - b_hat_dis_value * p_dis_value
                                - (b_hat_ch_up_value * r_ch_up_value + b_hat_ch_dn_value * r_ch_dn_value
                                   + b_hat_dis_up_value * r_dis_up_value + b_hat_dis_dn_value * r_dis_dn_value))
                # Accumulate the results into the total value
                total_value += term_storage

        # Summing the generator and wind turbine related costs
        total_gen = sum(
            c[t, g] * p[t, g].X + c_rs[t, g] * (r_up[t, g].X + r_dn[t, g].X) for t in range(T) for g in range(num_gen)
        )
        total_wt = sum(c_cur[t, j] * w_cur[t, j].X for t in range(T) for j in range(num_WT))

        # Include these in the total value
        total_value -= total_gen + total_wt

        print('Lower-level Market Objective:', total_value)

        # Calculate and print the total profit
        total_profit = 0.0
        for t in range(T):
            for s in range(num_storage):
                total_profit += (- (lambda_en[t].X + c_ch[s]) * p_ch[t, s].X
                                 + (lambda_en[t].X - c_dis[s]) * p_dis[t, s].X
                                 + lambda_up[t].X * (r_ch_up[t, s].X + r_dis_up[t, s].X)
                                 + lambda_dn[t].X * (r_ch_dn[t, s].X + r_dis_dn[t, s].X))

        print(f"Calculated total value: {total_profit}")

        print('--------------energy price-------------')
        print(f'lambda_en = {lambda_en.X}')

        print('--------------reserve price-------------')
        print(f'lambda_up = {lambda_up.X}')
        print(f'lambda_dn = {lambda_dn.X}')

        print('--------------generator power and reverse-------------')
        for t in range(T):
            for g in range(num_gen):
                print(f"p[{t}, {g}] = {p[t, g].X}, r_up[{t}, {g}] = {r_up[t, g].X}, r_dn[{t}, {g}] = {r_dn[t, g].X}")

        for g in range(num_gen):
            print(f"--------- Generator {g + 1} ---------")

            # Print all p values for this generator
            print("\n--- p values ---")
            for t in range(T):
                print(f"{p[t, g].X}")

            # Print all r_up values for this generator
            print("\n--- r_up values ---")
            for t in range(T):
                print(f"{r_up[t, g].X}")

            # Print all r_dn values for this generator
            print("\n--- r_dn values ---")
            for t in range(T):
                print(f"{r_dn[t, g].X}")

            print("-----------------------------------\n")

        print('--------------wind reserve due to mismatch-------------')
        # print(f'r_wm_up = {r_wm_up.X}')
        # print(f'r_wm_dn = {r_wm_dn.X}')
        print('-------------- Wind Reserve Due to Mismatch -------------')

        print("\n--- Wind Mismatch Upward Reserve (r_wm_up) ---")
        for t in range(T):
            print(f"{r_wm_up.X[t]:.2f}")

        print("\n--- Extra Upward Reserve (R_UP_EX) ---")
        for t in range(T):
            print(f"{R_UP_EX[t]:.2f}")

        print("\n--- Wind Mismatch Downward Reserve (r_wm_dn) ---")
        for t in range(T):
            print(f"{r_wm_dn.X[t]:.2f}")

        print("\n--- Extra Downward Reserve (R_DN_EX) ---")
        for t in range(T):
            print(f"{R_DN_EX[t]:.2f}")

        print('--------------reserve-------------')
        for t in range(T):
            secured_r_up = (r_ch_up[t, :].X.sum() + r_dis_up[t, :].X.sum() + r_up[t, :].X.sum())
            print(f"r_total_up[{t}] = {secured_r_up}, r_wm_up[{t}] = {r_wm_up[t].X}, R_UP_EX[{t}] = {R_UP_EX[t]}")

            secured_r_dn = (r_ch_dn[t, :].X.sum() + r_dis_dn[t, :].X.sum() + r_dn[t, :].X.sum())
            print(f"r_total_dn[{t}] = {secured_r_dn}, r_wm_dn[{t}] = {r_wm_dn[t].X}, R_DN_EX[{t}] = {R_DN_EX[t]}")

        print('--------------storage power-------------')
        print("\nBids for charging:")
        # print(f'b_hat_ch = {b_hat_ch.X}')
        for t in range(T):
            # print(f"b_hat_ch[{t}] = {b_hat_ch[t, 0].X}")
            print(b_hat_ch[t, 0].X,p_hat_ch[t, 0].X)



        print("\nBids for discharging:")
        # print(f'b_hat_dis = {b_hat_dis.X}')
        for t in range(T):
            # print(f"b_hat_dis[{t}] = {b_hat_dis[t, 0].X}")
            print(b_hat_dis[t, 0].X,p_hat_dis[t, 0].X)



        print("\nenergy stored in storage:")
        for t in range(T):
            print(e[t, 0].X)

        print('--------------storage reserve-------------')

        print("\nBids for upward charge reserve:")
        # print(f'b_hat_ch_up = {b_hat_ch_up.X}')
        for t in range(T):
            # print(f"b_hat_ch_up[{t}] = {b_hat_ch_up[t, 0].X}")
            print(b_hat_ch_up[t, 0].X)

        print("\nBids for downward charge reserve:")
        # print(f'b_hat_ch_dn = {b_hat_ch_dn.X}')
        for t in range(T):
            # print(f"b_hat_ch_dn[{t}] = {b_hat_ch_dn[t, 0].X}")
            print(b_hat_ch_dn[t, 0].X)


        print("\nBids for upward discharge reserve:")
        # print(f'b_hat_dis_up = {b_hat_dis_up.X}')
        for t in range(T):
            # print(f"b_hat_dis_up[{t}] = {b_hat_dis_up[t, 0].X}")
            print(b_hat_dis_up[t, 0].X)



        print("\nBids for downward discharge reserve:")
        # print(f'b_hat_dis_dn = {b_hat_dis_dn.X}')
        for t in range(T):
            # print(f"b_hat_dis_dn[{t}] = {b_hat_dis_dn[t, 0].X}")
            print(b_hat_dis_dn[t, 0].X)



        print('==================== Battery Status ====================')
        for t in range(T):
            print(f"--- Time Step {t} ---")
            for s in range(num_storage):
                print(f"  Storage Unit {s}:")
                print(f"    Energy Level (e[{t}, {s}]): {e[t, s].X:.2f}")
                print(f"    Charging Power (p_ch[{t}, {s}]): {p_ch[t, s].X:.2f}")
                print(f"    Discharging Power (p_dis[{t}, {s}]): {p_dis[t, s].X:.2f}")
                # print(f"    Energy quantity bid for Charging Power (p_hat_ch[{t}, {s}]): {p_hat_ch[t, s].X:.2f}")
                print(f"    Upward Charging Reserve (r_ch_up[{t}, {s}]): {r_ch_up[t, s].X:.2f}")
                # print(f"    Energy quantity bid Downward Charging Reserve (r_ch_dn[{t}, {s}]): {r_ch_dn[t, s].X:.2f}")
                print(f"    Downward Charging Reserve (r_ch_dn[{t}, {s}]): {r_ch_dn[t, s].X:.2f}")
                # print(f"    Energy quantity bid for Discharging Power (p_hat_dis[{t}, {s}]): {p_hat_dis[t, s].X:.2f}")
                print(f"    Upward Discharging Reserve (r_dis_up[{t}, {s}]): {r_dis_up[t, s].X:.2f}")
                # print(f"    Energy quantity bid for Upward Discharging Reserve (r_hat_dis_up[{t}, {s}]): {r_hat_dis_up[t, s].X:.2f}")
                print(f"    Downward Discharging Reserve (r_dis_dn[{t}, {s}]): {r_dis_dn[t, s].X:.2f}")
                # print(f"    Energy quantity bid for Downward Discharging Reserve (r_hat_dis_dn[{t}, {s}]): {r_hat_dis_dn[t, s].X:.2f}")
                print('    -----')
            print('-----------------------------------------')

        print('--------------continuous storage reserve-------------')

        for t in range(T):
            print(p_ch[t, 0].X)
        for t in range(T):
            print(p_dis[t, 0].X)
        print("\nCleared Quantity for upward charge reserve:")
        for t in range(T):
            print(r_ch_up[t, 0].X)

        print("\nCleared Quantity for downward charge reserve:")
        for t in range(T):
            print(r_ch_dn[t, 0].X)

        print("\nCleared Quantity for upward discharge reserve:")
        for t in range(T):
            print(r_dis_up[t, 0].X)

        print("\nCleared Quantity for downward discharge reserve:")
        for t in range(T):
            print(r_dis_dn[t, 0].X)
        print('--------------Compare-------------')
        print("\nCleared Quantity for charge:")
        for t in range(T):
            print(p_ch[t, 0].X)
        print("\nCleared Quantity for discharge:")
        for t in range(T):
            print(p_dis[t, 0].X)
        print("\nCleared Quantity for upward charge reserve:")
        for t in range(T):
            print(r_ch_up[t, 0].X)
        print("\nCleared Quantity for downward charge reserve:")
        for t in range(T):
            print(r_ch_dn[t, 0].X)
        print("\nCleared Quantity for upward discharge reserve:")
        for t in range(T):
            print(r_dis_up[t, 0].X)
        print("\nCleared Quantity for downward discharge reserve:")
        for t in range(T):
            print(r_dis_dn[t, 0].X)

    else:
        print("No feasible solution found.")

    #  calculate the out-of-sample JCC satisfaction rate
    # Assuming r_wm_up and r_wm_dn are Gurobi variables, get their values after optimization
    r_wm_up_values = np.array([var.X for var in r_wm_up])
    r_wm_dn_values = np.array([var.X for var in r_wm_dn])

    random_var_scenarios_test = WT_error_scenarios_test.reshape(WT_error_scenarios_test.shape[0], -1)

    constraints_up_satisfied = r_wm_up_values >= -random_var_scenarios_test
    constraints_dn_satisfied = r_wm_dn_values >= random_var_scenarios_test

    # Combine the two arrays into one, with each constraint's results following the other
    combined_constraints = np.concatenate([constraints_up_satisfied, constraints_dn_satisfied], axis=1)

    # Calculate the reliability percentage for each constraint set individually and for the combined set
    reliability_combined = np.mean(np.all(combined_constraints, axis=1)) * 100

    # print(f"Overall reliability across all constraints (combined): {reliability_combined:.2f}%")

    # print('------------------------------------')
    print('Optimal value:', prob.objVal)
    print(f'The out-of-sample JCC satisfaction rate is {reliability_combined}%')

    ##################################################################################################
    ############################# New Problem For Solving Market Clearing#############################
    ##################################################################################################
    if (prob.status == gp.GRB.OPTIMAL or prob.status == gp.GRB.TIME_LIMIT) and prob.SolCount>0:
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
        # post-processing the bids and offers
        for t in range(T):
            for s in range(num_storage):
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

    else:
        print("Optimal solution was not found.")



    energy_prices, reserve_up_prices, reserve_down_prices, reliability_combined_exact = calculate_prices(LOAD, R_UP_EX, R_DN_EX, T, num_storage,
                                                                             num_gen, num_WT, N, epsilon, theta, k, M,
                                                                             random_var_scenarios,
                                                                             Q_WM_UP, Q_WM_DN, P_MIN, P_MAX, R_MAX_UP,
                                                                             R_MAX_DN, W_FORE, c, c_rs, c_cur,
                                                                             b_hat_ch_values, b_hat_dis_values,
                                                                             b_hat_dis_up_values, b_hat_ch_up_values,
                                                                             b_hat_dis_dn_values, b_hat_ch_dn_values,
                                                                             p_hat_ch_values, p_hat_dis_values,
                                                                             r_hat_dis_up_values, r_hat_ch_up_values,
                                                                             r_hat_dis_dn_values, r_hat_ch_dn_values,WT_error_scenarios_test
                                                                             )
    print(f'The EXACT out-of-sample JCC satisfaction rate is {reliability_combined_exact}%')

    print("\n")
    print('======================================================')
    print('================Sensitivity Analysis==================')
    print('======================================================')
    print("Energy Prices:", energy_prices)
    print("Reserve Up Prices:", reserve_up_prices)
    print("Reserve Down Prices:", reserve_down_prices)
    print("\n")
    print('======================================================')
    print('================Fixing Binary Variales================')
    print('======================================================')
    p_ch_values, p_dis_values, r_ch_up_values, r_dis_up_values, r_ch_dn_values, r_dis_dn_values = calculate_prices_with_fixed_binary(
        LOAD, R_UP_EX, R_DN_EX, T, num_storage, num_gen, num_WT, N, epsilon, theta, k, M,
        random_var_scenarios, Q_WM_UP, Q_WM_DN, P_MIN, P_MAX, R_MAX_UP, R_MAX_DN, W_FORE,
        c,
        c_rs, c_cur, b_hat_ch_values, b_hat_dis_values, b_hat_dis_up_values,
        b_hat_ch_up_values, b_hat_dis_dn_values, b_hat_ch_dn_values, p_hat_ch_values,
        p_hat_dis_values, r_hat_dis_up_values, r_hat_ch_up_values, r_hat_dis_dn_values,
        r_hat_ch_dn_values, WT_error_scenarios_test, c_ch, c_dis)

    total_profit= 0.0
    for t in range(T):
        for s in range(num_storage):
            # Compute the total profit based on the dual variables and these values
            total_profit += (- (energy_prices[t] + c_ch[s]) * p_ch_values[t,s]
                                   + (energy_prices[t] - c_dis[s]) * p_dis_values[t,s]
                                   + reserve_up_prices[t] * (r_ch_up_values[t,s] + r_dis_up_values[t,s])
                                   + reserve_down_prices[t] * (r_ch_dn_values[t,s] + r_dis_dn_values[t,s]))

    print(f"Profit from MIP Model (quantity) and Sensitivity Analysis (price): {total_profit}")

    print('--------------storage power-------------')

    print("\nCleared Quantity for charge:")
    for t in range(T):
        print(p_ch_values[t, 0])

    print("\nCleared Quantity for discharge:")
    for t in range(T):
        print(p_dis_values[t, 0])

    print("\nCleared Quantity for upward charge reserve:")
    for t in range(T):
        print(r_ch_up_values[t, 0])


    print("\nCleared Quantity for downward charge reserve:")
    for t in range(T):
        print(r_ch_dn_values[t, 0])


    print("\nCleared Quantity for upward discharge reserve:")
    for t in range(T):
        print(r_dis_up_values[t, 0])


    print("\nCleared Quantity for downward discharge reserve:")
    for t in range(T):
        print(r_dis_dn_values[t, 0])

if __name__ == '__main__':
    main()