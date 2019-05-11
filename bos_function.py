'''
This script implements the Bayesian Optimal Stopping (BOS) function, which outputs the optimal decision rules
'''

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import GPy
from tqdm import tqdm

def run_BOS(init_curve, incumbent, training_epochs, bo_iteration):
    '''
    init_curve: initial learning curves used to generate forward simulations
    incumbent: currently optimal validation error
    training_epochs: the maximum number of epochs to train (N)
    bo_itetation: iteration of BO, starting from 0 (after initialization)
    '''
    
    grid_size = 100
    sample_number = 200
    fs_sample_number = 100000

    ###### define the cost parameters, including K_2, c, and the K_1 sequence
#     K1_init, K2, C, gamma = 100, 99, 1, 1.0 # fixed K1
    K1_init, K2, C, gamma = 100, 99, 1, 0.99 # small K1
#     K1_init, K2, C, gamma = 100, 99, 1, 0.95 # normal K1
#     K1_init, K2, C, gamma = 100, 99, 1, 0.89 # large K1 89

    K1 = K1_init / (gamma ** bo_iteration)

    T = training_epochs - len(init_curve)
    lc_curr_opt = 1 - incumbent

    ######## Below generates forward simulation samples #########
    # Use the initial samples from a learning curve to initialize the prior distribution
    initial_sample_len = len(init_curve)
    lc_x = np.arange(1, initial_sample_len+1).reshape(-1, 1)
    init_curve = np.array(init_curve).reshape(-1, 1)
    alpha_init, beta_init = initial_sample_len, np.sum(init_curve)

    k_exp = GPy.kern.src.ExpKernel(input_dim=1, active_dims=[0])
    m_gpy = GPy.models.GPRegression(lc_x, init_curve, k_exp)
    m_gpy.likelihood.variance.fix(1e-3) # fix the noise, to produce more diverse and realistic forward simulation samples
    m_gpy.optimize(messages=False)

    xx = np.arange(1, initial_sample_len+T+1).reshape(-1, 1)
    post_samples = m_gpy.posterior_samples_f(xx, full_cov=True, size=fs_sample_number)    
    post_samples = np.squeeze(post_samples)
    samples_data = post_samples.T[:, initial_sample_len:]
    print("samples_data: ", samples_data.shape)

    # remove those sampled curves that exceed the range [0, 1]
    ind = np.all(samples_data<1, axis=1)
    samples_data = samples_data[ind]
    ind = np.all(samples_data>0, axis=1)
    samples_data = samples_data[ind]

    ######### Below we run backward induction to get Bayes optimal decision ##########
    # calculate St from sample trajectories
    St = []
    for s in tqdm(samples_data):
        St.append(np.cumsum(s) / (np.arange(len(s)) + 1))
    St = np.array(St)

    grid_St = np.linspace(0, 1, grid_size)

    state = []
    for i in range(len(grid_St) - 1):
        state.append((grid_St[i] + grid_St[i + 1]) / 2)
    state = np.array(state)

    losses = np.zeros((T, grid_size - 1, 3))
    print("Calculating termination losses...")

    all_Pr_z_star = np.zeros((T, grid_size - 1))
    for step in tqdm((np.arange(T) + 1)):
        data_t = St[:, step - 1]
        Pr_z_star_samples = np.zeros(grid_size - 1)
        Pr_z_star_accum = np.zeros(grid_size - 1)

        for i in range(len(data_t)):
            error_last_step = samples_data[i, -1]

            val = data_t[i]
            ind_left = np.max(np.nonzero(val > grid_St)[0])
            if error_last_step > lc_curr_opt:
                Pr_z_star_accum[ind_left] += 1.0
            Pr_z_star_samples[ind_left] += 1

        # for each grid/cell, calculate the average over all next-step continuation losses
        for i in range(grid_size - 1):
            if Pr_z_star_samples[i] != 0:
                losses[step - 1, i, 2] 
                Pr_z_star = Pr_z_star_accum[i] / Pr_z_star_samples[i]
                all_Pr_z_star[step - 1, i] = Pr_z_star

                loss_d = K2 * Pr_z_star + C * step
                loss_d_star = K1 * (1 - Pr_z_star) + C * step
                losses[step - 1, i, 0] = loss_d
                losses[step - 1, i, 1] = loss_d_star    
            else:
                # this says that if the current cell is not visited by any forward simulation samples, we should always choose to continue, which is a convervative approach
                all_Pr_z_star[step - 1, i] = 100
                losses[step - 1, i, 0] = 1e4
                losses[step - 1, i, 1] = 1e4

    print("Calculating continuation losses...")
    step = T - 1
    while step > 0:
        print("Running step {0}".format(step))

        data_t = St[:, step - 1]
        grid_samples = np.zeros(grid_size - 1)
        grid_losses_cont = np.zeros(grid_size - 1)

        # go through all forward simulation samples; for each sample, find the index it ends up with in the next step, and find the corresponding optimal loss
        for i in range(len(data_t)):
            val = data_t[i]

            St_next = St[i, step] # the statistic at the next step
            St_next_ind_left = np.max(np.nonzero(St_next > grid_St)[0])

            if step == T - 1:
                loss_continue = np.min(losses[step + 1 - 1, St_next_ind_left, :2])
            else:
                loss_continue = np.min(losses[step + 1 - 1, St_next_ind_left, :])

            ind_left = np.max(np.nonzero(val > grid_St)[0])
            grid_losses_cont[ind_left] += loss_continue
            grid_samples[ind_left] += 1

        # for each grid/cell, calculate the average over all next-step continuation losses
        for i in range(len(grid_samples)):
            if grid_samples[i] > 30:
                losses[step - 1, i, 2] = grid_losses_cont[i] / grid_samples[i]
            else:
                # this says that if the current cell is visited by no more than 30 forward simulation samples, we should always choose to continue, which is taking a convervative approach
                losses[step - 1, i, 2] = 0

        step = step - 1
    
    # Below we extract the Bayes optimal decisions according to the losses calculated above
    # 0: decision d_0
    # 1: decision d_2
    # 2: decision d_1
    actions = np.zeros((T, len(state)))
    for s_ind in range(len(state)):
        actions[-1, s_ind] = np.argmin(losses[-1, s_ind, :2]) + 1
    for step in range(T-1):
        for s_ind in range(len(state)):
            if losses[step, s_ind, 2] != 0:
                actions[step, s_ind] = np.argmin(losses[step, s_ind, :]) + 1
            else:
                actions[step, s_ind] = 0
    print("Done")

    # return the obtained decision rules, as well as the space of summary statistics
    return actions, grid_St
