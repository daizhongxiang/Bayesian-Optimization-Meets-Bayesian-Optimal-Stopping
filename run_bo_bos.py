import numpy as np

### import the BO package
from bayesian_optimization import BayesianOptimization

### import the objetive function
from objective_functions import objective_function_LR_MNIST as objective_function
# from objective_functions import objective_function_CNN_CIFAR_10 as objective_function
# from objective_functions import objective_function_CNN_SVHN as objective_function

np.random.seed(0)

iterations_list = np.arange(1, 11)

for run_iter in iterations_list:
    '''
    The input arguments to "BayesianOptimization" are explained in the script "bayesian_optimization.py";
    In particular, set "no_BOS=True" if we want to run standard GP-UCB, and "no_BOS=False" if we want to run the BO-BOS algorithm;
    When running the "maximize" function, the intermediate results are saved after every BO iteration, under the file name log_file; the content of the log file is explained in the "analyze_results" ipython notebook script.
    '''
    
#     run without BOS
    BO_no_BOS = BayesianOptimization(f=objective_function,
            dim = 3, gp_opt_schedule=10, \
            no_BOS=True, use_init=None, \
            log_file="saved_results/bos_mnist_no_stop_" + str(run_iter) + ".p", save_init=True, \
            save_init_file="mnist_5_" + str(run_iter) + ".p", \
            parameter_names=["batch_size", "C", "learning_rate"])
    # "parameter_names" are dummy variables whose correspondance in the display is not guaranteed
    BO_no_BOS.maximize(n_iter=50, init_points=3, kappa=2, use_fixed_kappa=False, kappa_scale=0.2, acq='ucb')

#     run with BOS, using the same initializations as above
    BO_BOS = BayesianOptimization(f=objective_function,
            dim = 3, gp_opt_schedule=10, no_BOS=False, use_init="mnist_5_" + str(run_iter) + ".p", \
            log_file="saved_results/bos_mnist_with_stop_" + str(run_iter) + ".p", save_init=False, \
            save_init_file=None, \
            add_interm_fid=[0, 9, 19, 29, 39], parameter_names=["batch_size", "C", "learning_rate"])
    BO_BOS.maximize(n_iter=70, init_points=3, kappa=2, use_fixed_kappa=False, kappa_scale=0.2, acq='ucb')

