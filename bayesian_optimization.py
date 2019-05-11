# -*- coding: utf-8 -*-

import numpy as np
import GPy
from helper_funcs import UtilityFunction, unique_rows, PrintLog, acq_max
import pickle
from scipy.stats import expon, norm, gamma, truncexpon
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

init_path = "saved_init/"

class BayesianOptimization(object):
    def __init__(self, f, gp_opt_schedule, no_BOS=True, \
                 use_init=False, log_file=None, save_init=False, save_init_file=None, \
                 max_running_hours=6, add_interm_fid=[], N=50, N_init_epochs=8, dim=3, parameter_names=[], verbose=1):
        """
        gp_opt_schedule: we update the GP hyper-parameters via maximum likelihood every "gp_opt_schedule" iterations
        no_BOS: whether we run Bayesian Optimal Stopping
        use_init: whether to use existing initializations
        log_file: the log file in which the results are saved
        save_init: Boolean; whether to save the initialization
        save_init_file: the file name under which to save the initializations; only used if save_init==True
        max_running_hours: the maximum number of hours to run the script
        add_interm_fid: the intermedium results to be used in updating the GP surrogate function
        N: the maximum number of epochs
        N_init_epochs: the initial number of epochs to use in the BOS algorithm
        dim: the number of hyper-parameters
        verbose: verbosity        
        """
        
        self.use_init = use_init
        self.max_running_hours = max_running_hours
        
        self.add_interm_fid = add_interm_fid
        
        self.N = N
        self.N_init_epochs = N_init_epochs
    
        self.log_file = log_file
        
        self.no_BOS = no_BOS
        
        self.curr_max_X = None
        self.curr_opt_lc = None
        self.incumbent = None
        
        self.action_regions = None
        self.grid_St = None

        self.keys = parameter_names

        # Find number of parameters
        self.dim = dim

        # assume all hyper-parameters are bounded in the range [0.0, 1.0]
        bounds = np.zeros((self.dim, 2))
        bounds[:, 1] = 1.0
        self.bounds = bounds

        # The function to be optimized
        self.f = f

        # Initialization flag
        self.initialized = False

        # Initialization lists --- stores starting points before process begins
        self.init_points = []
        self.x_init = []
        self.y_init = []

        # Numpy array place holders
        self.X = None
        self.Y = None
        
        self.time_started = 0

        # Counter of iterations
        self.i = 0
        
        # Utility Function placeholder
        self.util = None

        # PrintLog object
        self.plog = PrintLog(self.keys)
        
        self.save_init = save_init
        self.save_init_file = save_init_file
        
        self.gp_opt_schedule = gp_opt_schedule

        self.res = {}
        self.res['all'] = {'params':[], 'init':[], 'eval_times':[], 'BOS_times':[], 'epoch_values':[], 'time_started':0}

        self.verbose = verbose

    def init(self, init_points):
        """
        randomly sample "init_points" points as the initializations
        """
        
        # Generate random points
        self.time_started = time.time()
        self.res['all']['time_started'] = self.time_started

        l = [np.random.uniform(x[0], x[1], size=init_points)
                 for x in self.bounds]

        self.init_points = l
        
        # Create empty list to store the new values of the function
        y_init = []

        fid_inits = []
        for x in self.init_points:
            curr_y, fid, BOS_time, epoch_values, time_func_eval = self.f(x, no_stop=True, bo_iteration=0, \
                    N=self.N, N_init_epochs=self.N_init_epochs)
            self.res['all']['eval_times'].append(time_func_eval)
            self.res['all']['BOS_times'].append(BOS_time)
            self.res['all']['epoch_values'].append(epoch_values)

            y_init.append(curr_y) # we need to negate the error because the program assumes maximization
            fid_inits.append(1.0)
            
            if self.verbose:
                self.plog.print_step(x, y_init[-1])

        self.X = np.concatenate((np.asarray(self.init_points), np.array(fid_inits).reshape(-1, 1)), axis=1)
        self.Y = np.asarray(y_init)

        self.incumbent = np.max(y_init)
        self.initialized = True

        init = {"X":self.X, "Y":self.Y, "init_epoch_values":self.res['all']['epoch_values'],\
               "init_eval_times":self.res['all']['eval_times'], \
                "init_time_started":self.res['all']['time_started']}
        self.res['all']['init'] = init

        if self.save_init:
            pickle.dump(init, open(init_path + self.save_init_file, "wb"))


    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 use_fixed_kappa=True,
                 kappa_scale=0.2,
                 xi=0.0,):
        """
        Main optimization method.

        Parameters
        ----------
        :param init_points:
            Number of randomly chosen points to sample the
            target function before fitting the gp.

        :param n_iter:
            Total number of times the process is to repeated. Note that
            currently this methods does not have stopping criteria (due to a
            number of reasons), therefore the total number of points to be
            sampled must be specified.

        :param acq:
            Acquisition function to be used, defaults to Upper Confidence Bound.

        Returns
        -------
        :return: Nothing
        """
        # Reset timer
        self.plog.reset_timer()

        # Set acquisition function
        self.util = UtilityFunction(kind=acq, kappa=kappa, use_fixed_kappa=use_fixed_kappa, kappa_scale=kappa_scale, xi=xi)

        # Initialize x, y and find current y_max
        if not self.initialized:
            if self.verbose:
                self.plog.print_header()

            # if we would like to use existing initializations
            if self.use_init != None:
                init = pickle.load(open(init_path + self.use_init, "rb"))
                
                print("[loaded init: {0}]".format(init["X"]))

                self.X, self.Y = init["X"], init["Y"]
                
                self.res['all']['eval_times'] = init["init_eval_times"]
                self.res['all']['epoch_values'] = init["init_epoch_values"]
                
                self.time_started = time.time() - (init["init_eval_times"][-1][-1] - init["init_time_started"])
                self.res['all']['time_started'] = self.time_started

                # if we use BOS, we add the intermedium results as inputs to the GP surrogate function
                if not self.no_BOS:
                    N = self.N
                    y_init = []
                    y_add_all = []
                    for i in range(len(self.X)):
                        epoch_values = self.res['all']['epoch_values'][i]
                        curr_y = self.Y[i]

                        y_init.append(curr_y)
                        
                        #### optionally add some intermediate results to the GP model
                        if self.add_interm_fid != []:
                            y_add = list(np.array(epoch_values)[self.add_interm_fid])
                            y_add_all += y_add

                            x_add = []
                            for fid in self.add_interm_fid:
                                x_add.append(list(self.X[i, :-1]) + [fid / float(N)])
                            x_add = np.array(x_add)
                            self.X = np.concatenate((self.X, x_add), axis=0)
                    y_init += y_add_all
                    self.Y = np.asarray(y_init)

                self.incumbent = np.max(self.Y)
                self.initialized = True
                self.res['all']['init'] = init

                print("Using pre-existing initializations with {0} points".format(len(self.Y)))
            else:
                self.init(init_points)

        y_max = self.Y.max()

        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)

        self.gp = GPy.models.GPRegression(self.X[ur], self.Y[ur].reshape(-1, 1), \
            GPy.kern.Matern52(input_dim=self.X.shape[1], variance=1.0, lengthscale=1.0))

        self.gp.optimize_restarts(num_restarts = 10, messages=False)
        self.gp_params = self.gp.parameters
        print("---Optimized hyper: ", self.gp)

        x_max = acq_max(ac=self.util.utility,
                        gp=self.gp,
                        y_max=y_max,
                        bounds=self.bounds, 
                        iteration=1)


        #### save the posterior stds to be used for the second criteria for early stopping
        N = self.N
        num_init_curve=self.N_init_epochs
        all_fids = np.linspace(0, 1, N+1)[1:]
        all_stds = []
        for fid in all_fids:
            x_fid = np.append(x_max, fid).reshape(1, -1)
            mean, var = self.gp.predict(x_fid)
            std = np.sqrt(var)[0][0]
            all_stds.append(std)

        # Print new header
        if self.verbose:
            self.plog.print_header(initialization=False)
        for i in range(n_iter):
            current_time = time.time()
            if current_time - self.time_started > self.max_running_hours * 3600:
                break

            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            pwarning = False
            if np.any(np.all(self.X[:, :-1] - x_max == 0, axis=1)):
                print("X repeated: ", x_max)
                x_max = np.random.uniform(self.bounds[:, 0],
                                          self.bounds[:, 1],
                                          size=self.bounds.shape[0])
                pwarning = True

            curr_y, curr_fid, BOS_time, epoch_values, time_func_eval = self.f(x_max, no_stop=self.no_BOS, \
                                     incumbent=y_max, bo_iteration=i, stds=all_stds, N=self.N, N_init_epochs=self.N_init_epochs)
            self.res['all']['eval_times'].append(time_func_eval)
            self.res['all']['BOS_times'].append(BOS_time)
            self.res['all']['epoch_values'].append(epoch_values)

            self.Y = np.append(self.Y, curr_y) # the negative sign converts minimization to maximization problem
            self.X = np.vstack((self.X, np.append(x_max, curr_fid).reshape(1, -1)))

            ##### optionally add some results from intermediate epochs to the input of GP
            if (not self.no_BOS) and (self.add_interm_fid != []):
                N = self.N
#                 y_add = np.array(epoch_values)[self.add_interm_fid]
#                 self.Y = np.append(self.Y, y_add)
                for fid in self.add_interm_fid:
                    if fid/float(N) < curr_fid:
                        x_add = np.append(self.X[-1, :-1], fid / float(N)).reshape(1, -1)
                        self.X = np.vstack((self.X, x_add))

                        y_add = epoch_values[fid]
                        self.Y = np.append(self.Y, y_add)

            print("y_max: ", y_max)
            # Update maximum value to search for next probe point.
            if self.Y[-1] > y_max:
                y_max = self.Y[-1]
                self.incumbent = self.Y[-1]

#             # Updating the GP.
            ur = unique_rows(self.X)

            self.gp.set_XY(X=self.X[ur], Y=self.Y[ur].reshape(-1, 1))

            if i >= self.gp_opt_schedule and i % self.gp_opt_schedule == 0:
                self.gp.optimize_restarts(num_restarts = 10, messages=False)
                self.gp_params = self.gp.parameters

                print("---Optimized hyper: ", self.gp)

            x_max = acq_max(ac=self.util.utility,
                            gp=self.gp,
                            y_max=y_max,
                            bounds=self.bounds, 
                            iteration=i+2)


            #### save the posterior stds to be used for the second criteria for early stopping
            N = self.N
            num_init_curve=self.N_init_epochs
            all_fids = np.linspace(0, 1, N+1)[1:]
            all_stds = []
            for fid in all_fids:
                x_fid = np.append(x_max, fid).reshape(1, -1)
                mean, var = self.gp.predict(x_fid)
                std = np.sqrt(var)[0][0]
                all_stds.append(std)

            # Print stuff
            if self.verbose:
                self.plog.print_step(x_max, self.Y[-1], warning=pwarning)

            # Keep track of total number of iterations
            self.i += 1

            self.curr_max_X = self.X[self.Y.argmax()]
            
            x_max_param = self.X[self.Y.argmax(), :-1]

            self.res['all']['params'].append(dict(zip(self.keys, self.X[-1])))

            if self.log_file is not None:
                pickle.dump(self.res, open(self.log_file, "wb"))
            
