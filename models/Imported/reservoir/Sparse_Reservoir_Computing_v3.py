import numpy as np

from matplotlib import pyplot as plt
from numpy import linalg as la
import os
from scipy import stats
from scipy import spatial as sp
from scipy import sparse

from scipy import linalg as sla
from scipy import integrate as integ



import pandas as pd
import networkx as nx
from tqdm import tqdm
#import multiprocess as mp

#My files

import models.SystemsSolved as s
import reservoir.Methods as m

import Sparse_Reservoir_Computing as rc

import RecurrencePlotsCode as rp

import Parallel_Grid_Search1 as pgs
import models.DynamicalSystems as ds

import multiprocess as mp


######### RNN Class 
class sparse_RNN:
    ''' 
    Creates a sparse weighted Erdos-Reyni recurrent neural network class. 

    Attributes
    ----------
    input_dim : int
        Dimension of the input data.
    nodes : int
        Number of nodes in the network.
    p : float
        Probability of edge creation in the Erdos-Reyni graph.
    seed : int, optional
        Random seed for reproducibility (default is 12).


    Variables
    ---------   
    network : scipy.sparse.csc_matrix
        Sparse adjacency matrix of the network with weights.

    network_nx : networkx.Graph
        NetworkX representation of the network.

    Methods
    -------
    set_spectral_radius(spectral_radius)
        Sets the spectral radius of the network to the specified value.


    '''
    def __init__(self, input_dim, nodes, p, seed=12):   
        self.input_dim = input_dim
        self.nodes = nodes
        self.p = p
        self.seed = seed

        #Sets Network by weighted erdos-reyni topology
        np.random.seed(self.seed) 
        network1 = nx.fast_gnp_random_graph(self.nodes, p, seed=self.seed) #Graph ER
        edges1 = nx.number_of_edges(network1) #number of edges
        wts = np.random.uniform(-1, 1, size=edges1) #weights for each edge

        #Getting weights as tuples
        edge_tuples = tuple(network1.edges) #list of edges 
        weighted_edge_tuples = []
        for i in range(0, edges1):
            weighted_edge_tuples.append(edge_tuples[i] + (wts[i],))
        
        network1.add_weighted_edges_from(weighted_edge_tuples) 
        self.network = nx.to_scipy_sparse_array(network1, format='csc')
        self.network_nx = network1

    def get_spectral_radius(self):
        ''' 
        Returns the spectral radius of the network.

        Returns
        -------
        float
            Spectral radius of the network.
        
        '''
        current_spectral_radius = np.abs(sparse.linalg.eigs(self.network, k=1, which='LM', return_eigenvectors=False)[0])
        return current_spectral_radius

    def set_spectral_radius(self, spectral_radius):
        ''' 
        Sets the spectral radius of the network to the specified value.

        Attributes
        ----------
        spectral_radius : float
            Desired spectral radius for the network.
        
        '''
        current_spectral_radius = np.abs(sparse.linalg.eigs(self.network, k=1, which='LM', return_eigenvectors=False)[0])
        
        self.network = (self.network / current_spectral_radius) * spectral_radius
    
   


###### Getting Reservoir Class 
class reservoir(sparse_RNN):
    '''
    Class to set up reservoir computer 
    '''
    def __init__(self, input_dim, nodes, p, seed=12):  
        '''
        - Inherits sparse RNN
        - Sets W_in Matrix 
        ''' 
        super().__init__(input_dim, nodes, p, seed)
        np.random.seed(self.seed)
         
        #Setting in matrix 
        W = np.zeros((self.nodes, self.input_dim))
        Col = np.random.randint(0, self.input_dim, self.nodes) #Selecting column
        Val = np.random.uniform(-1, 1, self.nodes) #values 
        
        for i in range(0, self.nodes): #Could elimiate this loop (return to)
            W[i, Col[i]] = Val[i]

        self.W_in = sparse.csc_matrix(W)

        ######################################
        #Setting rdot 
        #self.gamma = gamma 
        #self.sigma = sigma 
        self.u = None
        self.dt = None
        self.W_out = None

    def set_u(self, u, dt=0.01):
        '''
        - Sets the input u for the reservoir.

        Attributes
        ----------  
        u : np.ndarray
            Input data array of shape (dim, time).
        dt : float
            Time step between input data points (default is 0.01).
        '''
        if not isinstance(u, np.ndarray):
            raise TypeError("Input u must be a 2D numpy array of shape (dim, time).")
        else:
            self.u = u #(dim, time)
            self.dt = dt

    def f_listen(self, t, r, gamma, sigma):
        '''
        - Reservoir update functions for listening reservoir.
        - Must set input u and dt prior to calling this function.
        Attributes
        ----------  
        t : float
            Current time.       
        r : np.ndarray
            Current reservoir state. Shape (nodes, ).

        '''
        
        if self.u is None or self.dt is None:
            raise ValueError("Input u and dt must be set before calling f_listen.")
        elif self.u.shape[0] != self.input_dim:
            raise ValueError("Input u must have shape (input_dim, time).")
        elif self.u.shape[1] <= int(t/self.dt):
            raise ValueError("Time t exceeds the range of the drive system input, u.")
        else:
            a = self.network @ r
            b = sigma*(self.W_in @ self.u[:, int(t/self.dt)])
            out  = gamma*(-r + np.tanh(a + b))
            return out
    
    def q(self, r):
        '''
        Non linear function to breaking systemmtry in predicting reservoir.
        '''
        if type(r) is not np.ndarray:
            raise TypeError("Input r must be a numpy array.")
        
        elif len(r.shape) == 2: #(nodes, time)
            q = np.zeros((2*self.nodes, r.shape[1]))
            temp2 = r*r; temp1 = r
            q[0:self.nodes, :] = temp1; q[self.nodes:2*self.nodes, :] = temp2
            return q
        
        elif len(r.shape) == 1:

            q = np.zeros((2*self.nodes))
            temp2 = r*r; temp1 = r
            q[0:self.nodes] = temp1; q[self.nodes:2*self.nodes] = temp2
            return q
        else:
            raise ValueError("Input r must be a 1D or 2D numpy array.")
        
    def set_W_out(self, r, tStart, tEnd, lamda = 1e-6):
        ''' 
        Sets the W_out matrix for predicting. 

        Must use method set_u prior to calling this function.

        Parameters
        ----------  
        r : np.ndarray
            Reservoir states collected during listening phase. Shape (nodes, time).
        
        tStart : float
            Start time for training data.   

        tEnd : float
            End time for training data.

        lamda : float
            Regularization parameter for ridge regression (default is 1e-6).
        '''
        if self.u is None or self.dt is None:
            raise ValueError("Input u and dt must be set before calling set_W_out.")
        elif tEnd <= tStart:
            raise ValueError("tEnd must be greater than tStart.")
        elif tStart < 0 or int(tEnd/self.dt) > self.u.shape[1]:
            raise ValueError("tStart and tEnd must be within the range of the input data.")
        else:
            iStart = int(tStart/self.dt)
            iEnd = int(tEnd/self.dt)
            #iTrain = iEnd - iStart
            #print(f"Setting W_out using data from t={tStart} to t={tEnd}, indices {iStart} to {iEnd}.")

            #Training Data 
            uTrain = self.u[:, iStart:iEnd]
            rTrain = r[:, iStart:iEnd]

            #Break Symmetry in Data
            qTrain = np.zeros((2*self.nodes, iEnd - iStart))
            qTrain[0:self.nodes, :] = rTrain
            qTrain[self.nodes:2*self.nodes, :] = rTrain*rTrain

            #Q = np.concatenate([np.identity(self.R.Nodes), np.diag(rTrain)])
            #qTrain = Q @ rTrain

            A1 = np.matmul(qTrain, np.transpose(qTrain)) + np.identity(2*self.nodes)*(lamda)
            A2 = la.inv(A1)
            A3 = np.matmul(np.transpose(qTrain), A2)
            W = np.matmul(uTrain, A3)
            self.W_out = W 
            #print("W_out matrix set.")

    def f_predicting(self, t, r, gamma, sigma):
        ''' 
        Reservoir update functions for predicting reservoir.f_listen

        Must set W_out matrix before calling this function.
        '''
        if self.W_out is None:
            raise ValueError("Must set W_out matrix before calling f_predicting")
        else:
            q = np.zeros((2*self.nodes))
            temp2 = r*r; temp1 = r
            q[0:self.nodes] = temp1; q[self.nodes:2*self.nodes] = temp2

            u_hat = self.W_out @ q 
            
            a = self.network @ r
            b = sigma*(self.W_in @ u_hat)

            out = gamma*(-r + np.tanh(a+b))
            return out
        
class optimise_RC(reservoir):
    '''
    Class to optimise reservoir computer over multiple parameter combinations.
    Inherits reservoir class.
    Attributes
    ----------  
    data : np.ndarray
        Input data for the reservoir computer.
    dt : float
        Time step size.
    nodes : int
        Number of nodes in the reservoir.
    p : float, optional
        Sparsity of the reservoir connectivity matrix (default is 0.04).    
    
    Methods
    ------- 
    single_test(gamma, sigma, rho, t0_Listen, tEnd_Listen, t0_train, t_predicting):
        Tests a single combination of parameters and returns the predicted trajectory.

    multiple_test(gamma_list, sigma_list, rho_list, t0_Listen, tEnd_Listen, t0_train, t_predicting):
        Tests multiple combinations of parameters and returns a list of predicted trajectories.

    '''
    def __init__(self, data, dt, nodes, p=0.04):
        '''
        Initializes the optimise_RC class.
        Sets RNN and target data

        Target data should be of shape (dim, time) and have constannt time step dt. 
        '''
        super().__init__(input_dim = data.shape[0], nodes=nodes, p = p)
        self.set_u(data, dt=dt)
        self.data = data
        self.dt = dt
    
    def single_test(self, gamma, sigma, rho, t0_Listen, tEnd_Listen, t0_train, t_predicting):
        '''
        Tests a single combination of parameters and returns the predicted trajectory. 
        
        Returns the np.ndarray containing predicted trajectory for the given parameters.

        Parameters
        ----------  
        gamma : float
            Gamma (timescale) parameter in reservoir update equation.
        
        sigma : float
            Sigma (drive) parameter for the reservoir dynamics.

        rho : float
            Spectral radius of the reservoir network.

        t0_Listen : float
            Start time for the listening phase.

        tEnd_Listen : float
            End time for the listening phase.

        t0_train : float
            Start time for the training phase.

        t_predicting : float
            Duration of the predicting phase.
        '''
        self.set_spectral_radius(rho)

        #Listening
        r_listen = integ.solve_ivp(fun=self.f_listen, t_span=(t0_Listen, tEnd_Listen), 
                           y0=np.zeros(self.nodes), t_eval=np.arange(t0_Listen, tEnd_Listen, self.dt), method='RK45', args=(gamma, sigma)).y
        u_listen = self.data[:, int(t0_Listen/self.dt):int(tEnd_Listen/self.dt)]


        #Training
        #Need to supress printing here 
        self.set_W_out(r_listen, tStart=t0_train, tEnd=tEnd_Listen, lamda=10**-6)

        ## Predicting 
        t0_predicting = tEnd_Listen; tEnd_predicting = t_predicting + t0_predicting
        r_predicting = np.array(integ.solve_ivp(fun=self.f_predicting, t_span=(t0_predicting, tEnd_predicting), 
                                y0=r_listen[:, -1], t_eval=np.arange(t0_predicting, tEnd_predicting, self.dt), method='RK45', args=(gamma, sigma)).y)
        u_predicting = (self.W_out @ self.q(r_predicting))

        return u_predicting
    
    def multiple_test(self, gamma_list, sigma_list, rho_list, t0_Listen, tEnd_Listen, t0_train, t_predicting):
        '''
        Tests multiple combinations of parameters and returns a list of predicted trajectories.

        Gamma_list, sigma_list, rho_list should be lists of parameter values to test.

        Loading bars need to reworked (currently only shows sigma)

        Uses simple loops 
        '''
        num_test = len(gamma_list)*len(sigma_list)*len(rho_list)
        print('Number of parameter combinations tested:', num_test)

        test_traj_list = []
        for i in tqdm(range(0, len(sigma_list)), desc='Sigma Loop', leave=True):
            for j in tqdm(range(0, len(rho_list)), desc='Rho Loop', leave=False):
                for k in range(0, len(gamma_list)):
                    gamma = gamma_list[k]
                    sigma = sigma_list[i]
                    rho = rho_list[j]
                    u_predicting = self.single_test(gamma, sigma, rho, t0_Listen, tEnd_Listen, t0_train, t_predicting)
                    test_traj_list.append(u_predicting)

        #Listening
        self.test_traj_list = test_traj_list


        return test_traj_list

    def _parallel_single_test_wrapper(self, params):
        """
        Wrapper function for parallel execution of single_test.
        Takes a tuple of parameters and unpacks them for single_test.
        """
        gamma, sigma, rho, t0_Listen, tEnd_Listen, t0_train, t_predicting = params
        return self.single_test(gamma, sigma, rho, t0_Listen, tEnd_Listen, t0_train, t_predicting)

    def multiple_test_parallel(self, gamma_list, sigma_list, rho_list, t0_Listen, tEnd_Listen, t0_train, t_predicting, n_jobs=None):
        """
        Parallelized version of multiple_test using multiprocessing.
        
        Parameters
        ----------
        gamma_list : list
            List of gamma values to test.
        sigma_list : list
            List of sigma values to test.
        rho_list : list
            List of rho values to test.
        t0_Listen : float
            Start time for listening phase.
        tEnd_Listen : float
            End time for listening phase.
        t0_train : float
            Start time for training phase.
        t_predicting : float
            Duration of predicting phase.
        n_jobs : int, optional
            Number of parallel jobs. If None, uses all available cores.
        
        Returns
        -------
        list
            List of predicted trajectories for all parameter combinations.
        """
        from itertools import product
        
        num_test = len(gamma_list) * len(sigma_list) * len(rho_list)
        print(f'Number of parameter combinations tested: {num_test}')
        
        # Create all parameter combinations
        param_combinations = list(product(gamma_list, sigma_list, rho_list))
        
        # Add fixed parameters to each combination
        params_with_fixed = [(g, s, r, t0_Listen, tEnd_Listen, t0_train, t_predicting) 
                             for g, s, r in param_combinations]
        
        # Set number of cores
        if n_jobs is None:
            n_jobs = mp.cpu_count()
        
        print(f'Using {n_jobs} parallel jobs')
        
        # Execute in parallel
        with mp.Pool(processes=n_jobs) as pool:
            test_traj_list = list(tqdm(
                pool.imap_unordered(self._parallel_single_test_wrapper, params_with_fixed),
                total=len(params_with_fixed),
                desc='Grid Search Progress'
            ))
        
        self.test_traj_list = test_traj_list
        return test_traj_list

    def multiple_test_parallel_ordered(self, gamma_list, sigma_list, rho_list, t0_Listen, tEnd_Listen, t0_train, t_predicting, n_jobs=None):
        """
        Parallelized version of multiple_test with results in consistent order.
        
        Maintains the same order as the original multiple_test method (sigma -> rho -> gamma).
        Slightly slower than multiple_test_parallel due to ordered processing.
        
        Parameters
        ----------
        gamma_list : list
            List of gamma values to test.
        sigma_list : list
            List of sigma values to test.
        rho_list : list
            List of rho values to test.
        t0_Listen : float
            Start time for listening phase.
        tEnd_Listen : float
            End time for listening phase.
        t0_train : float
            Start time for training phase.
        t_predicting : float
            Duration of predicting phase.
        n_jobs : int, optional
            Number of parallel jobs. If None, uses all available cores.
        
        Returns
        -------
        list
            List of predicted trajectories in the same order as parameter combinations.
        """
        num_test = len(gamma_list) * len(sigma_list) * len(rho_list)
        print(f'Number of parameter combinations tested: {num_test}')
        
        # Create parameter combinations in nested loop order (matching original)
        param_combinations = []
        for sigma in sigma_list:
            for rho in rho_list:
                for gamma in gamma_list:
                    param_combinations.append((gamma, sigma, rho, t0_Listen, tEnd_Listen, t0_train, t_predicting))
        
        # Set number of cores
        if n_jobs is None:
            n_jobs = mp.cpu_count()
        
        print(f'Using {n_jobs} parallel jobs')
        
        # Execute in parallel with ordered results
        with mp.Pool(processes=n_jobs) as pool:
            test_traj_list = list(tqdm(
                pool.imap(self._parallel_single_test_wrapper, param_combinations),
                total=len(param_combinations),
                desc='Grid Search Progress'
            ))
        
        self.test_traj_list = test_traj_list
        return test_traj_list

class corr_dim():
    def __init__(self, data):
        '''
        Parameters
        ----------
        data : array
            Data points of the dynamical system.
        epsilon_pts : array
            Epsilon points to compute correlation integral.'''
        
        self.data = data
        self.epsilon_pts = None
        self.L2_matrix = rp.distance_matrix(data=self.data, p=2)
        self.coef = None
        self.epsilon_pts = None
    
    def compute_corr_int(self, epsilon_pts):
        '''
        Computes Correlation Integral for specificed epsilons. 

        '''
        self.epsilon_pts = epsilon_pts
        corr_int_list = []

        for epsilon in epsilon_pts:
            matrix = (self.L2_matrix < epsilon)
            ci = rp.correlation_integral(matrix)
            corr_int_list.append(ci)
        
        self.corr_int_list = corr_int_list
   
    
    def linear_regression(self, data, target):
        
        X = np.vstack((data, np.ones(data.shape[0]))).T
        Y = target
        try: 
            inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            self.coef = np.array([np.nan, np.nan])
            return None
        
        self.coef = inv @ (X.T @ Y)
        #if np.any(np.isnan(self.coef)):
        #    raise ValueError("Linear regression failed. Check input data.")
    
    def lr_predict(self, x):
        if self.coef is None:
            raise ValueError("Model is not fitted yet. Call 'linear_regression' first.")
        
        return self.coef[0]*x + self.coef[1]

    def test_lr(self, i_Start, i_End):
        if self.epsilon_pts is None or self.corr_int_list is None:
            raise ValueError("Correlation integral not computed. Call 'compute_corr_int' first.")
        
        log_epts = np.log(self.epsilon_pts[i_Start:i_End])
        log_cint = np.log(self.corr_int_list[i_Start:i_End])

        self.linear_regression(data = log_epts, target = log_cint)


        r_squared = np.corrcoef(self.lr_predict(log_epts), log_cint)[0, 1]**2
        return r_squared
    
    def find_correlation_dimension(self, max_iter=1000, r2_threshold=0.999, print_info=True):
        init_i_Start = 0
        init_i_End = self.epsilon_pts.shape[0]
        

        if print_info:
            print(' ')
            print("Finding Correlation Dimension:")

        for i in range(0, max_iter):
            #Discard bottom 
            a = self.test_lr(init_i_Start+1, init_i_End)
            #Discard Top
            b = self.test_lr(init_i_Start, init_i_End-1)
            if init_i_End - init_i_Start <= 2:
    
                return np.nan
            elif np.isnan(a) or np.isnan(b): 
                init_i_End -= 1
                init_i_Start += 1
            else: 
                if a > b: 
                    init_i_Start += 1
                    c = a
                elif b >= a:
                    init_i_End -= 1
                    c = b
           
                if c > r2_threshold:
                    break

        final_i_Start = init_i_Start
        final_i_End = init_i_End

        log_epts = np.log(self.epsilon_pts[final_i_Start:final_i_End])
        log_cint = np.log(self.corr_int_list[final_i_Start:final_i_End])

        self.linear_regression(data = log_epts, target = log_cint)
        if print_info:
            print('Correlation Dimension:', self.coef[0])
            print(' ')
            print('Info: ')
            print(f"Final Indices: {init_i_Start}, {init_i_End}")
            print(f"R^2: {c}")
            print('i =', i)
        return self.coef[0]



if __name__ == "__main__":
    #Using multiprocessing to speed up grid search
    
    sigma_list = np.linspace(0.01, 0.95, 10)
    rho_list = [0.5]
    gamma_list = [1]
    import multiprocess as mp
    
    #Get in right format for multiprocessing
    param_list = []
    for sigma in sigma_list:
        for rho in rho_list:
            for gamma in gamma_list:
                param_list.append([sigma, rho, gamma])
    
    # Single test wrapper 
    ## Data Generation: Coupled Rossler System
    nu1 = 0.02
    mu1 = 0.0
    mu2 = 0.5

    coupled_rossler = ds.yCoupledRossler(a = 0.2, b = 0.2, c = 5.7, nu1 = nu1, nu2 = nu1, mu1 = mu1, mu2 = mu2)
    t_Start = 0; t_Final = 1000; dt = 0.01


    coupled_rossler_data = integ.solve_ivp(fun=coupled_rossler.f, t_span=(t_Start, t_Final), y0=[1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                                    method = 'RK45', t_eval = np.arange(t_Start, t_Final, dt)).y
    coupled_rossler_data.shape

    A0 = coupled_rossler_data[0:3, :]
    B0 = coupled_rossler_data[3:6, :]
    orc = optimise_RC(data=A0, dt=0.01, nodes=100, p=0.04)


