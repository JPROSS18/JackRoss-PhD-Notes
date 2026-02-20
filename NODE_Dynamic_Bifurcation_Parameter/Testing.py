import os
import sys
from torchdiffeq import odeint as odeint



import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits import mplot3d
from numpy import linalg as la
from scipy import stats
from scipy import spatial as sp
from scipy import integrate as integ
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from ipywidgets import interact, IntSlider
import types



sys.path.append("/Users/jackr/Documents/GitHub/JackRoss-PhD-Notes")




import models.DynamicalSystems as ds 
import models.Neural_ODEs_v2 as node



import lyapynov as lya
import multiprocessing as mp

class DynamicalSystems_analysis: 
    '''
    Class to analyze the dynamical system, including computing Lyapunov exponents, bifurcation diagrams, and other relevant metrics.

    Attributes:
    - model: The dynamical system model to be analyzed. Should be an instance of DynamicalSystem_torch. 

    Methods:
    - compute_lyapunov_exponent(x0, dt, t_span): Computes the largest Lyapunov exponent for the given initial condition, time step, and time span.

    '''
    def __init__(self, model: ds.DynamicalSystem_torch):
        self.model = model 

    def lyapunov_exponents(self, x0, dt, t_span: tuple, t0=0, num_exponents=1, keep_data=False):
        '''
        Computes the largest Lyapunov exponent for the given model, initial condition, time step, and time span.

        Uses methods from the lyapynov package to compute the Lyapunov exponent.

        Currently requires numpy array - will need to modify to work with torch tensors in future. 

        Parameters:
        - x0: The initial condition for the system. Should be a numpy array.
        - dt: The time step for the computation.
        - t_span: A tuple (t_start, t_end) specifying the time span for the computation. Computation will start at t = 0 and discard items before t_start as transient, and compute the Lyapunov exponent over the interval [t_start, t_end].

        - num_exponents: The number of Lyapunov exponents to compute. Default is 1 (largest Lyapunov exponent).
        - keep_data: Whether to keep the full data of the Lyapunov exponent computation. Default is False (only keep the final exponent values).
        '''
        if not isinstance(x0, np.ndarray):
            raise ValueError("x0 should be a numpy array.")
        
        if not isinstance(dt, (int, float)):
            raise ValueError("dt should be a numeric value.")
        
        if num_exponents < 1 or num_exponents > self.model.system_dim:
            raise ValueError("num_exponents should be at least 1 and no greater than system dimension ({}).".format(self.model.system_dim))

        n_compute = int((t_span[1] - t_span[0]) / dt)
        n_transient = int(t_span[0] / dt)
        lya_system = lya.ContinuousDS(x0 = x0, t0 = t0, f=lambda x, t: self.model.f_numpy(t, x), jac=lambda x, t: self.model.jacobian_numpy(t, x), dt=dt)
        out = lya.LCE(system = lya_system, p=num_exponents, n_forward=n_transient, n_compute=n_compute, keep=keep_data)
        return out
    
    def lyapunov_exponents_parallel(self, x0, dt, interval_length, t_Final, processes=1, t0=0, num_exponents=1, keep_data=False):
        ''' 
        
        '''
        num_intervals = int(t_Final/interval_length) 
        
        list = []
        for i in range(0, num_intervals):
                list.append((interval_length*i, interval_length*(i+1)))

        def parallel_func(self, interval):
            return self.lyapunov_exponents(x0=x0, dt=dt, t_span=interval, num_exponents=num_exponents, keep_data=False)  
        
        with mp.Pool(processes=processes) as pool:
            results = pool.map_async(func=parallel_func, iterable=list)

        return results.get()

    
model_lorenz = torch.load('/Users/jackr/Documents/GitHub/JackRoss-PhD-Notes/NODE_Dynamic_Bifurcation_Parameter/networks/lorenz_driven_neural_ODE_m1.pt', weights_only=False)

class Neural_DynamicalSystem(ds.DynamicalSystem_torch):
    def __init__(self, model):
        super().__init__(dim=model.input_dim)
        self.model = model 

    def f(self, t, x):
        t, x = self.f_tests(t, x)
        return self.model(t, x)
    
node_lorenz = Neural_DynamicalSystem(model = model_lorenz)
node_lorenz.model.drdt = 0.01
lorenz_analysis = DynamicalSystems_analysis(model = node_lorenz)


def test_func2(x):
    return lorenz_analysis.lyapunov_exponents(x0=np.array([1.0, 1.0, 1.0, 10.0]), dt=0.01, t_span=x, num_exponents=1, keep_data=False)    
    
if __name__ == '__main__':
    print('  ')
    print('-------------------------------------------------')
    print('Running File: ')
    print(' ')
    print('node_lorenz dr/dt:', node_lorenz.model.drdt)

    output = lorenz_analysis.lyapunov_exponents_parallel(x0 = np.array([1.0, 1.0, 1.0, 10.0]), dt = 0.01, interval_length=50, t_Final = 2_000, processes=mp.cpu_count()-2, t0 = 0, num_exponents=1)
    print(output)
    '''
        list = []
        for i in tqdm(range(0, 20)):
            list.append((50*i, 50*(i+1)))

        print('CPU count: ', mp.cpu_count())

        with mp.Pool(processes=mp.cpu_count()-2) as pool:
            results = pool.map(test_func2, list)

        print(results)

    '''

    print(' ')
    print('End of File')
    print('-------------------------------------------------')
    print(' ')

    

