# Import packages
from torchdiffeq import odeint
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


class Lorenz:
    '''
    Class for the Lorenz System of Differential Equations
    --------------
    Parameters:
    rho : function of time
        The rho parameter as a function of time
    sigma : float
        The sigma parameter
    beta : float
        The beta parameter
    --------------
    Methods:
    f(t, x) : function      
    '''
    def __init__(self, rho = lambda t: 14, sigma = 10, beta = 8/3):
        #Setting system parameters
        self.sigma = sigma
        self.beta = beta
        self.rho = rho #should be a 1D function of time

    #Differential Equation
    def f(self, t, x):
        dx1_dt = self.sigma*(x[1]-x[0])
        dx2_dt = x[0]*(self.rho(t) - x[2]) - x[1]
        dx3_dt = x[0]*x[1] - self.beta*x[2]
        
        xdot = [dx1_dt,
                dx2_dt,
                dx3_dt ]

        return np.array(xdot)
    def solve(self, x0, t_span, dt):
        '''
        Solves the Lorenz system using scipy's solve_ivp. Modified to allow for multiple initial conditions.

        Returns a numpy array of shape (num_timepoints, num_trajectories, system dimension)
        --------------
        Parameters:
        x0 : list
            List of initial conditions [[x1_0, x2_0, x3_0], [x1_1, x2_1, x3_1], ...]
        dt : float
            The time step for the solution
        t_span : tuple
            The time span for the solution (t0, t_end)
     
        --------------
        '''
        list = []


        for ic in x0:
            output = integ.solve_ivp(self.f, t_span=t_span, y0=ic,
                                 method='RK45', t_eval=np.arange(t_span[0], t_span[1], dt)).y.T
            list.append(np.expand_dims(output, axis=1))
        
        sol = np.concatenate(list, axis=1)


        return sol