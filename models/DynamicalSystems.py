# Import packages
from torchdiffeq import odeint
import numpy as np
from matplotlib import pyplot as plt

from scipy import integrate as integ
import torch
import torch.nn as nn
from tqdm import tqdm
from ipywidgets import interact, IntSlider

class DynamicalSystem:
    def __init__(self):
        pass

    def f(self, t, x):
        #Constant function, to be overridden by subclasses
        return 1

    def solve(self, x0, t_span, dt):
        '''
        Solves the system using scipy's solve_ivp by RK45. Modified to allow for multiple initial conditions.

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
    

class DynamicalSystem_torch(DynamicalSystem):
    def __init__(self):
        super().__init__()
        pass

    def f(self, t, x): 
        return torch.tensor(self.f(t, x.numpy()), dtype=torch.float32)
    
    def solve(self, x0, t_span, dt):
        '''
        Solves the system using torchdiffeq's odeint. Modified to allow for multiple initial conditions.

        Returns a torch tensor of shape (num_timepoints, num_trajectories, system dimension)
        --------------
        Parameters:
        x0 : torch tensor
            Tensor of initial conditions of shape (system dimension, num_trajectories)
        dt : float
            The time step for the solution
        t_span : tuple
            The time span for the solution (t0, t_end)
     
        --------------
        '''
        t = torch.arange(t_span[0], t_span[1], dt)
        sol = odeint(func=self.f, y0=x0, t=t, method='rk4', options={'step_size': dt})  # shape (num_timepoints, system dimension, num_trajectories)
        return sol


#Systems
class Lorenz(DynamicalSystem_torch): 
    def __init__(self, sigma=10, beta=8/3, rho=28):
        super().__init__()
        self.sigma = sigma
        self.beta = beta
        self.rho = rho

    def f(self, t, x):
        dxdt = self.sigma * (x[1] - x[0])
        dydt = x[0] * (self.rho - x[2]) - x[1]
        dzdt = x[0] * x[1] - self.beta * x[2]
        return torch.tensor([dxdt.numpy(), dydt.numpy(), dzdt.numpy()], dtype=torch.float32)
    
class Rössler(DynamicalSystem_torch):
    def __init__(self, a=0.2, b=0.2, c=5.7):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def f(self, t, x):
        dxdt = -x[1] - x[2]
        dydt = x[0] + self.a * x[1]
        dzdt = self.b + x[2] * (x[0] - self.c)
        return torch.tensor([dxdt.numpy(), dydt.numpy(), dzdt.numpy()], dtype=torch.float32)
    
