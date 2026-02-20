# Import packages
from torchdiffeq import odeint
import numpy as np
from matplotlib import pyplot as plt

from scipy import integrate as integ
import torch
import torch.nn as nn
from tqdm import tqdm
from ipywidgets import interact, IntSlider

import lyapynov as lya
import multiprocessing as mp

'''
This file contains classes for defining and solving dynamical systems, as well as computing their Jacobians and Lyapunov exponents. The base class DynamicalSystem defines the structure for a dynamical system, while the DynamicalSystem_torch class extends it to use PyTorch for automatic differentiation and GPU acceleration. Specific systems like the Lorenz, Rössler, Hopf, and saddle-node bifurcation are implemented as subclasses of DynamicalSystem_torch. The file also includes functions for coordinate transformations between Cartesian and polar coordinates.

To Do: 

Update Lyapunov Exponent Code to run better, using pytorch and autograd functionality.
'''

class DynamicalSystem:
    def __init__(self):
        pass

    def f(self, t, x):
        #Constant function, to be overridden by subclasses
        return 0


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
    def __init__(self, dim):
        super().__init__()
        self.system_dim = dim


    def f_tests(self, t, x, driven = False):
        '''
        Tests for the function f to ensure that the input x is in the correct format.

        driven is a boolean that indicates whether the system is being driven by an external variable. If True, the function will expect x to have an additional dimension for the driver variable.
        '''
        if driven:
            system_dim = self.system_dim + 1
        else:
            system_dim = self.system_dim
        #x sould be (traj, dim)
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        elif len(x.shape) == 1: # When 1d tensor passed. Assume 1 trajectory.  
            if x.shape[0] != system_dim: 
                raise ValueError("Input x must have dimension of ", system_dim, "but got dimension of ", x.shape[0])
            else:
                x = x.unsqueeze(0) # Add dimension to make it (1, dim)
                return t, x # Return t and x for further processing in f.
        elif len(x.shape) == 2: 
            if x.shape[1] != system_dim: 
                raise ValueError("input x must have dimension of ", system_dim, "but got dimension of ", x.shape[1])
            else:
                return t, x
        else:
            raise ValueError("Input tensor x cannot have more than 2 dimensions (trajectory, dimension) but has ", len(x.shape))

    def f(self, t, x): 
        t, x = self.f_tests(t, x)

        return torch.tensor(self.f(t, x.numpy()), dtype=torch.float32)
    
    def na_f(self, t, x):
        t, x = self.f_tests(t, x, driven=True)

        return torch.tensor(self.f(t, x.numpy()), dtype=torch.float32)
    
        
    def jacobian(self, t, x):
        '''
        Computes the Jacobian of the system at a given time t and state x using PyTorch's autograd functionality.
        '''
        func = lambda x, t=t: self.f(t, x)
        return torch.autograd.functional.jacobian(func=func, inputs=x)
    

    def solve(self, x0, t_span, dt, driven=False):
        '''
        Solves the system using torchdiffeq's odeint. Modified to allow for multiple initial conditions.

        Returns a torch tensor of shape (num_timepoints, num_trajectories, system dimension)
        --------------
        Parameters:
        x0 : torch tensor
            Tensor of initial conditions of shape (num_trajectories, system dimension)
        dt : float
            The time step for the solution
        t_span : tuple
            The time span for the solution (t0, t_end)
        --------------
        '''
        if driven: 
            func = self.na_f
        else:
            func = self.f      

        t = torch.arange(t_span[0], t_span[1], dt)
        sol = odeint(func=func, y0=x0, t=t, method='rk4', options={'step_size': dt})  # shape (num_timepoints, system dimension, num_trajectories)
        return sol
    
    def largest_lyapunov_exponent(self, x0, dt, t_span):
        n_compute = int((t_span[1] - t_span[0]) / dt)
        n_transient = int(t_span[0] / dt)
        lya_system = lya.ContinuousDS(x0 = x0, t0 = 0, f=lambda x, t: self.f_numpy(t, x), jac=lambda x, t: self.jacobian_numpy(t, x), dt=dt)
        out = lya.LCE(system = lya_system, p=1, n_forward=n_transient, n_compute=n_compute, keep=False)
        return out
        

    # Numpy wrapper for f and jacobian. 
    def f_numpy(self, t, x):
        x = torch.tensor(x, dtype=torch.float32)
        out = self.f(t, x)
        return out.squeeze(0).detach().numpy()
    
    def jacobian_numpy(self, t, x):
        x = torch.tensor(x, dtype=torch.float32)
        return self.jacobian(t, x).squeeze(0).detach().numpy()
    


#Systems
class Lorenz(DynamicalSystem_torch): 
    def __init__(self, sigma=10, beta=8/3, rho=28):
        super().__init__(dim=3)
        self.sigma = sigma
        self.beta = beta
        self.rho = rho

        self.drive_rate = 0

    def f(self, t, x): # shape should be 
        t, x = self.f_tests(t, x) 
        #print("x shape in f: ", x.shape) # Expecting (num_trajectories, system dimension)
        dxdt = self.sigma * (x[:, 1] - x[:, 0])
        dydt = x[:, 0] * (self.rho - x[:, 2]) - x[:, 1]
        dzdt = x[:, 0] * x[:, 1] - self.beta * x[:, 2]
        
        return torch.stack([dxdt, dydt, dzdt], dim=1)
    
    def na_f(self, t, x):
        t, x = self.f_tests(t, x, driven=True)
        dxdt = self.sigma * (x[:, 1] - x[:, 0])
        dydt = x[:, 0] * (x[:, 3] - x[:, 2]) - x[:, 1]
        dzdt = x[:, 0] * x[:, 1] - self.beta * x[:, 2]
        drdt = self.drive_rate*torch.ones_like(dxdt)
        #print(drdt.shape, dxdt.shape)

        return torch.stack([dxdt, dydt, dzdt, drdt], dim=1)

    
class Rössler(DynamicalSystem_torch):
    def __init__(self, a=0.2, b=0.2, c=5.7):
        super().__init__(dim=3)
        self.a = a
        self.b = b
        self.c = c

    def f(self, t, x):
        t, x = self.f_tests(t, x)
        dxdt = -x[:, 1] - x[:, 2]
        dydt = x[:, 0] + self.a * x[:, 1]
        dzdt = self.b + x[:, 2] * (x[:, 0] - self.c)
        return torch.stack([dxdt, dydt, dzdt], dim=1)
    
class Hopf(DynamicalSystem_torch):
    def __init__(self, rho=1.0, alpha=1.0, omega = 1.0, beta=1.0):
        super().__init__(dim=2)
        self.rho = rho 
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        self.gamma = 0.1 #Rate of change of bifurcation parameter
        


    def f(self, t, x): 
        #x sould be (traj, dim)
        t, x = self.f_tests(t, x)
        
        xdot = self.rho * x[:, 0] - self.omega * x[:, 1] + (self.alpha*x[:, 0] - self.beta * x[:, 1])*(x[:, 0]**2 + x[:, 1]**2) #traj
        ydot = self.omega * x[:, 0] + self.rho * x[:, 1] + (self.beta*x[:, 0] + self.alpha * x[:, 1])*(x[:, 0]**2 + x[:, 1]**2) #(traj)
        return torch.stack([xdot, ydot], dim = 1)
    
    def na_f(self, t, x):
        t, x = self.f_tests(t, x, driven=True)
        xdot = x[:, 2] * x[:, 0] - self.omega * x[:, 1] + (self.alpha*x[:, 0] - self.beta * x[:, 1])*(x[:, 0]**2 + x[:, 1]**2) #traj
        ydot = self.omega * x[:, 0] + x[:, 2] * x[:, 1] + (self.beta*x[:, 0] + self.alpha * x[:, 1])*(x[:, 0]**2 + x[:, 1]**2) #(traj)
        rdot = self.gamma*torch.ones_like(x[:, 0]) #Rate of change of bifurcation parameter
        return torch.stack([xdot, ydot, rdot], dim = 1)

class saddlenode(DynamicalSystem_torch):
    def __init__(self, a=-1.0, r = 0.0):
        super().__init__(dim=1)
        self.a = a
        self.r = r
        self.dadt = 0.0
        self.drdt = 0.0

    def f(self, t, x):
        t, x = self.f_tests(t, x)
        return self.a + (x[:, 0] - self.r)**2
    
    def na_f(self, t, x, bif=True):
        self.f_tests(t, x, driven=True)
        if bif:
            dxdt = x[:, 1] + (x[:, 0] - self.r)**2
            dadt = self.dadt*torch.ones_like(dxdt)
            return torch.stack((dxdt, dadt), dim=1)
        else: 
            dxdt = self.a + (x[:, 0] - x[:, 1])**2
            drdt = self.drdt*torch.ones_like(dxdt)
            return torch.stack((dxdt, drdt), dim=1)

class DynamicalSystems_analysis: 
    '''
    Class to analyze the dynamical system, including computing Lyapunov exponents, bifurcation diagrams, and other relevant metrics.

    Attributes:
    - model: The dynamical system model to be analyzed. Should be an instance of DynamicalSystem_torch. 

    Methods:
    - compute_lyapunov_exponent(x0, dt, t_span): Computes the largest Lyapunov exponent for the given initial condition, time step, and time span.

    '''
    def __init__(self, model: DynamicalSystem_torch):
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

#Functions for coordinate transformations
def cartesian_to_polar(x, y):
    """
    Convert Cartesian coordinates to polar coordinates.
    
    Parameters:
    x, y: float or array-like
        Cartesian coordinates
    
    Returns:
    r: float or array
        Radius (distance from origin)
    theta: float or array
        Angle in radians (from positive x-axis)
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def polar_to_cartesian(r, theta):
    """
    Convert polar coordinates to Cartesian coordinates.
    
    Parameters:
    r: float or array-like
        Radius (distance from origin)
    theta: float or array-like
        Angle in radians (from positive x-axis)
    
    Returns:
    x, y: float or array
        Cartesian coordinates
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y