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
    
    
    def solve(self, x0, t_span, dt):
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
        t = torch.arange(t_span[0], t_span[1], dt)
        sol = odeint(func=self.f, y0=x0, t=t, method='rk4', options={'step_size': dt})  # shape (num_timepoints, system dimension, num_trajectories)
        return sol


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