from datetime import datetime

current_time = datetime.now()
print('Current Time:', current_time.strftime("%Y-%m-%d %H:%M:%S"), '\n')

print(current_time.strftime("%Y-%m-%d %H:%M:%S"), '--- Saddle Node EWS Example ---\n')



'''
Testing if EWS proposed in Habib et al (2023) is delayed in a dynamic saddle node bifurcation. 
'''

print('Importing Libraries...\n')

import os
import sys
sys.path.append('/Users/jackr/Documents/GitHub/DynamicalSystemsTorch')
os.chdir('/Users/jackr/Documents/GitHub/JackRoss-PhD-Notes/SN_EWS')

#print('Current System Path:', sys.path, '\n')
print('Current Working Directory:', os.getcwd(), '\n')


import matplotlib.pyplot as plt

from torchdiffeq import odeint
import numpy as np



from DynamicalSystemsTorch import Neural_DS as nodes
from DynamicalSystemsTorch import DynamicalSystems as ds


import torch

from tqdm import tqdm


class Nonlinear_Oscilator(ds.DynamicalSystem_torch):
    def __init__(self, a, b):
        '''
        Simple Nonlinear Oscilator. 

        x'' + x + a x' - b (x')^3 (1 - (x')^2) = 0

        Reduced to a pair of first order ODEs:
        x1' = x2
        x2' = - x1 - a x2 + b (x2)^3 (1 - (x2)^2)
        '''
        super().__init__(dim=2)
        self.a = a
        self.b = b
        self.dbdt = 0.00
      
    

    def f(self, t, x):
        t, x = self.f_tests(t, x)


       
        dx1dt = x[:, 1]
        dx2dt = - x[:, 0] - self.a * x[:, 1] + self.b * (x[:, 1]**3)*(1 - x[:, 1]**2)
        return torch.stack([dx1dt, dx2dt], dim = 1)
    
    def na_f(self, t, x):
        t, x = self.f_tests(t, x, driven = True)
        dx1dt = x[:, 1]
        dx2dt = - x[:, 0] - self.a * x[:, 1] + x[:, 2] * (x[:, 1]**3)*(1 - x[:, 1]**2)
        dx3dt = self.dbdt*torch.ones_like(dx1dt)
        print('dbdt:', self.dbdt)
        return torch.stack([dx1dt, dx2dt, dx3dt], dim=1)
    

print('--- Starting Testing ---\n')
system = Nonlinear_Oscilator(a=0.1, b=1.0)

dt = 0.01
t_End = 100
system.dbdt = 0.00

t_pts = torch.arange(0, t_End, dt)
x0 = torch.tensor([0.2, 0.2, 0.6])  # Initial condition

output = system.solve(x0=x0, t_span = (0, t_End), dt=dt, driven = True) # (time, 3)

fig, ax = plt.subplots(1, 2, figsize=(8, 6))
ax[0].plot(output[:, 0], output[:, 1])
ax[1].plot(t_pts, output[:, 0])
ax[1].plot(t_pts, output[:, 1])
plt.show()

print('--- End of Testing ---\n')
