
print("\n Importing libraries...\n")
import os
import sys
import time
from torchdiffeq import odeint as odeint

sys.path.append("/Users/jackr/Documents/GitHub/JackRoss-PhD-Notes")

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
import lyapynov as lya 

import multiprocessing as mp

import models.DynamicalSystems_new as ds

import models.Neural_ODEs_v2 as nodes

'''
File to compute the finite time lyapunov exponents (FTLE) for the Lorenz System. 
'''

print("\n-------- STARTING --------\n")

torch.set_default_dtype(torch.float32)
torch.set_default_device('mps')
device = torch.device("mps")

os.chdir("/Users/jackr/Documents/GitHub/JackRoss-PhD-Notes/NODE_Dynamic_Bifurcation_Parameter/Lyapunov_EWS_project")

# Create lorenz system and analysis wrapper
lorenz = ds.Lorenz()
lorenz.f = lorenz.na_f
lorenz_analysis = ds.DynamicalSystems_analysis(lorenz)


# Set rate of change for bifurcation parameter
lorenz.drive_rate = 0.01

# Setting interval length + other parameters 
rho_evolve = 25
dt = 0.01
final_t = int(rho_evolve / abs(lorenz.drive_rate))
final_t_pts = int(final_t / dt)


#
step = 10 #Difference between start points of FTLE computation.
step_pts = int(step / dt) #Difference between start points of FTLE computation, in number of time points.
num_intervals = int(final_t_pts / step_pts) # Number of FTLE values to be computed. 


'''
# Compute trajectory with time-varying bifurcation parameter
print(f"\nComputing trajectory length: {final_t} seconds, dt={dt}, total points={final_t_pts}")
start_time = time.time()

full_traj = lorenz.solve(
    x0=torch.tensor([[1.0, 1.0, 1.0, 10.0]], dtype=torch.float32, device=device),
    t_span=(0, final_t),
    dt=dt,
    driven=True
).squeeze(1)

solve_time = time.time() - start_time
print(f"Trajectory computed in {solve_time:.4f} seconds")

# Extract initial conditions at each interval
init = [full_traj[i * step_pts] for i in range(num_intervals)]
init_tensor = torch.stack(init, dim=0).to(device=device, dtype=torch.float32)

torch.save(init_tensor, "data/10-03-26-lorenz_init_tensor.pt")
'''


init_tensor =torch.load("data/10-03-26-lorenz_init_tensor.pt", map_location=device)


# Values of L 
L_values = torch.arange(30, 120, 20, dtype=torch.float32) 
dt_new = 0.01
print(f"\nComputing Lyapunov Exponents for {num_intervals} intervals for {L_values.shape[0]} "+
      f"different interval lengths and {init_tensor.shape[0]} values of a")
start_time = time.time()
temp_list = []
print(lorenz_analysis.model.f)
for i in range(0, L_values.shape[0] - 1):
    lle_lorenz = lorenz_analysis.lyapunov_spectrum(
        x0=init_tensor,
        k=1,
        dt=dt_new,
        t_span=(0, L_values[i].item()),
        t_transient=0,
        non_autonomous=True
    )
    temp_list.append(lle_lorenz )
    print(f"  Interval {i+1}/{L_values.shape[0]}: {L_values[i].item()} time units")

lle_tensor = torch.stack(temp_list, dim=0)

solve_time = time.time() - start_time
print(f"Lyapunov Exponents computed in {solve_time:.4f} seconds")
print(f"Results tensor shape: {lle_tensor.shape}")

torch.save(lle_tensor, "data/10-03-26-lorenz_lle_tensor.pt")
torch.save(L_values, "data/10-03-26-lorenz_L_values.pt")

print("\n-------- COMPLETED --------\n")