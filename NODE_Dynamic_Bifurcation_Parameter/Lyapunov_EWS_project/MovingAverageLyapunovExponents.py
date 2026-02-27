
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

import models.DynamicalSystems as ds

import models.Neural_ODEs_v2 as nodes

print("\n-------- STARTING --------\n")

torch.set_default_dtype(torch.float32)
torch.set_default_device('mps')
device = torch.device("mps")

os.chdir("/Users/jackr/Documents/GitHub/JackRoss-PhD-Notes/NODE_Dynamic_Bifurcation_Parameter")



model_lorenz = torch.load('networks/lorenz_driven_neural_ODE_m1.pt', weights_only=False).to(device)


node_lorenz = nodes.Neural_DynamicalSystem(model=model_lorenz)
lorenz_analysis_node = ds.DynamicalSystems_analysis(node_lorenz)

node_lorenz.model.drdt = 0.01
lorenz_analysis_node.model.model.drdt = 0.01

# Setting interval length + other parameters 
final_r = 20
dt = 0.05
final_t = int(final_r/lorenz_analysis_node.model.model.drdt)
final_t_pts = int(final_t/dt)

step_pts = int(20/dt)
num_intervals = int(final_t_pts/step_pts)


#print(f"\nComputing trajectory Trajectory length: {final_t} seconds, with dt={dt} seconds, resulting in {final_t_pts} total points.")
#start_time = time.time()
#full_traj = node_lorenz.solve(x0 = torch.tensor([1.0, 1.0, 1.0, 10.0], dtype=torch.float32), t_span=(0, final_t), dt=dt)
#solve_time = time.time() - start_time
#print(f"Trajectory computed in {solve_time:.4f} seconds")



#init = [full_traj[i*step_pts] for i in range(num_intervals)]
#init_tensor = torch.stack(init, dim=0).to(dtype=torch.float32)
#torch.save(init_tensor, f"Lyapunov_EWS_project/init_tensor_all_intervals_lorenz.pt")
init_tensor = torch.load(f"Lyapunov_EWS_project/init_tensor_all_intervals_lorenz.pt").to(device, dtype=torch.float32)


List_Intervals = torch.arange(20, 220.0, 20, dtype = torch.float32)

print(f"\nComputing Lyapunov Exponents for {num_intervals} intervals for {List_Intervals.shape[0]}"+
      f"different interval lengths and {init_tensor.shape[0]} values of rho")
start_time = time.time()
temp_list = []
for i in range(torch.load("Lyapunov_EWS_project/lle_node_num_saved.pt").item()-1, List_Intervals.shape[0]-2):
    lle_node = lorenz_analysis_node.lyapunov_spectrum(x0 =init_tensor.unsqueeze(2), k = 1, t0 = 0, dt = dt,
                                                                num_pts_compute=int(List_Intervals[i]/dt), t_transient_pts=0, non_autonomous=False)
    temp_list.append(lle_node)
    torch.save(lle_node, f"Lyapunov_EWS_project/lle_node_{List_Intervals[i].item():.0f}_interval_lorenz.pt")

lle_tensor = torch.stack(temp_list, dim=0)

    
solve_time = time.time() - start_time
print(f"Lyapunov Exponents computed in {solve_time:.4f} seconds")


print("\n-------- COMPLETED --------\n")