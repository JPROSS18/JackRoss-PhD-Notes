
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
lorenz.drive_rate = 0.00
# Setting interval length + other parameters 
rho_pts = torch.arange(10, 30, 0.1)

init = torch.tensor([1, 1, 1], dtype=torch.float32).repeat(rho_pts.shape[0], 1).to(device=device)

init_tensor = torch.cat((init, rho_pts.unsqueeze(1).to(device=device)), dim=1)




torch.save(init_tensor, "data/10-03-26-lorenz_frozen_init_tensor.pt")


init_tensor =torch.load("data/10-03-26-lorenz_frozen_init_tensor.pt", map_location=device)


# Values of L 

dt= 0.01


start_time = time.time()



lle_lorenz = lorenz_analysis.lyapunov_spectrum(
    x0=init_tensor,
    k=1,
    dt=dt,
    t_span = (0, 100),
    t_transient=0,
    non_autonomous=True
)




solve_time = time.time() - start_time
print(f"Lyapunov Exponents computed in {solve_time:.4f} seconds")
print(f"Results tensor shape: {lle_lorenz.shape}")

torch.save(lle_lorenz, "data/10-03-26-lorenz_frozen_lle_tensor.pt")


print("\n-------- COMPLETED --------\n")