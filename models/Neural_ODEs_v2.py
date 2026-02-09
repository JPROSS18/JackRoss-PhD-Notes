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


class Simple_FeedforwardNN(nn.Module):
    '''
    Class for a simple feedforward neural network.
    Takes as input the input dimension, depth (number of hidden layers), width (number of neurons per hidden layer), output dimension, and activation function.

    By default, the activation function is set to Tanh.

    Parameters:
    ----------
    input_dim : int
        Dimension of the input layer.   
    depth : int
        Number of hidden layers.
    width : int
        Number of neurons per hidden layer.
    output_dim : int
        Dimension of the output layer.
    activation_func : torch.nn module
        Activation function to be used in the hidden layers. Default is Tanh.
    '''
    def __init__(self, input_dim: int, depth: int, width: int, output_dim: int, activation_func: nn.Module = nn.Tanh()):
            # Zero-arg super avoids binding issues in interactive sessions
        super().__init__()

        layers = []
        previous_depth = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(previous_depth, width))
            layers.append(activation_func)
            previous_depth = width

        layers.append(nn.Linear(width, output_dim))
        self.network = nn.Sequential(*layers)

        #Setting initial weights
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.2)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, x):
            out = self.network(x)
            return out
    
class NODE(Simple_FeedforwardNN):
    '''
    Class that creates an Neural ODE. 
    '''
    def __init__(self, input_dim: int, output_dim: int, depth: int, hidden_dim: int, activation_func: nn.Module = nn.Tanh(), t_span: tuple = (0, 0.05), dt: float = 0.01):
        super().__init__(input_dim=hidden_dim, depth=depth, width=hidden_dim, output_dim=hidden_dim, activation_func=activation_func)
        if input_dim < output_dim:
            raise ValueError("Input dimension must be greater than or equal to output dimension in a neural ODE.")
        else:
            self.hidden_dim        = hidden_dim #width is number of neurons per hidden layer
            self.depth             = depth #Number of internal hidden layers
            self.input_dim         = input_dim
            self.output_dim        = output_dim
            self.activation_func   = activation_func

            self.input_layer = nn.Linear(input_dim, hidden_dim)
            self.output_layer = nn.Linear(hidden_dim, output_dim)

            #Integration settings
            self.t_span = t_span    
            self.dt = dt

    


    def f(self, t, x): #x shoudl be (num_traj, spatial_dim)
        '''Takes t and x as input, where x is a pytorch tensor with shape: [trajectories, dim].
        
        input x is hidden state. 
        '''
        out = self.network(x) #(num_traj, spatial_dim+drivers)
        return out
    
    def forward(self, t, x):
        '''
        Takes t and x as input, where x is a pytorch tensor with shape: [trajectories, dim].

        Maps to hidden dimension, then integrates ODE, then maps back to output dimension.

        Returns output with shape: (time, num_traj, output_dim)
        
        '''
        z = self.input_layer(x)
        out = odeint(self.f, z, torch.arange(self.t_span[0], self.t_span[1], self.dt), method='rk4', options={'step_size': self.dt}) # out shape (time, num_traj, hidden_dim)
        out = self.output_layer(out) # out2 shape (time, num_traj, output_dim)
        return out 
    
    
class neural_ODE(Simple_FeedforwardNN):
    '''
    Class that creates an Neural ODE. 
    '''
    def __init__(self, input_dim: int, output_dim: int, depth: int, hidden_dim: int, activation_func: nn.Module = nn.Tanh()):
        super().__init__(input_dim=input_dim, depth=depth, width=hidden_dim, output_dim=output_dim, activation_func=activation_func)
        if input_dim < output_dim:
            raise ValueError("Input dimension must be greater than or equal to output dimension in a neural ODE.")
        else:
            self.hidden_dim        = hidden_dim #width is number of neurons per hidden layer
            self.depth             = depth #Number of internal hidden layers
            self.input_dim         = input_dim
            self.output_dim        = output_dim
            self.activation_func   = activation_func

            #self.input_layer = nn.Linear(input_dim, hidden_dim)
            #self.output_layer = nn.Linear(hidden_dim, output_dim)

            #Integration settings
            
    def forward(self, t, x): #x shoudl be (num_traj, spatial_dim)
        '''Takes t and x as input, where x is a pytorch tensor with shape: [trajectories, dim].
        
        input x is hidden state. 
        '''
        out = self.network(x) #(num_traj, spatial_dim+drivers)
        return out
    
class driven_neural_ODE(neural_ODE):
    '''
    Class that creates an Neural ODE with known bifurcation parameter.

    The bifurcation parameter is treated as an additional input to the neural network. 
        
    By default the parameter is static, but can be made time varying by setting the drdt attribute to a nonzero value.
    '''
    def __init__(self, input_dim: int, drivers: int, output_dim: int, depth: int, hidden_dim: int, activation_func: nn.Module = nn.Tanh()):
        super().__init__(input_dim = input_dim+drivers, output_dim = output_dim, depth = depth, hidden_dim = hidden_dim, activation_func = activation_func)
        self.num_drivers = drivers
        self.drdt = 0 #rate of change of the driver variable. 
            
    def forward(self, t, x): #x shoudl be (num_traj, spatial_dim)
        '''
        Takes t and x as input, where x is a pytorch tensor with shape: [trajectories, dim+drivers].
        '''
        out = self.network(x) #(num_traj, spatial_dim+drivers) -> (num_traj, output_dim)
        drdt_tensor = torch.ones(out.shape[0], self.num_drivers)*self.drdt # shape (num_traj, drivers)
        final_out = torch.cat((out, drdt_tensor), dim=1) 
        return final_out
    
    
class Trainer:
    def __init__(self, model, data_loader, optimizer, loss_fn, t_eval, dt):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.t_eval = t_eval
        self.dt = dt
        self.loss_list = []
        self.epoch_trained = 0

        pred_x = odeint(func=self.model, y0=data_loader.dataset[:, 0, :], t=self.t_eval, method='rk4', options={'step_size': self.dt})
        loss = self.loss_fn(pred_x[-1, :, :], data_loader.dataset[:, 1, :])
        self.loss_list.append(loss.item())

    def loss(self):
        pred_x = odeint(func=self.model, y0=self.data_loader.dataset[:, 0, :], t=self.t_eval, method='rk4', options={'step_size': self.dt})
        loss = self.loss_fn(pred_x[-1, :, :], self.data_loader.dataset[:, 1, :])
        return loss.item()

    def train(self, num_epochs, t_eval):
        
        for epoch in tqdm(range(num_epochs), desc= "Current Loss: " + str(self.loss_list[-1]) + "  - Epochs: " + str(self.epoch_trained)):
            self.epoch_trained += 1
            epoch_loss = 0.0
            for i, (x_batch) in enumerate(self.data_loader): #x_batch shape (num_traj, 2, dim)

                self.optimizer.zero_grad()
                
                #ODE integration to get predictions + compute loss
                pred_x = odeint(func=self.model, y0=x_batch[:, 0, :], t=self.t_eval, method='rk4', options={'step_size': self.dt})
                loss = self.loss_fn(pred_x[-1, :, :], x_batch[:, 1, :])
                epoch_loss += loss.item()
                
                #Backpropagation and optimization step
                loss.backward()
                self.optimizer.step()

            avg_epoch_loss = epoch_loss / len(self.data_loader)
            self.loss_list.append(avg_epoch_loss)
    


class NODE_Bifurcation(Simple_FeedforwardNN):
    '''
    Class that creates an autonomous Neural ODE with a fixed bifurcation parameter. 

    The bifurcation parameter must be set using the set_bif_param method.
    '''
    def __init__(self, spatial_dim: int, drivers: int, depth: int, width: int, activation_func: nn.Module = nn.Tanh()):
        super().__init__(spatial_dim+drivers, depth, width, spatial_dim, activation_func)
        self.width              = width #width is number of neurons per hidden layer
        self.depth              = depth #Number of internal hidden layers
        self.variables          = spatial_dim
        self.drivers           =  drivers
        self.bif_param          =  None

    def set_bif_params(self, param: torch.Tensor):
        '''
        Method to set the bifurcation parameter.

        Parameter must be a float or a torch pytorch tensor with shape: [1]
        '''
        if not torch.is_tensor(param):
            raise TypeError("Input param must be a torch pytorch tensor")
        else:
            self.bif_param = param.float()

        


    def forward(self, t, x): #x shoudl be (num_traj, spatial_dim)
        '''Takes t and x as input, where x is a pytorch tensor with shape: [trajectories, dim].'''
        
        if self.bif_param is None:
            raise ValueError("Bifurcation parameter not set. Please use the set_bif_param method to set it before calling forward.")
        
        elif not torch.is_tensor(x):
            raise TypeError("Input x must be a torch pytorch tensor with shape: [trajectories, dim]")

        else:
            r_val = self.bif_param
            input = torch.cat((x, r_val), dim=1)
            out = self.network(input) #(num_traj, spatial_dim+drivers)
            return out
        
class normalize_data:
    ''' 
    Class to normalize and denormalize data. 

    Data input shape default is (time, dim).
    '''
    
    def __init__(self, data, axis=0):
        self.data_mean = np.mean(data, axis=axis, keepdims=True)
        self.data_std = np.std(data, axis=axis, keepdims=True)

    def normalize(self, data):
        normalized_data = (data - self.data_mean) / self.data_std
        return normalized_data

    def denormalize(self, normalized_data):
        data = normalized_data * self.data_std + self.data_mean
        return data

#### Batching 
# Want output ot be (time, traj, dim)
def batch(data, t, batch_length, num_batch): # data shape (traj, dim, time) #All traj same length 
    '''
    Generates batches of data for training. 

    Input shape should be (time, traj, dim).
    '''
    data_length = data.shape[0]
    if batch_length >= data_length:
        raise ValueError("Batch length must be less than the length of the data.")
    else:
    #num_traj = data.shape[0]
        #traj_indices = np.random.randint(0, num_traj, batch_size)
        ic_indices = np.random.randint(0, data_length - batch_length, num_batch)

        batch_list = []
        batch_time_list = []

        for i in range(0, num_batch):
            batch_list.append(data[ic_indices[i]:ic_indices[i]+batch_length, :, :].float())
            batch_time_list.append(t[ic_indices[i]:ic_indices[i]+batch_length].float().requires_grad_(True))

        return batch_list, batch_time_list
    
class Piecewise_Auto_NODE(Simple_FeedforwardNN):
    '''
    Class that creates an Neural ODE which is piecewise autonomous on learned time intervals. 

    Parameters:
    ----------
    spatial_dim : int
        Dimension of the spatial variables of the training data. 

    depth : int
        Number of hidden layers in the neural network.

    width : int
        Number of neurons per hidden layer in the neural network.

    time_range : list
        List containing the start and end time of the training data [time_start: int, time_end: int] 
    
    num_breakpoints : int
        Number of breakpoints to use in the piecewise autonomous Neural ODE.
    
    activation_func : torch.nn module
        Activation function to be used in the hidden layers. Default is Tanh.
    
    Attributes:
    ----------
    breakpoints : torch.Tensor
        Tensor containing the location of the temporal breakpoints. Initialized uniformly in the time range.

    break_params : torch.nn.Parameter
    
    '''
    def __init__(self, spatial_dim: int, depth: int, width: int, time_range: list, num_breakpoints: int, activation_func: nn.Module=nn.Tanh()):
        super().__init__(spatial_dim, depth, width, spatial_dim*(num_breakpoints+1), activation_func)

        self.depth              = depth #Number of internal hidden layers
        self.width              = width #Number of neurons per hidden layer

        self.variables          = spatial_dim
        self.num_breakpoint     = num_breakpoints
        self.time_range         = time_range

        #Number of autonomous intervals. 
        self.num_intervals             = (num_breakpoints+1) 
        self.breakpoints        = torch.linspace(time_range[0], time_range[1], num_breakpoints+2) 
        self.break_params       = nn.Parameter(self.breakpoints[1:-1])
        self.k = 1  #steepness parameter for sigmoid function
    
    
    def sigmoid(self, x):
       out = torch.sigmoid(self.k*x)
       return out
    


    def forward(self, t, x):
        '''Takes t and x as input, where x is a pytorch tensor with shape: [trajectories, dim].'''
        num_traj = x.shape[0]
        net_out = self.network(x).reshape(num_traj, self.variables, self.num_intervals) # output in shape (num_traj, spatial_dim*(num_intervals)) -> (num_traj, spatial_dim, num_intervals)

        #Concatenate indicators for each interval, format keeps gradient tracking working. 
        #Want inticator shape (1, spatial_dim, num_intervals)
        #First interval. Shae (spatial_dim, 1)
        i1 = self.sigmoid(self.break_params[0] - t).repeat(self.variables).unsqueeze(1) 
        output = i1

        #Internal Intervals
        for i in range(0, self.break_params.shape[0]-1):

            a = self.sigmoid(t - self.break_params[i])
            b = self.sigmoid(self.break_params[i+1] - t)
            c = a*b 
            
            output = torch.concatenate([output, c.repeat(self.variables).unsqueeze(1)], dim=1) #Like this to keep gradient tracking working
            
        #Final interval 
        #print(output.shape)
        i_final = self.sigmoid(t - self.breakpoints[-1]).repeat(self.variables).unsqueeze(1)
        indicator = torch.cat([output, i_final], dim=1).unsqueeze(0) # shape (1, spatial_dim, num_intervals)
        
        self.breakpoints[1:-1] = self.break_params.detach() #Updating breakpoints attribute to current break_params values.
        net_out2 = net_out*indicator # shape (num_traj, spatial_dim, num_intervals) x (1, spatial_dim, num_intervals) -> (num_traj, spatial_dim, num_intervals)
        net_out_final = torch.sum(net_out2, dim=2) # shape (num_traj, spatial_dim)
        return net_out_final
    
if __name__ == "__main__":
    print('')
    print('Start of File ')
    print('')
    


    a = torch.tensor([0.5]).repeat(3).unsqueeze(0)
    model = Piecewise_Auto_NODE(spatial_dim=3, depth=3, width = 30, time_range=[0,20], num_breakpoints=3, activation_func=nn.Tanh())

    #Saving loss and setting optimiser 
    loss_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    training_data_tensor = torch.ones(100, 2, 3)
    dt = 0.01
    t_eval_tensor = torch.arange(0, 1, dt)#
    

    #Single training loop

    optimizer.zero_grad()
    pred_x = odeint(model, training_data_tensor[0, :, :].float(), t_eval_tensor.float(), method='rk4', options={'step_size': dt})

    loss_fn = nn.L1Loss()
    train_loss = loss_fn(pred_x, training_data_tensor.float())

    train_loss.backward()
    optimizer.step()
    print('')
    print('Total Loss: ', train_loss)
    print('')
    

    print('End of File ')
    print('')

