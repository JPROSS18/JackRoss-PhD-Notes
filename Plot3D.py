#Plotting 
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as la
from scipy import stats
from scipy import spatial as sp
from scipy import interpolate as ip
import pandas as pd


class plot3D: #Must plot in same block 
    def __init__(self, cols=1, rows=1, figsize = (10, 10)):
        self.figRow = rows #Rows
        self.figCol = cols #Cols
        self.figSize = figsize #figsize 

        #Creating figure 
        self.fig = plt.figure(figsize=self.figSize)
        self.pltColours = ["royalblue", "green", "red", "orchid"]
    
    def plot3D(self, U, plotNum, title="Plot", axlab = ['x', 'y', 'z'], title_font_size=10, axes_fontsize = [10, 10, 10], tick_label_size=10, tick_label_color = 'k'):
        self.ax = self.fig.add_subplot(self.figRow, self.figCol, plotNum, projection='3d') #setting which plot 

        #Setting plot attributes
        self.ax.set_title(title, fontsize=title_font_size)
        self.ax.set_xlabel(axlab[0], fontsize = axes_fontsize[0])
        self.ax.set_ylabel(axlab[1], fontsize = axes_fontsize[1])
        self.ax.set_zlabel(axlab[2], fontsize = axes_fontsize[2])
        self.ax.tick_params(direction='out', length=6, width=2, colors=tick_label_color,
                                grid_color=tick_label_color, grid_alpha=1, labelsize=tick_label_size)
        #Plotting 
        plt.plot(U[0, :], U[1, :], U[2, :], color=self.pltColours[(plotNum-1) % len(self.pltColours)])
    
    def plot2D(self, x, y, plotNum, title="Plot", axlab = ['x', 'y'], title_font_size=10, axes_fontsize = [10, 10], tick_label_size=0, tick_label_color='k', label="1"):
        
        ax = self.fig.add_subplot(self.figRow, self.figCol, plotNum) #setting which plot 
        
        #Setting plot attributes
        ax.set_title(title, fontsize=title_font_size)
        ax.set_xlabel(axlab[0], fontsize=axes_fontsize[0])
        ax.set_ylabel(axlab[1], fontsize=axes_fontsize[1])
        ax.tick_params(direction='out', length=6, width=2, colors=tick_label_color,
                                grid_color=tick_label_color, grid_alpha=1, labelsize=tick_label_size)
    
        #Plotting 
        plt.plot(x, y, color=self.pltColours[(plotNum-1) % len(self.pltColours)], label=label)

    def add_plot2D(self, x, y, title="Plot", axlab = ['x', 'y'], title_font_size=10, axes_fontsize = [10, 10], tick_label_size=0, tick_label_color='k', label="1"):
        
        #Setting plot attributes
        #ax.set_title(title, fontsize=title_font_size)
        #ax.set_xlabel(axlab[0], fontsize=axes_fontsize[0])
        #ax.set_ylabel(axlab[1], fontsize=axes_fontsize[1])
        #ax.tick_params(direction='out', length=6, width=2, colors=tick_label_color,
         #                       grid_color=tick_label_color, grid_alpha=1, labelsize=tick_label_size)
    
        #Plotting 
        plt.plot(x, y, color=self.pltColours[(plotNum-1) % len(self.pltColours)], label=label)

    def plot_recurrence(self, Matrix, plotNum, title = "Heat Map", axlab = ['x', 'y'] ): #Heat Map / Recurrence Plot
        ax = self.fig.add_subplot(self.figRow, self.figCol, plotNum)

        ax.set_title(title)
        ax.set_xlabel(axlab[0])
        ax.set_ylabel(axlab[1])
        
        plt.imshow(Matrix, cmap='binary', origin='lower')
