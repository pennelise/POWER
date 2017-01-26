# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 12:56:57 2016

@author: jnewman
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


def bin_mean_func(data_min,data_max,bin_width,x_data,y_data):
    """Find avgs for each bin and return bin center and bin mean to plot. 
    
    Arguments:
    data_min -- the minumum value included in the x data
    data max -- the maximum values included in the x data
    bin_width -- the width of bins desired
    x_data -- your independent variable
    y_data -- your dependent variable
    """
    n_bins = (data_max-data_min)/bin_width

    bin_mean = []
    bin_center = []
    
    for i in np.arange(data_min,data_max,bin_width):
        mask = [x_data > i, x_data <= i + bin_width,~np.isnan(x_data),~np.isnan(y_data)]
        mask = reduce(np.logical_and, mask)
        data_temp = y_data[mask]
        if len(data_temp) > len(y_data)/(2*n_bins):
            bin_mean.append(np.mean(data_temp))
        else:
            bin_mean.append(np.nan)
        bin_center.append(i + bin_width/2.)    
            
    return np.array(bin_center),np.array(bin_mean)     

    
def linear_regression_intercept(x,y):
    """Find the slope, intercept and R^2 value for a scatterplot. 
    
    Arguments:
    x -- independent variable
    y -- dependent variable
    """
    masks = [~np.isnan(x),~np.isnan(y)]
    total_mask = reduce(np.logical_and, masks)
    x = x[total_mask]
    y = y[total_mask]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    y_pred = m*x + c
    total = 0
    for i in range(len(y)):
        total+=(y_pred[i]-np.mean(y))**2
    total = 0    
    for i in y:
        total+=(i-np.mean(y))**2
    SST2 = LA.norm((y-np.mean(y)))**2
    total = 0    
    for i in range(len(y)):
        total+=(y_pred[i]-y[i])**2
    SSE2 = LA.norm((y_pred-y))**2
    R_squared = 1-SSE2/SST2
    return m,c,R_squared    
    
def bin_mean_plot(data_min,data_max,bin_width,x_data,y_data,var_name):
    """Plot data and return statistical data. 
    
    Arguments: 
    data_min -- the minumum value included in the x data
    data max -- the maximum values included in the x data
    bin_width -- the width of bins desired
    x_data -- your independent variable
    y_data -- your dependent variable
    var_name -- the name of the variable you will test for sensitivity (string)
    
    Returns: 
    A list containing: slope, sensitivity (slope * std), r^2, r*sensitivity
    """
    #If sensitivity > 0.5 or r*sensitivitiy > 0.1, it is sensitive.    
    
    
    #from IEC_sensitivity_functions import bin_mean_func
    [bin_center,bin_mean] = bin_mean_func(data_min,data_max,bin_width,x_data,y_data)
    #from stats_functions import linear_regression_intercept
    [m,c,r_squared] = linear_regression_intercept(bin_center,bin_mean)
    x_vals = np.arange(data_min,data_max,bin_width)
    y_vals = x_vals*m + c
    
    plt.figure()
    plt.scatter(x_data,y_data,s=5,color="blue")
    plt.scatter(bin_center,bin_mean,s=20,color="red")
    plt.plot(x_vals,y_vals,color="red",label = 'y =  %(number)0.2f*x + %(number2)0.2f. R$^2$ = %(number3)0.2f'%\
        {"number": m,"number2": c,"number3": r_squared} )
    plt.xlim([data_min,data_max])
    plt.ylim([-50,50])    
    plt.legend(loc='best')
    plt.xlabel(var_name)
    plt.ylabel('TI % Difference')
    var_stats = [m,m*np.std(x_data[~np.isnan(x_data)]),r_squared,np.sqrt(r_squared)*m*np.std(x_data[~np.isnan(x_data)])]
    return var_stats