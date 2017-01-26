# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:21:55 2016

@author: epenn

Funtions for evaluating wind resource.
"""

import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
from pytz import timezone
from windrose import WindroseAxes
#import collections
import glob
import os
import re
from numpy import linalg as LA
from matplotlib import cm
from matplotlib.mlab import griddata

#Print an error message.
def print_err():
    """Print \"Error!! Your inputs do not have equal lengths.\""""
    print("Error!! Your inputs do not have equal lengths.")

"""
Read in Files
-------------
"""  
def read_in(directory, start_date, end_date, var):
    """Read in a set of .pyz files with filenames which are dates.
    
    Arguments:
    directory -- the path to the directory where files are stored
    start_date -- the first date to average over
    end_sate -- the last date to average over
    var -- a list of variables to read in (default: read in all variables)
    """
    #start = float(start_date.strftime('%Y%m%d'))
    #end = float(end_date.strftime('%Y%m%d'))
     
    #fplen = len(directory)
    
    #Get filenames within the date range. 
    #filenames = [t for t in glob.glob(directory+'*') if float(t[fplen:fplen+8]) >= start and float(t[fplen:fplen+8]) <= end]

    filenames = []
    time_step = datetime.timedelta(days = 1)
    while start_date <= end_date:
        filenames.append(os.path.normpath('%s/%s' % (directory, start_date.strftime('%Y%m%d.npz'))))
        start_date += time_step    
    filenames = np.array(filenames)

    #print filenames

    #Read each variable into the dictionary.
    #d = collections.defaultdict(list)
    d = {}
    for v in var:
        d[v] = []
    
    for n in filenames:
        if os.path.exists(n): #Check that the file for this date exists.
            f = np.load(n)
            for v in var:
                d[v].extend(f[v])
            f.close()
        else: #If it does not, append a 144 nons in each key.
            for v in var: #fix_time will fix this later.
                d[v].append(np.nan)
            
    #Convert each list to an np.array.
    for key in d:
        d[key] = np.array(d[key], dtype = type(d[key][0]))
        
#    final_dictionary = {}
#    for key in d:
#        print(key)
#        temp_array = np.array([])
#        for i in d[key]:
#            temp_array = np.append(temp_array,i)
#        final_dictionary[key] = temp_array
        
    return d

def save_figure(name):
    """Save a figure under the filename \'name.png,\' then close and clear the figure.
    Saves in the current directory of this file.
    """
    directory = os.path.normpath('%s/Figures/' % (os.getcwd()))
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Directory %s created." % (directory))
    figure_string = os.path.normpath('%s/%s.png' % (directory,name))
    plt.savefig(figure_string)
    plt.close()
    plt.clf()   
    
    
#[new_folder_name, date, old_dictionary, step_in_minutes, turbine_names] = ['SCADA',start_time,winter_met_file,10, turbine_names]    
    
def fix_time(new_folder_name, date, old_dictionary,step_in_minutes, turbine_names = None):
    """Fill in missing dates with np.nan and save data in a new folder with
    dates as titles of the filenames. 
    
    Arguments:
    new_folder_name -- the name of the folder where corrected files are stored
    dictionary -- a dictionary containing all data you want to store
    time -- the array in the dictionary which contains the times
    step_in_minutes -- timestep between each data point recorded (usu. 10mins)
    """
        
    #Create a folder for formatted data, if it does not already exist.
    formatted_data_dir = os.path.normpath('%s/FormattedData/' % (os.getcwd()))
    if not os.path.exists(formatted_data_dir):
        os.makedirs(formatted_data_dir)
        print('Directory %s created.' % (formatted_data_dir))

    new_folder = os.path.normpath('%s/%s/' % (formatted_data_dir,new_folder_name))
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
        print('Directory %s created.' % (new_folder))
        
    date_str = date.strftime('%Y%m%d')        
        
    old_time = np.array(old_dictionary['time'])
    
    #Create an array with complete times for every 10 minutes.
    start_time = datetime.datetime(date.year,date.month,date.day,0,0,0) #Start at midnight
    end_time = datetime.datetime(date.year, date.month, date.day,23,50,00) #End at 11:50pm
    new_time = []
    time_step = datetime.timedelta(minutes=step_in_minutes)
    while start_time <= end_time:
        new_time.append(start_time)
        start_time += time_step        
    new_time = np.array(new_time)

    #Create a list of times which spans the range of TIME
    if turbine_names is None:     
        #Create a new dictionary to hold the full new_time (rather than incomplete old_time)
        new_dictionary = {}
        for key in old_dictionary:
            new_dictionary[key] = np.zeros([len(new_time)],type(old_dictionary[key][0]))
        
        #This breaks if there is more than one of the same timestamp. 
        for nt in range(len(new_time)):
            #Loop over old_time until you reach a value = new_time
            print(new_time[nt])
            ot = 0
            while ot < len(old_time):
                if old_time[ot] == new_time[nt]:
                    for key in old_dictionary: #Loop over keys in old_dictionary. 
                        if key != 'time': #Do not create an incomplete 'time' array. This is useless. 
                            new_dictionary[key][nt] = old_dictionary[key][ot]
                    ot +=1
                    break #If a value is found, stop searching.
                else: #If no value matches the new time array, fill with nan.
                    for key in old_dictionary: #Loop over keys in old dictionary.
                        if key != 'time': #Do not create an incomplete 'time' array. This is useless.
                            new_dictionary[key][nt] = np.nan
                    ot += 1

        #Set your time entry equal to the new, fully-populated time array.
        new_dictionary['time'] = np.array(new_time)

        #Save your new dictionary as an npz after deleting any file of the same name.
        file_path = os.path.normpath("%s/%s.npz" % (new_folder,date_str))
        if os.path.exists(file_path): #Remove the file if it already exists.
            os.remove(file_path)
        np.savez(file_path,**new_dictionary)
                
    else: #If turbine_names are input. 
        all_turbine_names = old_dictionary['turbine_name'] #Create an array listing each turbine once.

        new_dictionary = {}
        for key in old_dictionary:
                new_dictionary[key] = [None]*(len(new_time)*len(turbine_names))
        
        for nt in range(len(new_time)):
            for tb in range(len(turbine_names)):
                turb_time = np.array([])
                
                #If none of the turbines are valid, turb_time == [ nan].
                if ~np.any([all_turbine_names == turbine_names[tb]]):
                    turb_time = np.append(turb_time,np.nan)
                else:
                    turb_time = old_time[all_turbine_names == turbine_names[tb]]
                                
                tt = 0
                while tt < len(turb_time):
                    if turb_time[tt] == new_time[nt]:
                        for key in old_dictionary:
                            if key == 'time':
                                new_dictionary[key][nt*len(turbine_names) + tb] = new_time[nt]
                            elif key == 'turbine_name':
                                new_dictionary[key][nt*len(turbine_names) + tb] = turbine_names[tb]
                            else:
                                new_dictionary[key][nt*len(turbine_names) + tb] = old_dictionary[key][all_turbine_names == turbine_names[tb]][tt]
                                #print old_dictionary[key][all_turbine_names == unique_turb[tb]][tt]
                                #print "Line 168: %s" % (new_dictionary[key][nt*len(unique_turb) +tb])
                        tt += 1
                        break
                    else: #If the times do not match.
                        for key in old_dictionary:
                            if key == 'time':
                                new_dictionary[key][nt*len(turbine_names) + tb] = new_time[nt]
                            elif key == 'turbine_name':
                                new_dictionary[key][nt*len(turbine_names) + tb] = turbine_names[tb]
                            else:
                                new_dictionary[key][nt*len(turbine_names) + tb] = np.nan
                        tt += 1
                    #print "Line 177: %s" % (new_dictionary[key][0:90])
                        
        #Convert dictionary to np.array type.
                        
        final_dictionary = {}
        for key in new_dictionary:
            temp_array = np.array([])
            for i in new_dictionary[key]:
                temp_array = np.append(temp_array,i)
            final_dictionary[key] = temp_array
                
        #Save your new dictionary as an npz after deleting any file of the same name.
        file_path = os.path.normpath("%s/%s.npz" % (new_folder,date_str))
        if os.path.exists(file_path): #Remove the file if it already exists.
            os.remove(file_path)
        np.savez(file_path,**final_dictionary)
        
        
        
def save_total_power(data,times,SCADA_faults,filename):
    total_power = np.array([])
    new_times = np.array([])
    percent_active = np.array([])    
    for time in np.unique(times):
        state_fault = SCADA_faults[times == time]
        fault_mask = [state_fault == 2,state_fault == 1]
        fault_mask = reduce(np.logical_or,fault_mask)
        
        total_power = np.append(total_power,np.sum(data[times == time]))
        new_times = np.append(new_times,time)
        percent_active = np.append(percent_active,float(np.sum(fault_mask))/float(len(fault_mask)))
        
        
    total_dictionary = {}
    total_dictionary['total_power'] = total_power
    total_dictionary['time'] = new_times
    total_dictionary['percent_active'] = percent_active
        
    file_path = os.path.normpath('%s/FormattedData/%s' % (os.getcwd(),filename))
    np.savez(file_path,**total_dictionary)


"""
Unit Conversions
----------------
""" 
def convert_to_central(times_UTC):
    """Convert a list of naive UTC datetimes to Central Time."""
    times_central = []
    for time in times_UTC:
        #Tell the time variable what timezone it is in
        time_utc = time.replace(tzinfo=timezone('UTC'))
        #Convert from UTC to Central (presumably including DST)
        time_central = time_utc.astimezone(timezone('US/Central'))
        times_central.append(time_central)
    return times_central
   
def convert_to_uv(wind_directions,wind_speeds):
    """Convert arrays of wind directions and wind speeds to arrays of u,v vectors.
    
    Arguments:
    wind_directions -- np array of wind directions in meteorological degrees
    wind_speeds -- np array of corresponding wind speeds
    """
    uv = [] 
    u = - abs(wind_speeds) * np.sin(np.deg2rad(wind_directions)) #Negative because u,v shows the direction wind is moving
    v = - abs(wind_speeds) * np.cos(np.deg2rad(wind_directions)) #rather than direction wind comes from (as in met_deg)
    uv.append(u)
    uv.append(v)
    return np.array(uv) #[[list of u values],[list of v values]]

def convert_to_degspd(u,v):
    """Convert arrays of u,v vectors to arrays of direction (in met.deg) and speed
        
    Arguments:
    u - np array of the u (east) part of the vector
    v - np array of the v (north) part of the vector
    """
    dir_spd = [] 
    wind_direction = (np.rad2deg(np.arctan2(-u,-v))) % 360.0
    wind_speed = np.sqrt(u ** 2 + v ** 2)
    dir_spd.append(wind_direction)
    dir_spd.append(wind_speed)
    return np.array(dir_spd) #[[list of wind dirs], [list of wind spds]]
    
def correct_density(wind_speed,pressure,temperature,relative_humidity):
    """Calculates air density based on IEC 61400-12-1, 2016 edition.
    
    Arguments:
    pressure -- list of 10-min avg pressures. Units: Pa
    temperture -- list of 10-min avg temperatures. Units: K
    relative_humidity -- list of 10-min avg RH. Units: Fraction between 0 and 1.
    """
    gas_const_air = 287.05 # J/kgK; gas constant of dry air
    gas_const_water = 461.5 # J/kgK; gas constant of water vapor
    vapor_pressure = 0.0000205 * np.exp(0.0631846 * temperature) # Pa
    
    density = (1 / temperature) * ((pressure / gas_const_air) - ( (relative_humidity * vapor_pressure) * ((1.0 / gas_const_air) - (1.0 / gas_const_water)) ))    
    density_sea_level = 1.225 # kg/m^3

    new_ws = wind_speed * (density / density_sea_level) ** (1.0/3.0)
    return new_ws

def grid(x, y, z, resX=100, resY=100):
    """"Convert 3 column data to matplotlib grid
    Credit: Elyase of Stackoverflow."""
    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi,interp='linear')
    X, Y = np.meshgrid(xi, yi)
    return X, Y, Z

"""
Filtering Data
--------------
"""

def filter_icing(data,temp,RH,precip):
    """Remove data points where the instrument experienced icing.
    Icing occurs when temp is below 0.0 degC AND relative humidity is above 80.0
    OR when temp is below 0.0 degC AND precipitation occurs

    Arguments:
    data -- a np array of data you wish to filter
    temp -- a np array of corresponding temperatures for these data
    RH -- a np array of corresponding relative humidity values for these data
    precip -- a np array of corresponding precipitation values for these data
    """
    if len(temp) == len(RH) == len(data):
        deiced_data = []
        for (temp_10m,RH_10m,precip_10m,data_10m) in zip(temp,RH,precip,data):
            if np.isnan(temp_10m):
                deiced_data.append(np.nan)
            elif temp_10m <= 0.0 and RH_10m >= 80.0:
                deiced_data.append(np.nan)
            elif temp_10m <= 0.0 and precip_10m != 0.0:
                deiced_data.append(np.nan)
            else:
                deiced_data.append(data_10m)
        return np.array(deiced_data)
    else:
        print_err()
        
def filter_streaking(data):
    destreaked_data = np.empty(len(data))
    i = 0
    while i < len(data)-3:
        if data[i] == data[i+1] and data[i] == data[i+2]: #Compare 3 values at one time.
            destreaked_data[i] = np.nan
            destreaked_data[i+1] = np.nan
            destreaked_data[i+2] = np.nan
            i += 3 #Move forward 3 steps if they are all equal
            while data[i] == data[i-1] and i < len(data)-3: #Move forward by 1's afterwards until the value changes.
                destreaked_data[i] = np.nan
                i += 1
        else: #If three in a row not equal, move by only one step. 
            destreaked_data[i] = data[i]
            i += 1
    while i >= len(data)-3 and i <= len(data)-1: #For values near the end, the procedure changes to keep within the indices of the array.
        if data[i] == data[i-1] and data[i] == data[i-2]:
            destreaked_data[i] = np.nan
            i += 1
        elif i != len(data)-1 and data[i] == data[i-1] and data[i] == data[i+1]:
                destreaked_data[i] = np.nan
                i += 1
        else:
            destreaked_data[i] = data[i]
            i += 1
    return destreaked_data
    
def filter_obstacles(data,wind_directions,direction_filtered,sector_filtered):
    """Remove data in the specified sector. 
    
    Arguments: 
    data -- a np array of the data you wish to filter
    temp -- a np array of corresponding wind directions for these data
    direction_filtered -- the direction (in meteorological degrees) you wish to remove
    sector_filtered -- the size of sector (in degrees) you wish to remove
    """
    min_filtered_angle = (float(direction_filtered) - float(sector_filtered) / 2.0) % 360.0
    max_filtered_angle = (float(direction_filtered) + float(sector_filtered) / 2.0) % 360.0
    filtered_data = []
    if len(data) == len(wind_directions):           #Check that both lists have the same length.
        if max_filtered_angle < min_filtered_angle: #If sector crosses 0deg, this is true. 
            for (datapoint,direction) in zip(data,wind_directions):    #(e.g. if excluded range is 350deg - 10 deg).
                if direction >= min_filtered_angle or direction <= max_filtered_angle:
                    filtered_data.append(np.nan)
                else:
                    filtered_data.append(datapoint)
        else:                                       #If sector does not cross 0 deg, this is true. 
            for (datapoint,direction) in zip(data,wind_directions):
                if direction >= min_filtered_angle and direction <= max_filtered_angle:
                    filtered_data.append(np.nan)
                else:
                    filtered_data.append(datapoint)
        return np.array(filtered_data)
    else:
        print_err()
        
def filter_temp(temp_array):
    """Filter temp if it is < -50 or > 130 deg C (or F).
    
    Arguments:
    temp_array -- a np array of temperatures to filter.
    """
    filtered_temp = np.array([])
    for temp in temp_array:
        if temp < -50.0 or temp > 130.0:
            filtered_temp = np.append(filtered_temp, np.nan)
        else: 
            filtered_temp = np.append(filtered_temp, temp)
            
    return filtered_temp
            
def sub_data(data,filtered_sub,wind_directions,direction_filtered,sector_filtered):
    """Remove data in a specified sector and replace with data for another met tower.
    
    Arguments:
    data -- orig. data filtered for icing but NOT for direction.
    filtered_sub -- fully filtered data (filtered for wind direction as well!!)
    direction_filtered -- the direction (in meteorological degrees) you wish to remove
    sector_filtered -- the size of sector (in degrees) you wish to remove
    """
    min_filtered_angle = (float(direction_filtered) - float(sector_filtered) / 2.0) % 360.0
    max_filtered_angle = (float(direction_filtered) + float(sector_filtered) / 2.0) % 360.0
    filled_data = np.array([])
    if max_filtered_angle < min_filtered_angle: #If sector crosses 0deg, this is true. 
        for i in range(len(data)):    #(e.g. if excluded range is 350deg - 10 deg).
            if wind_directions[i] >= min_filtered_angle or wind_directions[i] <= max_filtered_angle:
                filled_data = np.append(filled_data,filtered_sub[i]) #Replace filtered values with the substitute met tower values
            else:
                filled_data = np.append(filled_data,data[i])
    else:                                       #If sector does not cross 0 deg, this is true. 
        for i in range(len(data)):
            if wind_directions[i] >= min_filtered_angle and wind_directions[i] <= max_filtered_angle:
                filled_data = np.append(filled_data,filtered_sub[i]) #Replace filtered values with the substitute met tower values
            else:
                filled_data = np.append(filled_data,data[i])
    return np.array(filled_data)

"""
Finding Hourly Wind Speed and Direction
---------------------------------------
"""

def hourly_wind_speed(wind_speeds, times):
    """Average wind speed over hours and return a 1x24 numpy array.
    
    Arguments:
    wind_speeds -- a np array of all wind speeds
    times -- a np array of all times with indexes corresponding to wind_speeds
    """
    avg_hourly_ws = []
    new_times = []
    hours = np.array([t.hour for t in times]) #Make an array of just the hours.
    for i in range(24):
        avg_hourly_ws.append(np.nanmean(wind_speeds[hours == i]))
        new_times.append(i)
    return np.array(new_times), np.array(avg_hourly_ws) #Return the wind speeds and their corresponding times as a NumPy array

#Gets average wind dir for each hour of the day (returns 24h averaged over multiple days)
def hourly_wind_direction(wind_directions,times):
    """Average wind direction over hours and return a 1x24 numpy array.
    
    Arguments:
    wind_directions -- a np array of all wind speeds
    times -- a np array of all times with indexes corresponding to wind_speeds
    """
    wind_speeds = np.ones(len(wind_directions)) #Assign wind speeds unit values.
    uv = convert_to_uv(wind_directions,wind_speeds) #Conver to u,v unit vectors
    u, v = np.array(uv[0]), np.array(uv[1])
    avg_hourly_u, avg_hourly_v, new_times = [], [], []
    avg_hourly_wd = []
    hours = np.array([t.hour for t in times])
    for i in range(24):
        avg_hourly_u.append(np.nanmean(u[hours == i])) #Average over u and v
        avg_hourly_v.append(np.nanmean(v[hours == i]))
        new_times.append(i)
    avg_hourly_wd = convert_to_degspd(np.array(avg_hourly_u), np.array(avg_hourly_v))[0] #Convert u,v to wind direction. Make sure inputs are arrays.
    return np.array(new_times), np.array(avg_hourly_wd) #Return the wind directions and their corresponding times as a NumPy array

#def hourly_wind_speed(wind_speeds, times):
#    """Find average wind speed for each hour for which there is data
#    
#    Arguments:
#    wind_speeds -- np array of wind speeds for each hour
#    times -- np array of datetimes corresponding to each wind_speed
#    """
#    hourly_wind_speeds = []
#    new_times = []
#    #Concatenate yyyymmddhh for each measurement from times. Intentionally drop mins.
#    individual_hours = np.array([datetime.datetime(t.year, t.month, t.day, t.hour, tzinfo = t.tzinfo) for t in times])
#    #Make a list of unique hours listed in the time variable
#    unique_hours = np.unique(individual_hours)
#    for hour in unique_hours:
#        hourly_wind_speeds.append(np.nanmean(wind_speeds[individual_hours == hour])) #Append the average of values at that unique hour
#        new_times.append(hour) #Append the corresponding unique hour to new_times
#    return [new_times,hourly_wind_speeds]
#    
#def hourly_wind_direction(wind_directions,wind_speeds,times):
#    """ Gets average wdir for each hour measured (returns a value for each hour for which there is data) """
#    uv = convert_to_uv(wind_directions,wind_speeds)
#    u = uv[0]
#    v = uv[1]
#    hourly_u = []
#    hourly_v = []
#    hourly_wd = []
#    new_times = []
#    #Concatenate yyyymmddhh for each measurement from times. Intentionally dropped mins.
#    individual_hours = np.array([datetime.datetime(t.year, t.month, t.day, t.hour, tzinfo = t.tzinfo) for t in times])
#    unique_hours = np.unique(individual_hours)
#    #Make a list of unique hours listed in the time variable
#    for hour in unique_hours:
#        hourly_u.append(np.nanmean(u[individual_hours == hour])) #Append the average of u,v values at that unique hour
#        hourly_v.append(np.nanmean(v[individual_hours == hour]))
#        new_times.append(hour)
#    hourly_wd = convert_to_degspd(np.array(hourly_u),np.array(hourly_v))[0]  #Convert from u, to degrees and speed. Make sure inputs are arrays
#    return np.array([new_times,hourly_wd]) #Return the wind directions and their corresponding times. 
    
"""
Creating a Wind Rose
--------------------
author: Lionel Roubeyrie
"""
def new_axes():
    """Create a set of wind rose axes.""" 
    fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='w')
    rect = [0.1, 0.1, 0.8, 0.8]
    ax = WindroseAxes(fig, rect, axisbg='w')
    fig.add_axes(ax)
    return ax

def set_legend(ax):
    """Create a legend for the wind rose."""
    l = ax.legend(borderaxespad=-0.10)
    plt.setp(l.get_texts(), fontsize=8)
    
def plot_wind_rose(wind_directions,wind_speeds,bins):
    """Plot a wind rose.
    
    Arguments:
    wind_directions -- a np array of wind directions filtered for icing
    wind_speeds -- a np array of filtered wind speed corresponding to wind_directions
    """
    mask = [~np.isnan(wind_directions),~np.isnan(wind_speeds)]
    mask = reduce(np.logical_and,mask)
    ax = new_axes()
    ax.bar(wind_directions[mask], wind_speeds[mask],normed=True, opening=0.8, edgecolor='white', bins = bins)
    set_legend(ax)
    
def plot_power_rose(wind_directions,power,num_wd_bins):
    """Plot a power rose. Kind of a hacked wind rose. 
    
    Arguments:
    wind_directions -- a np array of wind directions filtered for icing
    power -- a np array of percent power production corresponding to wind_directions
    num_wd_bins -- the number of wind direction bins to include on the rose.
    """
    dir_bins = np.array(np.linspace(0.0,360.0 - 360.0 / num_wd_bins,num_wd_bins))
    #Find the total amount of power produced in each sector.
    dir_power = np.array([np.nansum(filter_obstacles(power,wind_directions,(wd + 180.0) % 360.0, 360 - 360/float(num_wd_bins))) for wd in dir_bins])
    dir_power = np.round(dir_power * 100.0 / np.nansum(dir_power), decimals=0)   #Normalize it and round to nearest int. 
    
    proportional_wd = np.array([])
    for i in range(len(dir_power)):
        for n in range(int(dir_power[i])): #Loop as many times as the percent of power produced in this sector.
                proportional_wd = np.append(proportional_wd,dir_bins[i]) #i.e., if 50% of power comes from the south, append 50 instances of 180.0 degrees.
    ones = np.ones(len(proportional_wd))
    
    ax = new_axes()
    ax.bar(proportional_wd, ones,normed=False, opening=0.8, edgecolor='white', bins = [0.0,100.], cmap=cm.RdGy)
    set_legend(ax)

    
"""
Sensitivity Functions
---------------------

All sensitivity code from Jennifer Newman
"""

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
    
def calc_wind_shear(time,heights,wind_speeds):
    """Plot data and return statistical data. 
    
    Arguments:
    time -- array of times. Same length as wind_speeds sub-arrays
    wind_speeds -- array of at least 2 arrays of wind speeds. 
        Each sub-array corresponds to a different measurement height.
    heights -- List of heights corresponding to each array of wind_speeds
    """
        
    
    wind_shear = np.array([])
    for t in time:
        #Extract just the wind speeds that correspond to this time.
        this_wind_speed = np.array([])
        for i in range(len(heights)):
            this_wind_speed = np.append(this_wind_speed, wind_speeds[i][time == t])
        #Find alpha
        if np.sum(~np.isnan(this_wind_speed)) >= 2: #Alpha only exists if there are at least 2 values.
            wind_shear = np.append(wind_shear, linear_regression_intercept(np.log(heights), np.log(this_wind_speed))[0])
        else:
            wind_shear = np.append(wind_shear, np.nan)
        
        del this_wind_speed
        
    return wind_shear