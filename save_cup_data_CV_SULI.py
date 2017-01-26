# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:09:56 2015

@author: jnewman
"""

from datetime import datetime 
from datetime import timedelta 
import numpy as np


def get_array(column):
    var_temp = np.loadtxt(filename, delimiter=',', usecols=(column,),skiprows=1,dtype=str)
    var = np.zeros(len(var_temp))
    for k in range(len(var_temp)):
        try:
            var[k] = float(var_temp[k])
        except:
            var[k] = np.nan  
    return var    

months = [1]
tower_loc = "B5"

dir_name = '/Users/jnewman/Desktop/Data for Elise/' + tower_loc + ' Data/Processed Python Files/' 
data_dir = '/Users/jnewman/Desktop/Dissertation Data/Chisholm_View_Data/Chisholm_View_OU_LLNL_Shared_Data/Met Tower Data/'


for j in months:
    
    if j == 11:
        filename = data_dir + 'Met ' + tower_loc + ' Data 2013 1107-1130.csv'
    if j == 12:
        filename = data_dir + 'Met ' + tower_loc + ' Data 2013 1201-1231.csv'
    if j == 1:
        filename = data_dir + 'Met ' + tower_loc + ' Data 2014 0101-0114.csv' 
        
    if j == 5 or j==6:
        filename = data_dir + 'Met ' + tower_loc + ' Data 2014 0501-0630.csv' 
        
    timestamp = np.loadtxt(filename, delimiter=',', usecols=(0,),dtype=str, unpack=True,skiprows=1)

    time_datenum = []
    for i in timestamp:
        try:
            time_datenum.append(datetime.strptime(i,"%m/%d/%Y %H:%M")- timedelta(minutes=20)+ timedelta(hours=6))   
        except:
            time_datenum.append(datetime.strptime(i,"%m/%d/%y %H:%M")- timedelta(minutes=20)+ timedelta(hours=6))   
        
    pressure = np.loadtxt(filename, delimiter=',', usecols=(2,),skiprows=1)
    rain_rate = np.loadtxt(filename, delimiter=',', usecols=(8,),skiprows=1)
    RH_76 = np.loadtxt(filename, delimiter=',', usecols=(9,),skiprows=1)
    temp_76 = get_array(13)
    temp_10 = get_array(17)
    wd_76_1 = get_array(25)
    wd_76_2 = get_array(29)
    wd_43 = get_array(33)
    ws_78 = np.loadtxt(filename, delimiter=',', usecols=(37,),skiprows=1)
    ws_78_std_dev = np.loadtxt(filename, delimiter=',', usecols=(40,),skiprows=1)
    ws_80 = np.loadtxt(filename, delimiter=',', usecols=(41,),skiprows=1)
    ws_80_std_dev = np.loadtxt(filename, delimiter=',', usecols=(44,),skiprows=1)
    ws_74 = np.loadtxt(filename, delimiter=',', usecols=(45,),skiprows=1)
    ws_74_std_dev = np.loadtxt(filename, delimiter=',', usecols=(48,),skiprows=1)
    ws_38 = np.loadtxt(filename, delimiter=',', usecols=(49,),skiprows=1)
    ws_38_std_dev = np.loadtxt(filename, delimiter=',', usecols=(52,),skiprows=1)

    time_all = []    
    pressure_all = []
    rain_rate_all = []
    RH_76_all = []
    temp_76_all = []
    temp_10_all = []
    wd_76_1_all = []
    wd_76_2_all = []
    wd_43_all = []
    ws_78_all = []
    ws_78_std_dev_all = []
    ws_80_all = []
    ws_80_std_dev_all = []
    ws_74_all = []
    ws_74_std_dev_all = []
    ws_38_all = []
    ws_38_std_dev_all = []

    for mm in months:
        for dd in range(32):
            for ii in range(len(time_datenum)):
                if time_datenum[ii].month == mm and time_datenum[ii].day == dd:
                    time_all.append(time_datenum[ii])
                    pressure_all.append(pressure[ii])
                    rain_rate_all.append(rain_rate[ii])
                    RH_76_all.append(RH_76[ii])
                    temp_76_all.append(temp_76[ii])
                    temp_10_all.append(temp_10[ii])
                    wd_76_1_all.append(wd_76_1[ii])
                    wd_76_2_all.append(wd_76_2[ii])
                    wd_43_all.append(wd_43[ii])
                    ws_78_all.append(ws_78[ii])
                    ws_78_std_dev_all.append(ws_78_std_dev[ii])
                    ws_80_all.append(ws_80[ii])
                    ws_80_std_dev_all.append(ws_80_std_dev[ii])
                    ws_74_all.append(ws_74[ii])
                    ws_74_std_dev_all.append(ws_74_std_dev[ii])
                    ws_38_all.append(ws_38[ii])
                    ws_38_std_dev_all.append(ws_38_std_dev[ii])
            if len(pressure_all) > 3:   
                filename = dir_name + '2014' + str(mm).zfill(2) + str(dd).zfill(2)
                np.savez(filename,time = time_all,pressure=pressure_all,rain_rate = rain_rate_all,RH_76 = RH_76_all,temp_76 = temp_76_all,temp_10=temp_10_all,wd_76_1 = wd_76_1_all,\
                wd_76_2 = wd_76_2_all,wd_43 = wd_43_all,ws_78 = ws_78_all,ws_78_std_dev = ws_78_std_dev_all,ws_80 = ws_80_all,ws_80_std_dev = ws_80_std_dev_all,\
                ws_74 = ws_74_all,ws_74_std_dev = ws_74_std_dev_all,ws_38 = ws_38_all,ws_38_std_dev = ws_38_std_dev_all)
            time_all = []        
            pressure_all = []
            rain_rate_all = []
            RH_76_all = []
            temp_76_all = []
            temp_10_all = []
            wd_76_1_all = []
            wd_76_2_all = []
            wd_43_all = []
            ws_78_all = []
            ws_78_std_dev_all = []
            ws_80_all = []
            ws_80_std_dev_all = []
            ws_74_all = []
            ws_74_std_dev_all = []
            ws_38_all = []
            ws_38_std_dev_all = []
    

                                             