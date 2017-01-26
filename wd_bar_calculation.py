# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:18:13 2016

@author: jnewman
"""

import numpy as np

ws = [5,5.3,5.2,4.8,4.5]
wd = [180,190,185,175,170]
#ws: 10-min. mean horizontal wind speed
#wd: 10-min. mean wind direction

u = np.sin(np.radians(wd) - np.pi)*ws #10-min. mean u values
v = np.cos(np.radians(wd) - np.pi)*ws #10-min. mean v values

u_bar = np.mean(u) #Average u value for hour-long time period
v_bar = np.mean(v) #Average v value for hour-long time period
wd_bar = (180./np.pi)*(np.arctan2(u_bar,v_bar) + np.pi)
