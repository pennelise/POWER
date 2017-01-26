# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 10:15:18 2016

@author: jnewman
"""


def power_curve_query(ws,TI,opt="normal_TI"):
    import numpy as np
    hub_height_ws = np.arange(3,13.5,0.5)
    
    power_normal_TI = np.array([0,20,63,116,177,248,331,428,540,667,812,972,1141,1299,1448,1561,1633,1661,1677,1678,1680])
    power_low_TI = np.array([0,18,61,114,174,244,325,421,532,657,801,961,1134,1304,1463,1585,1654,1675,1680,1680,1680])
    power_high_TI = np.array([0,24,68,123,185,258,344,446,562,693,841,994,1148,1287,1419,1519,1589,1637,1665,1679,1680])
    
    if "var_TI" not in opt:    
        if "normal_TI" in opt:
            power = power_normal_TI
        if "low_TI" in opt:
            power = power_low_TI
        if "high_TI" in opt:
            power = power_high_TI    
            
        power_interp = np.interp(ws, hub_height_ws, power)
    else:
        from power_curve_query_func import power_curve_var_TI 
        power_interp = power_curve_var_TI(ws,TI)
        
    return power_interp    
        
        
def power_curve_var_TI(ws,TI):

    import numpy as np
    hub_height_ws = np.arange(3,13.5,0.5)
    
    power_normal_TI = np.array([0,20,63,116,177,248,331,428,540,667,812,972,1141,1299,1448,1561,1633,1661,1677,1678,1680])
    power_low_TI = np.array([0,18,61,114,174,244,325,421,532,657,801,961,1134,1304,1463,1585,1654,1675,1680,1680,1680])
    power_high_TI = np.array([0,24,68,123,185,258,344,446,562,693,841,994,1148,1287,1419,1519,1589,1637,1665,1679,1680])  

    power_interp = np.zeros(len(ws))
    power_interp[:] = np.nan
    index = 0
    for i,j in zip(ws,TI):
        if j < 10:
            power_interp[index] = np.interp(i, hub_height_ws, power_low_TI)
        if j >= 10 and j < 15:   
            power_interp[index] = np.interp(i, hub_height_ws, power_normal_TI)
        if j >= 15 and j < 20:
            power_interp[index] = np.interp(i, hub_height_ws, power_high_TI)  
        index += 1    
    return power_interp        
