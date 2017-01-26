# -*- coding: utf-8 -*-b
"""
Spyder Editor

Read in met tower and SCADA data.
"""

import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
import windeval as we
from sklearn.cross_validation import train_test_split
from machine_learning_example import machine_learning_RF
from power_curve_query_func import power_curve_query as pc
from matplotlib import cm

"""
BEGIN USER INPUT
"""

file_dir = 'C:\\Users\\epenn\\Documents\\Sample Data\\Sample Data\\'

#Read in all SCADA data
SCADA_dir = 'C:\\Users\\epenn\\Documents\\wind_site_data\\SCADA\\'
met_south_dir = 'C:\\Users\\epenn\\Documents\\wind_site_data\\met_south\\'
met_north_dir = 'C:\\Users\\epenn\\Documents\\wind_site_data\\met_north\\'

max_power = 1000 #maximum power produced in the entire plant
nturbines = 10 #number of turbines in your plant

SCADA_keys = ['operating_state', 'nacelle_ws', 'pitch_southngle', 'power', 'nacelle_ws_std', 'rotor_speed', 'nacelle_position', 'time', 'state_fault', 'turbine_name']
met_keys = ['ws_1_std_dev', 'ws_1', 'temp_76', 'ws_2', 'rain_rate', 'ws_2_std_dev', 'wd_2', 'pressure', 'wd_3', 'wd_1', 'time', 'ws_4', 'temp_1', 'RH_1', 'ws_4_std_dev', 'ws_3', 'ws_3_std_dev']
turbine_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] #Input the names of your turbines in your documents.
 

bins = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5])
power_curve = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21.])

#start_time = datetime.datetime(2013,11,7)
#end_time = datetime.datetime(2014,1,31)

#start_time = datetime.datetime(2014,5,1)
#end_time = datetime.datetime(2014,6,30)

start_time = datetime.datetime(2013,11,7)
end_time = datetime.datetime(2014,6,30)

fix_time_series = False

"""
END USER INPUT
"""



"""
THIS PART READS IN ALL FILES
"""

#THIS PART IS WHERE I FIX THE TIME SERIES AND CREATE NEW DATASETS FROM THE ORIGINALS.
if fix_time_series:
    
    #SCADA
    time_range = []
    
    time_step = datetime.timedelta(days = 1)
    entry = start_time
    while entry <= end_time:
        time_range.append(entry)
        entry += time_step        
    time_range = np.array(time_range)
    
    for t in time_range:
        file_name = we.read_in(SCADA_dir,t,t,SCADA_keys)
        we.fix_time("SCADA",t,file_name,10, turbine_names = turbine_names)
        
    #south met tower
    time_range = []
    
    time_step = datetime.timedelta(days = 1)
    entry = start_time
    while entry <= end_time:
        time_range.append(entry)
        entry += time_step        
    time_range = np.array(time_range)
    
    for t in time_range:
        file_name = we.read_in(met_south_dir,t,t,met_keys)
        we.fix_time("met_south",t,file_name,10, turbine_names = turbine_names)
       
    #north met tower
    time_range = []
    
    time_step = datetime.timedelta(days = 1)
    entry = start_time
    while entry <= end_time:
        time_range.append(entry)
        entry += time_step        
    time_range = np.array(time_range)
    
    for t in time_range:
        file_name = we.read_in(met_north_dir,t,t,met_keys)
        we.fix_time("met_north",t,file_name,10, turbine_names = turbine_names)


fixed_SCADA_file = "C:\\Users\\epenn\\Documents\\FormattedData\\SCADA\\"
fixed_south_file = 'C:\\Users\\epenn\\Documents\\FormattedData\\met_south\\'
fixed_north_file = 'C:\\Users\\epenn\\Documents\\FormattedData\\met_north\\'
winter_file = 'C:\\Users\\epenn\\Documents\\FormattedData\\winter.npz' 
summer_file = 'C:\\Users\\epenn\\Documents\\FormattedData\\summer.npz' 

print("SCADA")
SCADA = we.read_in(fixed_SCADA_file,start_time,end_time,SCADA_keys)

winter_total = np.load(winter_file)
summer_total = np.load(summer_file)


print("met_south")
met_south = we.read_in(fixed_south_file,start_time,end_time,met_keys)
print("met_north")
met_north = we.read_in(fixed_north_file,start_time,end_time,met_keys)
#we.save_total_power(SCADA['power'],SCADA['time'],SCADA['state_fault'],'summer')                                                                                                                         



"""
This part cleans the data.
"""
print('Clean')

filtered_total = np.zeros(len(winter_total['total_power']))
filtered_total[:] = np.nan
filtered_total[winter_total['total_power'] >= 0.98] = winter_total['total_power'][winter_total['total_power'] >= 0.98]

clean_south_temp = we.filter_temp(met_south['temp_76'])
temp_south_K = clean_south_temp + 273.15
pressure_south_Pa = met_south['pressure'] * 100.0
clean_south_wd = we.filter_icing(met_south['wd_2'],clean_south_temp,met_south['RH_1'],met_south['rain_rate'])
#clean_south_ws =  we.filter_obstacles(we.filter_obstacles(we.filter_icing(we.filter_streaking(met_south['ws_4']), clean_south_temp, met_south['RH_1'], met_south['rain_rate']),clean_south_wd,285.0,74.0),clean_south_wd,40.0,74.0)
deiced_south_ws =  we.correct_density(we.filter_icing(we.filter_streaking(met_south['ws_4']), clean_south_temp, met_south['RH_1'], met_south['rain_rate']),pressure_south_Pa,temp_south_K,met_south['RH_1']/100.0)
deiced_south_ws_1 = we.correct_density(we.filter_icing(we.filter_streaking(met_south['ws_1']), clean_south_temp, met_south['RH_1'], met_south['rain_rate']),pressure_south_Pa,temp_south_K,met_south['RH_1']/100.0)
deiced_south_std_dev = we.filter_icing(we.filter_streaking(met_south['ws_4_std_dev']), clean_south_temp, met_south['RH_1'], met_south['rain_rate'])

clean_north_temp = we.filter_temp(met_south['temp_76'])
temp_north_K = clean_north_temp + 273.15
pressure_north_Pa = met_north['pressure'] * 100.0
clean_north_wd = we.filter_icing(met_north['wd_2'],clean_south_temp,met_south['RH_1'],met_south['rain_rate'])
clean_north_ws = we.correct_density(we.filter_obstacles(we.filter_obstacles(we.filter_icing(we.filter_streaking(met_north['ws_4']), clean_north_temp, met_north['RH_1'], met_south['rain_rate']),clean_north_wd,164.0,60.0),clean_north_wd, 241.0,70.0),pressure_north_Pa,temp_north_K,met_north['RH_1']/100.0)
clean_north_ws_1 = we.correct_density(we.filter_obstacles(we.filter_obstacles(we.filter_icing(we.filter_streaking(met_north['ws_1']), clean_north_temp, met_north['RH_1'], met_south['rain_rate']),clean_north_wd,164.0,60.0),clean_north_wd, 241.0,70.0),pressure_north_Pa,temp_north_K,met_north['RH_1']/100.0)
clean_north_std_dev = we.filter_obstacles(we.filter_obstacles(we.filter_icing(we.filter_streaking(met_north['ws_4_std_dev']), clean_north_temp, met_north['RH_1'], met_north['rain_rate']),clean_north_wd,164.0,60.0),clean_north_wd,164.0,60.0)

#Replace waked values from met tower south with waked values from met tower north.
met_ws = we.sub_data(we.sub_data(deiced_south_ws, clean_north_ws, clean_south_wd, 285.0, 74.0), clean_north_ws, clean_south_wd, 40.0, 74.0)
met_std_dev = we.sub_data(we.sub_data(deiced_south_std_dev, clean_north_std_dev, clean_south_wd, 285.0, 74.0), clean_north_std_dev, clean_south_wd, 40.0, 74.0)

Ti = np.array(met_std_dev) / np.array(met_ws) * 100.0

#Calculate shear
all_clean_ws = np.zeros(2)
all_clean_ws_1 = we.sub_data(we.sub_data(deiced_south_ws_1, clean_north_ws_1, clean_south_wd, 285.0, 74.0), clean_north_ws_1, clean_south_wd, 40.0, 74.0)
all_clean_ws_1[all_clean_ws_1 <= 1.0] = np.nan #Filter values below the threshold of the instrument.
all_clean_ws_4 = met_ws
wind_shear = we.calc_wind_shear(met_south['time'], np.array([38.0, 80.0]), [all_clean_ws_1,all_clean_ws_4])



"""
Uncomment these to access different features.
"""

#Machine Learning
"""
inputs_orig = np.transpose([met_ws,wind_shear,Ti,clean_south_wd])
targets_orig = np.transpose(filtered_total)

indices = np.arange(np.array(targets_orig).shape[0])  
#Splits into training set and testing set
x_train, x_test, y_train, y_test,idx_train,idx_test = train_test_split(inputs_orig,targets_orig,indices,test_size=0.25,random_state= 42)
        
y_train = y_train.ravel()
y_test = y_test.ravel() 

y_test2, power_pred = machine_learning_RF(x_train,y_train,x_test,y_test)

plt.figure(num=15)
plt.scatter(y_test2 / max_power / nturbines, power_pred / max_power / nturbines, s = 4)
regline_ml = we.linear_regression_intercept(y_test2 / max_power / nturbines, power_pred / max_power / nturbines)
plt.plot(y_test2 / max_power / nturbines, y_test2 / max_power / nturbines, color ='green', label = 'x=y')
plt.plot(y_test2 / max_power / nturbines, y_test2 / max_power / nturbines * regline_ml[0] + regline_ml[1], color ='red', label = 'line of regression')
plt.xlim(0.0,1.05)
plt.ylim(0.0,1.05)
plt.title("Machine Learning vs. Actual Values")
axes = plt.gca()
xmin, xmax = axes.get_xlim()
ymin, ymax = axes.get_ylim()
plt.text(xmax-0.35*(xmax-xmin), ymin+0.05*(ymax-ymin),  \
    "Filtered for icing and wind direction. \nm = %s \nc = %s \nr^2 = %s" %  \
    ("{0:.2f}".format(regline_ml[0]), "{0:.2f}".format(regline_ml[1]), "{0:.2f}".format(regline_ml[2])), fontsize = 8)
plt.legend(loc=0)

#Predict using Manufacturer's Power Curve
mpc_pred = pc(x_test[:,0],x_test[:,2],'normal_TI')
plt.figure(num=16)
plt.scatter(y_test / max_power / nturbines, mpc_pred / max_power, s = 4)
regline_mpc = we.linear_regression_intercept(y_test / max_power / nturbines, mpc_pred / max_power)
plt.plot(y_test / max_power / nturbines, y_test / max_power / nturbines, color ='green', label = 'x=y')
plt.plot(y_test / max_power / nturbines, y_test / max_power / nturbines * regline_mpc[0] + regline_mpc[1], color ='red', label = 'line of regression')
plt.xlim(0.0,1.05)
plt.ylim(0.0,1.05)
plt.title("Manufacturer's Power Curve vs. Actual Values")
axes = plt.gca()
xmin, xmax = axes.get_xlim()
ymin, ymax = axes.get_ylim()
plt.text(xmax-0.35*(xmax-xmin), ymin+0.05*(ymax-ymin),  \
    "Filtered for icing and wind direction. \nm = %s \nc = %s \nr^2 = %s" %  \
    ("{0:.2f}".format(regline_mpc[0]), "{0:.2f}".format(regline_mpc[1]), "{0:.2f}".format(regline_mpc[2])), fontsize = 8)
plt.legend(loc=2)
"""


#Create a power curve.
"""
print('plot')
plt.figure(num=42)
mask = [summer_total['percent_southctive'] > 0.98]
clean_total = np.empty(len(summer_total['total_power']))
clean_total[:] = np.nan
clean_total[mask] = summer_total['total_power'][mask] / 1000.0 / 235.2 # Normalized
plt.scatter(met_ws, clean_total)

plt.xlim([0.0,20])
plt.ylim([0.0,1.05])
plt.title('Summer Power Curve')
plt.ylabel('Total Plant Production (normalized)')
plt.xlabel('Wind Resource (m/s)')
"""


#Color-code a power curve.
    #Wind direction
    #Day/Night
"""
current_power = winter_total
print('plot')
mask = [current_power['percent_southctive'] > 0.98]
clean_total = np.empty(len(current_power['total_power']))
clean_total[:] = np.nan
clean_total[mask] = current_power['total_power'][mask] / 1000.0 / 235.2 # Normalized

plt.figure(num=134)

#South 
mask2 = [clean_south_wd <= 225, clean_south_wd >= 135]
mask2 = reduce(np.logical_southnd,mask2)
plt.scatter(met_ws[mask2], clean_total[mask2], color = 'green', alpha = 0.4, label = 'South')

#East 
mask2 = [clean_south_wd <= 135, clean_south_wd >= 45]
mask2 = reduce(np.logical_southnd,mask2)
plt.scatter(met_ws[mask2], clean_total[mask2], color = 'blue', alpha = 0.4, label = 'East')

#North 
mask2 = [clean_south_wd <= 45, clean_south_wd >= 315]
mask2 = reduce(np.logical_or,mask2)
plt.scatter(met_ws[mask2], clean_total[mask2], color = 'red', alpha = 0.4, label = 'North')

#West 
mask2 = [clean_south_wd <= 315, clean_south_wd >= 225]
mask2 = reduce(np.logical_southnd,mask2)
plt.scatter(met_ws[mask2], clean_total[mask2], color = 'yellow', alpha = 0.4, label = 'West')

#        
#    del SCADA_power
plt.xlim([0.0,20])
plt.ylim([0.0,1.05])
plt.title('Summer Power Curve')
plt.ylabel('Total Plant Production (normalized)')
plt.xlabel('Wind Resource (m/s)')
plt.legend(loc=2)

    #Binned Turbulence
shear_mask = [wind_shear>=0.2,wind_shear<=0.5]
shear_mask = reduce(np.logical_southnd,shear_mask)
tot = np.zeros(len(clean_total))
tot[:] = np.nan
tot[shear_mask] = clean_total[shear_mask]

ws = np.zeros(len(met_ws))
ws[:] = np.nan
ws[shear_mask] = met_ws[shear_mask]

plt.figure(num=23)
mask = [Ti>=0., Ti<5.]
mask = reduce(np.logical_southnd,mask)
plt.scatter(ws[mask],tot[mask], c='blue', alpha=0.3)

mask = [Ti>=5.,Ti<10.]
mask = reduce(np.logical_southnd,mask)
plt.scatter(ws[mask],tot[mask], c='green', alpha=0.3)

mask = [Ti>=10.,Ti<15.]
mask = reduce(np.logical_southnd,mask)
plt.scatter(ws[mask],tot[mask], c='yellow', alpha=0.3)

mask = [Ti>=15.]
plt.scatter(ws[mask],tot[mask], c='red', alpha=0.3)

    #Turbulence
#Make a plot with vertical colorbar
fig, ax = plt.subplots()
#mask_shear = [wind_shear <= 0.3, wind_shear>= 0.2]
#mask_shear = reduce(np.logical_southnd,mask_shear)
plot = ax.scatter(met_ws, clean_total, c=wind_shear, cmap=cm.hot_r, alpha = 0.3)
cbar = fig.colorbar(plot, ax=ax, orientation='vertical')
plot.autoscale()
plot.set_clim(0.0,0.75)  
#plt.xlim([0.0,20])
plt.ylim([0.0,1.05])
cbar.set_label('Shear exponent')

    #Day vs Night
plt.figure(num=92)
hours = np.array([t.hour for t in met_south['time']])

#Night
mask_time = [hours < 7, hours > 17]
mask_time = reduce(np.logical_or,mask_time)
plt.scatter(met_ws[mask_time],clean_total[mask_time],c='blue',alpha=0.3)

#Day
mask_time = [hours > 7, hours < 17]
mask_time = reduce(np.logical_southnd,mask_time)
plt.scatter(met_ws[mask_time],clean_total[mask_time],c='orange',alpha=0.3)
"""


#Create unfiltered power curve
"""
unfiltered_total = np.array([])
for time in np.unique(met_south['time']):
    unfiltered_total = np.append(unfiltered_total,np.sum(SCADA['power'][SCADA['time'] == time]))
 
plt.figure(num=999)
plt.scatter(met_south['ws_4'], unfiltered_total / max_power / nturbines, c = '#660520')

plt.xlim([0.0,20])
plt.ylim([0.0,1.05])
plt.title('Unfiltered Winter Power Curve')
plt.ylabel('Total Plant Production (normalized)')
plt.xlabel('Wind Resource from south Met Tower (m/s)')
"""


#Plot a wind rose!
"""
bins = [0.,3.,6.,9.,12.,15.]
we.plot_wind_rose(clean_south_wd,deiced_south_ws,bins)
we.save_figure('wind_rose_winter')
"""


#Make a power rose!
"""
we.plot_power_rose(clean_south_wd,winter_total['total_power'] / max_power / nturbines * 100.0,16)
we.save_figure('power_rose_winter')
"""


#Make a wind speed histogram!
"""
plt.figure(num=45)

ws_northins = np.arange(0,20,0.5)
mask = ~np.isnan(met_ws)

plt.hist(met_ws[mask],normed=1,bins=ws_northins)
plt.xlabel("wind speed (m/s)")
plt.ylabel("frequency (%)")
plt.title('Summer Wind Speed Frequency')
"""


#Plot hourly wind speed
"""
plt.figure(num=2)
ws_time_38, ws_1 = we.hourly_wind_speed(all_clean_ws_1,met_south['time'])
ws_time_80, ws_4 = we.hourly_wind_speed(all_clean_ws_4,met_south['time'])


plt.plot(ws_time_80,ws_4,label='wind speed 80m')
plt.plot(ws_time_38,ws_1,label='wind speed 38m')
plt.title('Winter Wind Velocity over 24h')
plt.xlim(0,23)
plt.ylim(0,9.3)
plt.xlabel('local time (hr)')
plt.ylabel('wind speed (m/s)')
plt.legend(loc=4)
we.save_figure('hourly_wind_speed')
"""


#Test Sensitivity
"""
difference = np.empty(len(clean_south_ws))
difference[:] = np.nan

for ws in bins:
    mask = [clean_south_ws > ws - 0.25, clean_south_ws < ws + 0.25]
    mask = reduce(np.logical_southnd, mask)
    
    difference[mask] = power_curve[bins == ws] * nturbines / 1000 - clean_total[mask]
    
#x = Ti
x = wind_shear_south
y = difference
stats = we.bin_mean_plot(np.nanmin(x),np.nanmax(x),0.05,x,y,'Wind Shear Exponent')
plt.title("sensitivity: %s" % (stats[1]))
axes = plt.gca()
xmin, xmax = axes.get_xlim()
ymin, ymax = axes.get_ylim()
plt.text(xmax-0.35*(xmax-xmin), ymin+0.05*(ymax-ymin), "Filtered for icing and wind direction. \n>98% of turbines active.", fontsize = 8)
"""


#Create a resource map
"""
ll_filename = 'C:\\Users\\epenn\\Documents\\latlonturb.csv'

ll_turbine_name = np.loadtxt(ll_filename, delimiter=',', usecols=(0,),dtype=str, unpack=True,skiprows=1)
lat = np.loadtxt(ll_filename, delimiter=',', usecols=(1,),dtype=float, unpack=True,skiprows=1)
lon = np.loadtxt(ll_filename, delimiter=',', usecols=(2,),dtype=float, unpack=True,skiprows=1)

rotor_d = 82.5 #rotor diameter in m
rated_power = 1000 #in kW

lat_dist = 111.2*1000 #Distance betwen lines of longitude. This is constant with lon. In m. 
lon_dist = 89.50573266967802*1000 #Distance between lines of longitude at mean latitude. In m.

#Normalized to units of rotor diameter.
lat_norm = (lat - min(lat)) * lat_dist / rotor_d
lon_norm = (lon - min(lon)) * lon_dist / rotor_d

#Convert all np.nans into 0's for averaging... CF doesn't care WHY a turbine isn't producing power, just that it isn't.
SCADA['power'][np.isnan(SCADA['power'])] = 0.0

avg_cf = np.array([])
for turb in ll_turbine_name: 
    avg_cf = np.append(avg_cf, (np.mean(SCADA['power'][SCADA['turbine_name'] == turb]) / max_power) * 100)
    
plt.figure(num = 15)
mask = [avg_cf <= 44.]
plt.scatter(lon_norm[mask],lat_norm[mask], color = 'purple', label='CF <= 44%')

mask = [avg_cf > 44., avg_cf <= 45.]
mask = reduce(np.logical_southnd,mask)
plt.scatter(lon_norm[mask],lat_norm[mask], color = 'blue', label='44% < CF <= 45%')

mask = [avg_cf > 45., avg_cf <= 46.]
mask = reduce(np.logical_southnd,mask)
plt.scatter(lon_norm[mask],lat_norm[mask], color = 'green', label='45% < CF <= 46%')

mask = [avg_cf > 46., avg_cf <= 47.]
mask = reduce(np.logical_southnd,mask)
plt.scatter(lon_norm[mask],lat_norm[mask], color = 'orange', label='46% < CF <= 47%')

mask = [avg_cf > 47.5]
plt.scatter(lon_norm[mask],lat_norm[mask], color = 'red', label='47.5% < CF')


plt.gca().set_southspect('equal',adjustable='box')
plt.title('Southern Plains Wind Farm - Summer')
plt.legend(loc=3)
"""


#Create a (directional) resource map
"""
#directions = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
directions = [0.0, 90.0, 180.0, 270.0]
ll_filename = 'C:\\Users\\epenn\\Documents\\latlonturb.csv'

ll_turbine_name = np.loadtxt(ll_filename, delimiter=',', usecols=(0,),dtype=str, unpack=True,skiprows=1)
lat = np.loadtxt(ll_filename, delimiter=',', usecols=(1,),dtype=float, unpack=True,skiprows=1)
lon = np.loadtxt(ll_filename, delimiter=',', usecols=(2,),dtype=float, unpack=True,skiprows=1)
elevation = np.loadtxt(ll_filename, delimiter=',', usecols=(3,),dtype=float, unpack=True,skiprows=1)

for wd in directions:
    rotor_d = 82.5 #rotor diameter in m
    rated_power = 1000 #in kW
    
    lat_dist = 111.2*1000 #Distance between lines of longitude. This is constant with lon. In m. 
    lon_dist = 89.50573266967802*1000 #Distance between lines of longitude at mean latitude. In m.
    
    #Normalized to units of rotor diameter.
    lat_norm = (lat - min(lat)) * lat_dist / rotor_d
    lon_norm = (lon - max(lon)) * lon_dist / rotor_d
    
    #Convert all np.nans into 0's for averaging... CF doesn't care WHY a turbine isn't producing power, just that it isn't.
    SCADA['power'][np.isnan(SCADA['power'])] = 0.0
    
    avg_cf = np.array([])
    for turb in ll_turbine_name: 
        #Filter all but the 90 degrees surrounding wd.
        filtered_power = we.filter_obstacles(SCADA['power'][SCADA['turbine_name'] == turb], clean_south_wd, (wd + 180.0) % 360, 360.0 - (360.0 / float(len(directions))))
        avg_cf = np.append(avg_cf, (np.nanmean(filtered_power / max_power) * 100))
        
    #Make a plot with vertical colorbar
    fig, ax = plt.subplots()
    X, Y, Z = we.grid(lon_norm, lat_norm, elevation)
    contour = plt.contourf(X,Y,Z, cmap = cm.Greys, alpha = 0.5)
    #plt.clabel(contour)
    plot = ax.scatter(lon_norm, lat_norm, c=avg_cf, cmap=cm.CMRmap_r, s = 64)
    cbar = fig.colorbar(plot, ax=ax, orientation='horizontal')
    plot.autoscale()
    plot.set_clim(20.0,50.0)  
    cbar.set_label('Capacity Factor (%)')
    
    #plt.xlim([-10.0,310])
    plt.gca().set_southspect('equal',adjustable='box')
    plt.title('Winter - %s Degrees' % (wd))
"""


#Plot with OS filtered out        
"""      
clean_south_wd = we.filter_icing(met_south['wd_2'],clean_south_temp,met_south['RH_1'],met_south['rain_rate'])
print('Plot2')
plt.figure(num=2)
total_power = np.array([])
for time in met_south['time']:
    #SCADA_fault = SCADA['state_fault'][SCADA['time'] == time]
    SCADA_fault = SCADA['state_fault'][SCADA['time'] == time]
    met_ws = clean_south_ws[met_south['time'] == time]
    total_power = np.sum(SCADA['power'][SCADA['time'] == time])
    
    fault_mask = [SCADA_fault == 2,SCADA_fault ==1]
    fault_mask = reduce(np.logical_or,fault_mask)
    
    if float(np.sum(fault_mask))/float(len(fault_mask)) >= 0.98:
        plt.scatter(met_ws,total_power)

plt.title('Wind Plant Power Curve - A')
plt.xlabel('wind speed (m/s)')
plt.ylabel('power (kW)')
"""
