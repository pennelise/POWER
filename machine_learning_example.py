# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:20:19 2016

@author: jnewman
"""
import numpy as np
from sklearn.cross_validation import train_test_split

def machine_learning_RF(x_train,y_train,x_test,y_test):
    import numpy as np
    mask = []
    
    #Gets rid of NaNs
    for i in range(np.shape(x_train)[1]):
        mask.append(~np.isnan(x_train[:,i]))
    mask.append(~np.isnan(np.transpose(y_train)))  
    mask = np.transpose(reduce(np.logical_and, mask))
    mask = mask.reshape(len(mask),)
    
    inputs = x_train[mask,:]
    targets = y_train[mask]
    
    mask2 = []
    for i in range(np.shape(x_test)[1]):
        mask2.append(~np.isnan(x_test[:,i]))  
    mask2 = np.transpose(reduce(np.logical_and, mask2))
    inputs_test = x_test[mask2,:]
    #End getting rid of NaNs
    
    #Sets up forest
    #n-estimators is how many "trees" (samples) you will take
    from sklearn.ensemble import RandomForestRegressor
    rfc_new = RandomForestRegressor(n_estimators=100,random_state=42,max_features=2)
    #Training
    rfc_new = rfc_new.fit(inputs,targets)
    #Predicting
    predicted_y = rfc_new.predict(inputs_test)
    print rfc_new.feature_importances_    
    return y_test[mask2], predicted_y 
        

shear_exp = [-0.1,0.3,0.4,-0.2]
TI_vals = [35,12,20,25]
power = [1500,1200,1000,850]

inputs_orig = np.transpose([shear_exp,TI_vals])
targets_orig = np.transpose(power)

indices = np.arange(np.array(targets_orig).shape[0])  
#Splits into training set and testing set
x_train, x_test, y_train, y_test,idx_train,idx_test = train_test_split(inputs_orig,targets_orig,indices,test_size=0.25,random_state= 42)
        
y_train = y_train.ravel()
y_test = y_test.ravel() 

power_pred = machine_learning_RF(x_train,y_train,x_test,y_test)



