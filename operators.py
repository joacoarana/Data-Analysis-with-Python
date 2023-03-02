# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:45:46 2022

@author: rjara
"""
#%%
import numpy as np
import pandas as pd


#%% BOOLEAN OPERATORS
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than 18.5 or smaller than 10
print(np.logical_or(18.5<my_house, my_house<10))

# Both my_house and your_house smaller than 11
print(np.logical_and(my_house<11, your_house<11))


#%%

#Subset es para ver las filas donde algo es true

# Create car_maniac: observations that have a cars_per_cap over 500
cpc= cars['cars_per_cap']
many_cars = cpc>500


# Print car_maniac
car_maniac=cars[many_cars]
print(car_maniac)

#%%

# Create medium: observations with cars_per_cap between 100 and 500
cpc = cars['cars_per_cap']
xd = np.logical_and(cpc>100, cpc<500)
medium= cars[xd]


# Print medium
print(medium)