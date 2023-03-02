# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 14:41:04 2022

@author: rjara
"""
#%%
import pandas as pd


#%%
pop= [30.55, 2.77, 39.21]
countries= ["afghanistan", "albania", "algeria"]

world= {"afghanistan":30.55, "albania": 2.77, "algeria":39.21}

world["albania"]
world["algeria"]

world.keys()

world["sealand"]=0.000027 # meter en el diccionario

world["sealand"]=0.000028 #actualizo valor

del(world["sealand"]) #sacar key

#%% DIctionary of dictionaries
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }


# Print out the capital of France
print(europe["france"]["capital"])

# Create sub-dictionary data
data= {"capital":"rome", "population": 59.83}

# Add data to europe under key 'italy'
europe["italy"]= data

# Print europe
print(europe)

eur= pd.DataFrame(europe)

eur
#%%

# Pre-defined lists
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]

# Import pandas as pd
import pandas as pd

# Create dictionary my_dict with three key:value pairs: my_dict
my_dict= {'country': names, 'drives_right' : dr, 'cars_per_cap': cpc}

# Build a DataFrame cars from my_dict: cars
cars= pd.DataFrame(my_dict)

# Print cars
print(cars)

# Definition of row_labels
row_labels = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars
cars.index=row_labels

# Print cars again
print(cars)

#%% 

# Import the cars.csv data: cars
cars= pd.read_csv("cars.csv", index_col=0) #Acordarse del path (donde esta el archivo en la compu)













