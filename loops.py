# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 19:53:14 2022

@author: rjara
"""

#%%
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Change for loop to use enumerate() and update print()
for index,a in enumerate(areas) :
    print('room '+ str(index)+ ':' + str(a))
    
#%% diccionarios

europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }
          
# Iterate over europe

for country, capital in europe.items():
    print(f'the capital of {country} is {capital}') 


#%% arrays 1D y 2D

# For loop over np_height 1D
for a in np_height:
    print(f'{a} inches')

# For loop over np_baseball 2D
for b in np.nditer(np_baseball):
    print(b)
    
    
#%% Pandas
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Iterate over rows of cars
for a,b in cars.iterrows():
    print(a) #LABEL
    print(b) # ROW

#%%Agregar columna

# Code for loop that adds COUNTRY column
for a,b in cars.iterrows():
    cars.loc[a,['COUNTRY']]= b['country'].upper()

# o tambien

cars["COUNTRY"] = cars["country"].apply(str.upper)








