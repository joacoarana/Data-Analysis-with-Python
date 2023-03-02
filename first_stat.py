# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 21:57:21 2022

@author: rjara
"""

#%%
# Import numpy as np
import numpy as np
import matplotlib.pyplot as plt

# Set the seed
np.random.seed(123)

# Generate and print random float
print(np.random.rand())

# Use randint() to simulate a dice
print(np.random.randint(1,7))

# Use randint() again
print(np.random.randint(1,7))

#%%

all_walks = []

# Simulate random walk 10 times
all_walks = []
for i in range(250) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)

        # Implement clumsiness
        if np.random.rand()<=0.001 :
            step = 0

        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))
np_aw_t
plt.plot(np_aw_t)
plt.show()

# Select last row from np_aw_t: ends
ends = np_aw_t[-1,:]

# Plot histogram of ends, display plot
plt.hist(ends)
plt.show()

np.mean(ends>=60)