# Upper Confidence Bound (UCB)
# Solving the Multi-Armed Bandit problem

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math

# Step 1
N = 10000 # for number of users
d = 10 # for number of ads in the dataset
ads_selected = [] # the list for our selected ads
numbers_of_selections = [0] * d # initial conditions
sums_of_rewards = [0] * d
total_reward = 0

# Step 2
for n in range(0, N): # Looping the algorithm
    ad = 0 # Starting with the first ad
    max_upper_bound = 0 # to be updated during iterations
    for i in range(0, d): # Implementing the algorithm
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
            
# Step 3             
        else:
            upper_bound = 1e400 # using a super high value to obtain higher Upper bound
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
