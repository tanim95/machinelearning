# the dataset used here represents 10 ads showing to online user and 0,1
# means wheather user clicked it or not and based on this result we need to decide which ads to show to a certain user and we need to
# figure it so fast as showing ads is costly!
# Note: each add has a certain conversion rate or fixed distribution like mult-arm bandit problem slot machine

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# importing data
df = pd.read_csv('./data/Ads_CTR_Optimisation.csv')
print(df.head())
# implementing upper confidence bound(UCB) algorithm


N = 10000  # Number of rounds
d = 10  # Number of ads

add_selected = []  # List to store selected ads
number_of_selections = np.zeros(d)  # Num of selections for each ad
sum_of_rewards = np.zeros(d)  # Sum of rewards for each ad
total_reward = 0  # Total reward obtained

for n in range(1, N + 1):
    ad = 0
    max_upper_bound = 0

    for i in range(0, d):
        if number_of_selections[i] > 0:
            average_reward = sum_of_rewards[i] / number_of_selections[i]
            confidence_interval = math.sqrt(
                2 * math.log(n) / number_of_selections[i])
            upper_bound = average_reward + confidence_interval
        else:
            upper_bound = 1e400  # Large upper bound for unexplored arms

        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i  # Updating the selected ad based on the highest UCB value

    add_selected.append(ad)  # Record the selected ad
    number_of_selections[ad] += 1  # Increment the count for the selected ad
    reward = df.values[n - 1, ad]
    sum_of_rewards[ad] += reward

ads_df = pd.DataFrame({'Ad_Selected': add_selected})
plt.figure(figsize=(8, 6), dpi=150)
sns.countplot(data=ads_df, x='Ad_Selected', palette='viridis')
plt.title('Number of Times Each Ad was Selected')
plt.xlabel('Ad')
plt.ylabel('Frequency')
plt.show()
