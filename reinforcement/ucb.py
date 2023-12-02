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

N = 10000
d = 10
add_selected = []
number_of_selections = [0] * d
sum_of_rewards = [0] * d
total_reward = 0
for i in range(0, N):
    ad = 0
    max_ucb = 0
    for j in range(0, d):
        if (number_of_selections[j] > 0):
            avarage_reward = sum_of_rewards[j] / number_of_selections[j]
            confidence_interval = math.sqrt(
                3 / 2 * math.log(i+1) / number_of_selections[j])
            upper_bound = avarage_reward + confidence_interval
        else:
            upper_bound = 1e400
        if (upper_bound > max_ucb):
            max_ucb = upper_bound
            ad = j
            add_selected.append(ad)
            number_of_selections[ad] += 1
            reward = df.values[i, ad]
            sum_of_rewards[ad] += reward
            total_reward += reward
ads_df = pd.DataFrame({'Ad_Selected': add_selected})
plt.figure(figsize=(8, 6), dpi=150)
sns.countplot(data=ads_df, x='Ad_Selected', palette='viridis')
plt.title('Number of Times Each Ad was Selected')
plt.xlabel('Ad')
plt.ylabel('Frequency')
plt.show()
