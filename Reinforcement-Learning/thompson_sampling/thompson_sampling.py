import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('Ads_CTR_Optimisation.csv')

import random
N=10000
d=10
ads_selected=[]
number_of_reward_1=[0]*d
number_of_reward_0=[0]*d
total_reward=0
for n in range(N):
    ad=0
    max_random=0
    for i in range(10):
        random_beta=random.betavariate(number_of_reward_1[i]+1,number_of_reward_0[i]+1)
        if random_beta>max_random:
            max_random=random_beta
            ad=i
    ads_selected.append(ad)
    reward=df.values[n,ad]
    if reward==1:
        number_of_reward_1[ad]+=1
    else:
        number_of_reward_0[ad]+=1
    total_reward+=reward
plt.hist(ads_selected)