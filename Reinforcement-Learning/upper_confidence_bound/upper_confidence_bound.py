# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

df=pd.read_csv('Ads_CTR_Optimisation.csv')
N=10000
d=10
number_of_selection=[0]*d
sums_of_reward=[0]*d
ads_selected=[]
total_reward=0
for n in range(N):
    ad=0
    max_upper_bound=0
    for i in range(d):
        if (number_of_selection[i]>0):
            avg_reward=sums_of_reward[i]/number_of_selection[i]
            delta_i=math.sqrt(3/2*math.log(n+1)/number_of_selection[i])
            upper_b=avg_reward+delta_i
        else:
            upper_b=1e400  
        if upper_b>max_upper_bound:
            max_upper_bound=upper_b
            ad=i
    ads_selected.append(ad)
    number_of_selection[ad]+=1
    reward=df.values[n,ad]
    sums_of_reward[ad]+=reward
    total_reward+=reward
plt.hist(ads_selected)
plt.show()