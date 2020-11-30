# -*- coding: utf-8 -*-
"""
Created on Tue Jul  11 10:41:16 2020

@author: Sumeet
"""
#%reset -f
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df =pd.read_csv('Mall_Customers.csv')

x=df.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmean=KMeans(n_clusters=i,init='k-means++',n_init=10,random_state=42)
    kmean.fit(x)
    wcss.append(kmean.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('Number of cluster')
plt.ylabel('wcss')
plt.show()

kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
y_means=kmeans.fit_predict(x)

plt.scatter(x[y_means==0,0],x[y_means==0,1],s=50,c='red',label='Cluster 1')
plt.scatter(x[y_means==1,0],x[y_means==1,1],s=50,c='b',label='Cluster 2')
plt.scatter(x[y_means==2,0],x[y_means==2,1],s=50,c='g',label='Cluster 3')
plt.scatter(x[y_means==3,0],x[y_means==3,1],s=50,c='c',label='Cluster 4')
plt.scatter(x[y_means==4,0],x[y_means==4,1],s=50,c='m',label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=50,c='y',label='centroids')
plt.title('cluster of client')
plt.xlabel('Anual income')
plt.ylabel('spending score(1-100)`')
plt.legend()
plt.show()
