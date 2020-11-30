# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('Salary_Data.csv')
x=df.iloc[:,:-1].values
y=df.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_predict=regressor.predict(x_test)

plt.scatter(x_train,y_train,c='y')
plt.plot(x_train,regressor.predict(x_train),c='r')
plt.title('Salary vs experience')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

plt.scatter(x_test,y_test,s=5,c='y')
plt.plot(x_train,regressor.predict(x_train),c='r')
plt.title('Salary vs experience')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()