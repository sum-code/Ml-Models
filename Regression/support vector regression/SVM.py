import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('Position_Salaries.csv')
x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

y = y.reshape(len(y),1)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)

y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

plt.scatter(x,y,c='b')
plt.plot(x,regressor.predict(x),c='r')
plt.title('Truth or bluf')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x,y, c = 'r')
plt.plot(x_grid,regressor.predict(x_grid), c = 'b')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()