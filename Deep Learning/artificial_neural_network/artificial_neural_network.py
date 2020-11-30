# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

df=pd.read_csv('Churn_Modelling.csv')
x=df.iloc[:,3:13].values
y=df.iloc[:,13].values


from sklearn.preprocessing import OneHotEncoder,LabelEncoder
label_1=LabelEncoder()
x[:,1]=label_1.fit_transform(x[:,1])
label_2=LabelEncoder()
x[:,2]=label_2.fit_transform(x[:,2])
onehoten=OneHotEncoder(categorical_features=[1])
x=onehoten.fit_transform(x).toarray()
x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


import keras
from keras.models import Sequential
from keras.layers import Dense
classifier=Sequential()
classifier.add(Dense(units=6,activation='relu'))
classifier.add(Dense(units=6,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(x_train,y_train,batch_size=32,epochs=100)


y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

