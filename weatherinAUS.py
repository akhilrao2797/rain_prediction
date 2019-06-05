# -*- coding: utf-8 -*-
"""
@author:akhil
"""

#importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import keras

from keras.models import Sequential
from keras.layers import Dense


#importing dataset
dataset = pd.read_csv('weatherAUS.csv')

X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, 23:24].values


#data preprocessing 
# Gives mean for every NaN
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=0)
#axis =0 means along column
imputer = imputer.fit(X[:, 1:6])
X[:,1:6] = imputer.transform(X[:,1:6])
imputer = imputer.fit(X[:, 7:8]) 
X[:,7:8] = imputer.transform(X[:,7:8])
imputer = imputer.fit(X[:, 10:-2]) 
X[:,10:-2] = imputer.transform(X[:,10:-2])
imputer = imputer.fit(X[:, 21:22]) 
X[:,21:22] = imputer.transform(X[:,21:22])

# Converts string to integers
from sklearn.preprocessing import LabelEncoder
labelEncoder_place=LabelEncoder()
X[:,0] = labelEncoder_place.fit_transform(X[:,0])
X[:,6] = labelEncoder_place.fit_transform(X[:,6].astype(str))
X[:,8] = labelEncoder_place.fit_transform(X[:,8].astype(str))
X[:,9] = labelEncoder_place.fit_transform(X[:,9].astype(str))
X[:,20] = labelEncoder_place.fit_transform(X[:,20].astype(str))
Y[:,0] = labelEncoder_place.fit_transform(Y[:,0].astype(str))

Y=Y.astype('int64',copy=True)

#OutlierAnalysis = sb.pairplot(pd.DataFrame(X[:,4:]),hue="WindGustSpeed",palette="hls")
#splitting the data to testset and trainset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)

#af=pd.DataFrame(X)
#gf=pd.DataFrame(X_test)
#hf=pd.DataFrame(X_train)
#lf=pd.DataFrame(Y_test)
#jf=pd.DataFrame(Y_train)


#KNN
from sklearn.neighbors import KNeighborsClassifier
classfier = KNeighborsClassifier()
classfier.fit(X_train,Y_train.ravel())

Y_pred=classfier.predict(X_test)

from sklearn.metrics import confusion_matrix
KNNcm = confusion_matrix(Y_test,Y_pred)


#SVM
from sklearn.svm import SVC
classifier =SVC(kernel='linear',random_state=0)
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
SVMcm = confusion_matrix(Y_test,Y_pred)


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,Y_train)


Y_pred1=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
NaiveBayescm=confusion_matrix(Y_test,Y_pred1)
print(NaiveBayescm)


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)

Y_pred2=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
DecisionTreecm=confusion_matrix(Y_test,Y_pred2)
print(DecisionTreecm)


#Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=20,random_state=0)
regressor.fit(X_train,Y_train)

Y_pred3=regressor.predict(X_test)

from sklearn.metrics import confusion_matrix
RandomForestcm=confusion_matrix(Y_test,Y_pred3)
print(RandomForestcm)

#ANN
classifier=Sequential()
classifier.add(Dense(units=15,kernel_initializer='glorot_uniform',activation='relu'))

classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,Y_train,batch_size=10,epochs=5)

Y_pred4=classifier.predict(X_test)
Y_pred4=(Y_pred4>0.5)

from sklearn.metrics import confusion_matrix
cm4=confusion_matrix(Y_test,Y_pred4)


#Analysing the data through the graph
LocationVsWindSpeed = sb.lmplot(x='Location',y='MinTemp',
           data=dataset,fit_reg=False)
LocationVsWindSpeed.fig.set_size_inches(50,4)
LocationVsWindSpeed = sb.lmplot(x='Location',y='MaxTemp',
           data=dataset,fit_reg=False)
LocationVsWindSpeed.fig.set_size_inches(50,4)
LocationVsWindSpeed = sb.lmplot(x='Location',y='Rainfall',
           data=dataset,fit_reg=False)
LocationVsWindSpeed.fig.set_size_inches(50,4)
LocationVsWindSpeed = sb.lmplot(x='Location',y='Evaporation',
           data=dataset,fit_reg=False)
LocationVsWindSpeed.fig.set_size_inches(50,4)
LocationVsWindSpeed = sb.lmplot(x='Location',y='Sunshine',
           data=dataset,fit_reg=False)
LocationVsWindSpeed.fig.set_size_inches(50,4)
'''
plt.rcParams['figure.figsize']=(60,10)
LocationVsWindGustDir = plt.scatter(dataset.iloc[:,1],pd.DataFrame(X[:,7]),color='k')
plt.show()'''
LocationVsWindSpeed = sb.lmplot(x='Location',y='WindGustSpeed',
           data=dataset,fit_reg=False)
LocationVsWindSpeed.fig.set_size_inches(50,4)
'''
LocationVsWindGustDir = plt.scatter(dataset.iloc[:,1],dataset.iloc[:,9],color='k')
plt.show()
LocationVsWindGustDir = plt.scatter(dataset.iloc[:,1],dataset.iloc[:,10],color='k')
plt.show()
'''
LocationVsWindSpeed = sb.lmplot(x='Location',y='WindSpeed9am',
           data=dataset,fit_reg=False)
LocationVsWindSpeed.fig.set_size_inches(50,4)
LocationVsWindSpeed = sb.lmplot(x='Location',y='WindSpeed3pm',
           data=dataset,fit_reg=False)
LocationVsWindSpeed.fig.set_size_inches(50,4)
LocationVsWindSpeed = sb.lmplot(x='Location',y='Humidity9am',
           data=dataset,fit_reg=False)
LocationVsWindSpeed.fig.set_size_inches(50,4)
LocationVsWindSpeed = sb.lmplot(x='Location',y='Humidity3pm',
           data=dataset,fit_reg=False)
LocationVsWindSpeed.fig.set_size_inches(50,4)
LocationVsWindSpeed = sb.lmplot(x='Location',y='Pressure9am',
           data=dataset,fit_reg=False)
LocationVsWindSpeed.fig.set_size_inches(50,4)
LocationVsWindSpeed = sb.lmplot(x='Location',y='Pressure3pm',
           data=dataset,fit_reg=False)
LocationVsWindSpeed.fig.set_size_inches(50,4)
LocationVsWindSpeed = sb.lmplot(x='Location',y='Cloud9am',
           data=dataset,fit_reg=False)
LocationVsWindSpeed.fig.set_size_inches(50,4)
LocationVsWindSpeed = sb.lmplot(x='Location',y='Cloud3pm',
           data=dataset,fit_reg=False)
LocationVsWindSpeed.fig.set_size_inches(50,4)
LocationVsWindSpeed = sb.lmplot(x='Location',y='Temp9am',
           data=dataset,fit_reg=False)
LocationVsWindSpeed.fig.set_size_inches(50,4)
LocationVsWindSpeed = sb.lmplot(x='Location',y='Temp3pm',
           data=dataset,fit_reg=False)
LocationVsWindSpeed.fig.set_size_inches(50,4)
'''LocationVsWindSpeed = sb.lmplot(x='Location',y='RainToday',
           data=dataset,fit_reg=False)
LocationVsWindSpeed.fig.set_size_inches(50,4)'''
LocationVsWindSpeed = sb.lmplot(x='Location',y='RISK_MM',
           data=dataset,fit_reg=False)
LocationVsWindSpeed.fig.set_size_inches(50,4)