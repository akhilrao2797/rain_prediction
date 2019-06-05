# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 09:23:23 2019

@author: p nageswara rao
"""

#importing the libraries

import numpy as np
import pandas as pd
from datetime import date
#import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
#from keras.models import Sequential
#from keras.layers import Dense


class weatherinAUS:
    
    def __init__(self,attributes):
        self.data = pd.DataFrame(attributes)
        self.dataset = pd.read_csv('weatherAUS.csv')
        self.dataset = self.dataset.append(self.data,sort=False)
        self.X = self.dataset.iloc[:, 1:-1].values
        self.Y = self.dataset.iloc[:, 23:24].values
        self.X_train = []
        self.Y_train = []
        self.Y_test = []
        self.X_test = []
        self.prediction = {}
        
    def display(self):
        return self.dataset

    def preprocessingData(self):
        """data preprocessing 
        Gives mean for every NaN"""
        
        imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=0)
        #axis =0 means along column
        imputer = imputer.fit(self.X[:, 1:6]) 
        self.X[:,1:6] = imputer.transform(self.X[:,1:6])
        imputer = imputer.fit(self.X[:, 7:8]) 
        self.X[:,7:8] = imputer.transform(self.X[:,7:8])
        imputer = imputer.fit(self.X[:, 10:-2]) 
        self.X[:,10:-2] = imputer.transform(self.X[:,10:-2])
        imputer = imputer.fit(self.X[:, 21:22]) 
        self.X[:,21:22] = imputer.transform(self.X[:,21:22])
    
        # Converts string to integers
        
        labelEncoder_place = LabelEncoder()
        self.X[:,0] = labelEncoder_place.fit_transform(self.X[:,0])
        self.X[:,6] = labelEncoder_place.fit_transform(self.X[:,6].astype(str))
        self.X[:,8] = labelEncoder_place.fit_transform(self.X[:,8].astype(str))
        self.X[:,9] = labelEncoder_place.fit_transform(self.X[:,9].astype(str))
        self.X[:,20] = labelEncoder_place.fit_transform(self.X[:,20].astype(str))
        self.Y[:,0] = labelEncoder_place.fit_transform(self.Y[:,0].astype(str))
    
        self.Y=self.Y.astype('int64',copy=True)
    
    def splitData(self):
        '''splitting the data to testset and trainset''' 
        
        self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(self.X,self.Y,test_size=0.20,random_state=0)
    
        #af=pd.DataFrame(X)
        #gf=pd.DataFrame(X_test)
        #hf=pd.DataFrame(X_train)
        #lf=pd.DataFrame(Y_test)
        #jf=pd.DataFrame(Y_train)
    def KNN(self,data):
        classifier = KNeighborsClassifier()
        classifier.fit(self.X_train,self.Y_train)
        
        prediction = classifier.predict(data)
        
        return prediction
    
    
    
    def SVM(self,data):
        #SVM
        classifier = SVC(kernel='linear',random_state=0)
        classifier.fit(self.X_train,self.Y_train)
        
        prediction = classifier.predict(data)
        
        return prediction
    
    
    
    def NaiveBayes(self,data): 
        #Naive Bayes
        
        classifier=GaussianNB()
        classifier.fit(self.X_train,self.Y_train)
        
        prediction = classifier.predict(data)
        
        return prediction
    
    
        
    def DecisionTree(self,data):
        #Decision Tree
        
        classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
        classifier.fit(self.X_train,self.Y_train)
        
        prediction = classifier.predict(data)
        
        return prediction
    
    
    
    def RandomForest(self,data):        
        #Random Forest
        
        regressor=RandomForestRegressor(n_estimators=20,random_state=0)
        regressor.fit(self.X_train,self.Y_train)
        
        prediction = regressor.predict(data)
        
        return prediction
        
    
    
    
    def start(self):
        self.preprocessingData()
        data = self.X[-1,:]
        self.X = self.X[:-1,:]
        self.Y = self.Y[:-1,]
        data=np.reshape(data,(1,-1))
        self.splitData()
        self.prediction["KNN"]=self.KNN(data).ravel()[0]
        self.prediction["SVM"]=self.SVM(data).ravel()[0]
        self.prediction["NaiveBayes"]=self.NaiveBayes(data).ravel()[0]
        self.prediction["RandomForest"]=self.RandomForest(data).astype('int64',copy=True).ravel()[0]
        self.prediction["DecisionTree"]=self.DecisionTree(data).ravel()[0]
        finalOutput = self.analyse()
        return finalOutput
        
        
    def analyse(self):
        countOfYes = 0
        countOfNo = 0
        
        for key,value in self.prediction.items():
            if(value == 0):
                countOfNo += 1
            else:
                countOfYes += 1
        if(countOfYes == countOfNo):
            return "YESorNO"
        elif(countOfYes > countOfNo):
            return "Yes"
        else:
            return "No"
        
    def ANN(self):
        classifier=Sequential()
        classifier.add(Dense(units=15,kernel_initializer='glorot_uniform',activation='relu'))
        
        classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))
        
        classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        
        classifier.fit(X_train,Y_train,batch_size=10,epochs=5)
        
        Y_pred4=classifier.predict(X_test)
        Y_pred4=(Y_pred4>0.5)
        
        from sklearn.metrics import confusion_matrix
        cm4=confusion_matrix(Y_test,Y_pred4)
'''        
#if __name__ = "__main__":
attributes = {}
attributes["Date"] = [date.today().strftime("%d-%m-%Y")]  
attributes["Location"] = ['Albury']
attributes["MinTemp"] = [13.5]
attributes["MaxTemp"] = [23]
attributes["Rainfall"] = [0.2]
attributes["Evaporation"] = ['nan']
attributes["Sunshine"] = ['nan']
attributes["WindGustDir"] = ['WNW']
attributes["WindGustSpeed"] = [20]
attributes["WindDir9am"] = ['W']
attributes["WindDir3pm"] = ['WNW']
attributes["WindSpeed9am"] = [20]
attributes["WindSpeed3pm"] = [28]
attributes["Humidity9am"] = [60]
attributes["Humidity3pm"] = [22]
attributes["Pressure9am"] = [1005]
attributes["Pressure3pm"] = [1010]
attributes["Cloud9am"] = [9]
attributes["Cloud3pm"] = ['nan']
attributes["Temp9am"] = [15]
attributes["Temp3pm"] = [28]
attributes["RainToday"] = ['No']
attributes["RISK_MM"] = [0]
attributes = {'Date': ['nan'], 'Location': ['Katherine'], 'MinTemp': [10], 'MaxTemp': [12], 'Rainfall': [23], 'Evaporation': [0], 'Sunshine': [0], 'WindGustDir': ['N'], 'WindGustSpeed': [20], 'WindDir9am': ['NE'], 'WindDir3pm': ['SW'], 'WindSpeed9am': [23], 'WindSpeed3pm': [30], 'Humidity9am': ['nan'], 'Humidity3pm': [0], 'Pressure9am': [1000], 'Pressure3pm': [1000], 'Cloud9am': [0], 'Cloud3pm': [0], 'Temp9am': [10], 'Temp3pm': [10], 'RainToday': ['Yes'], 'RISK_MM': [2]}
weather = weatherinAUS(attributes)
k=weather.display()
output =weather.start()        '''

