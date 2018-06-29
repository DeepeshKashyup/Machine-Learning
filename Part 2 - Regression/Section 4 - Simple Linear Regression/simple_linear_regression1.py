# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 13:00:53 2018

@author: Deepesh
"""
#Simple Linear Regresssion

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


#Splitting the dataset into the training and test dataset 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 1/3,random_state=0)

# Feature Scaling - this is taken care of by linear regression models
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


# Fitting the simple linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# Predictiing the test set result
y_pred = regressor.predict(X_test)

#Visualizing the Training Set Results
plt.scatter(X_train,y_train,c='red')
plt.plot(X_train,regressor.predict(X_train),color='Blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()



#Visualizing the Test Set Results
plt.scatter(X_test,y_test,c='red')
# no need to change the parameters for regression line as the model is already built and changeing
#to test data will result in same line, as the line equation will be the same
plt.plot(X_train,regressor.predict(X_train),color='Blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()