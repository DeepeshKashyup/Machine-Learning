# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# Not needed since dataset is very small and we need all the data for accurate 
# prediction for this business case
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
#"""from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#3X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)"""


# Fitting Linear regression to the dataset
# Simply for comparison 

from sklearn.linear_model import LinearRegression

LinReg = LinearRegression()
LinReg.fit(X,y)

# Fitting Polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree=4)
X_Poly = polyReg.fit_transform(X)

LinReg_2 = LinearRegression()
LinReg_2.fit(X_Poly,y)

#Visualizing the linear regression result
plt.scatter(X,y,color='Red')
plt.plot(X,LinReg.predict(X),color='Blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary Level')
plt.show()

#Visualizing the Polynomial Regression result
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='Red')
plt.plot(X_grid,LinReg_2.predict(polyReg.fit_transform(X_grid)),color= 'Blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary Level')
plt.show()

LinReg_2.predict(polyReg.fit_transform(6.5))
#Predicting a new result with Linear Regression
LinReg.predict(6.5)

#Predicting a new result with Polynomial Regression
