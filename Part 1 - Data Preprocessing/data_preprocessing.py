# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# taking care of the missing data 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding Categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0 ] = labelEncoder_X.fit_transform(X[:,0])
X[:,0]

oneHotEncoder = OneHotEncoder(categorical_features = [0])
X= oneHotEncoder.fit_transform(X).toarray()

labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

#Splitting the dataset into the training and test dataset 

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.3)

#Feature Scaling - Helps improve machine learning algorithm, as it removes the bias of larger values such as 
#salary in this dataset ( even if the algorith is not based on Eucledian distance, it help algo converge faster)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
#Fits the training dataset to scaler and transforms the dataset as well. 
sc_X=sc_X.fit(X_train)
X_train = sc_X.transform(X_train)
#Do not fit the test dataset again, as we want it to scale as per train set.
X_test = sc_X.transform(X_test)

