# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# removing one dummy variable to remove dummy varialbe trap 
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Part -2 Lets make the ANN!

# Importing the keras lib and the required packages
import keras
from keras.models import Sequential
from keras.layers import Dense
 
#initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layers
#Units - No of outputs *keep average of avg of input+output ie 11+1/2 = 6
#kernel_initializer - sets the initial weights to the input
#activation - activation function - relu is rectifier
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

# Addiing the output layer
#activation - choosing sigmoid for output layer - activation = sigmoid
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

#Compiling the ANN
#optimizer - algo used to find optimal weights - adam - stochastic gradient descent aglo
#loss - cost function that we have to optimize using sgd- logarithmic loss 1 category dependent variable
# binary_crossentropy
# metrics - accuracy metrics
classifier.compile(optimizer= 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

# Fitting the ANN to the training set

classifier.fit(X_train,y_train,batch_size = 10,epochs=100 )

# Predict the test set result 
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)