# Dataframes 


# Visualizations 
# https://community.modeanalytics.com/python/tutorial/python-histograms-boxplots-and-distributions/

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('forestfires.csv')

print(dataset.shape)

print(dataset.columns)

print(dataset.info())

print(dataset.describe())

print(dataset.describe(include = ['object','int64','float64']))

#look at the distribution 
print(dataset['ISI'].value_counts())


#sorting
#f.sort_values(by='Total day charge', ascending=False).head()

#Indexing and retriving
# single variable -- > DataFrame['Name']
#df.loc[0:5, 'State':'Area code']
#df.iloc[0:5, 0:3]

# some imports and "magic" commands to set up plotting 
dataset.hist(bins=20)

# Log transformation
dataset['area'].apply(lambda x:np.log(x+1)).hist()



import matplotlib.pyplot as plt 
# pip install seaborn 
plt.scatter(x=dataset['temp'].values,y=dataset['area'].apply(lambda x:np.log(x+1)).values)


dataset['area'] = dataset['area'].apply(lambda x:np.log(x+1))




import statsmodels.api as sm

y= dataset.iloc[:,12]
X = dataset.iloc[:,0:12]

from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
X.iloc[:,2] = labelEncoder_X.fit_transform(X.iloc[:,2].values)
X.iloc[:,3] = labelEncoder_X.fit_transform(X.iloc[:,3].values)



regressor_OLS = sm.OLS(y,X).fit()

regressor_OLS.summary()

plt.scatter(x=X['wind'],y=dataset['area'])
plt.show()