#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('intracity_fare_train.csv')
dataset_2=pd.read_csv('intracity_fare_test.csv')
X_train=dataset.iloc[:,:-1].values
y_train=dataset.iloc[:,11].values
X_test=dataset_2.iloc[:,:].values
X_train=X_train[:,2:]
X_test=X_test[:,2:]

#Using Imputer for removing the mssing values of X_train
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, 0:4])
X_train[:, 0:4] = imputer.transform(X_train[:, 0:4])

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:,5:7])
X_train[:,5:7] = imputer.transform(X_train[:,5:7])


#Using Imputer for removing the mssing values of X_test
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test[:, 0:4])
X_test[:, 0:4] = imputer.transform(X_test[:, 0:4])

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test[:, 5:7])
X_test[:, 5:7] = imputer.transform(X_test[:,5:7])


#Encoding the vehicle series from X_train and X_test
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_train = LabelEncoder()
X_train[:, 4] = labelencoder_X_train.fit_transform(X_train[:, 4])
onehotencoder = OneHotEncoder(categorical_features = [4])
X_train = onehotencoder.fit_transform(X_train).toarray()

labelencoder_X_test = LabelEncoder()
X_test[:, 4] = labelencoder_X_test.fit_transform(X_test[:, 4])
onehotencoder = OneHotEncoder(categorical_features = [4])
X_test= onehotencoder.fit_transform(X_test).toarray()

#Making the regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)