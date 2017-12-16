#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the training dataset
train = pd.read_csv('Train_n.csv')

#Making X_train and y_train
X_train = train.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]].values
y_train = train.iloc[:,19].values

#Importing the test dataset
test_1= pd.read_csv('test.csv')

#Making X_test
X_test=test_1.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]].values


#Importing StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(sc_X.transform(X_test))
y_pred = sc_y.inverse_transform(y_pred)


