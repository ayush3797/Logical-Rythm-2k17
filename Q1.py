
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
X_train = dataset.iloc[:,:-1].values
y_train = dataset.iloc[:, 20].values

dataset_2 = pd.read_csv('test.csv')
X_test = dataset_2.iloc[:,:].values
X_test=X_test[:,1:]

# Feature Scaling since the svm library doesn't do feature scaling on itself like the linear_model library
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

