# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:03:18 2017

@author: umer
"""

import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing 

#data=pd.read_excel('C:\Users\umer\Documents\Visual Studio 2015\Projects\datamining22\datamining22\Book23.xlsx')
#data.to_excel('C:\Users\umer\Documents\Visual Studio 2015\Projects\datamining22\datamining22\Book23.xlsx')
data = pd.read_excel('Book23.xlsx')
#data.head()
#print(data)

lookup_data =dict(zip(data.Physician_Primary_Type.unique(), data.Physician_Specialty.unique()))

print(lookup_data)



Y = data[['Total_Amount_Invested_USDollars']].values
X1 = data['Physician_Specialty'].values

#hash(tuple(np.array([X1])))

#s=set(X1)
#D=dict( zip(s,range(len(s))))
#X=[D[X1] for X1_ in X1]
#hash(tuple(np.array([X1])))
lb = LabelEncoder()
X= lb.fit_transfrom(X1)
lb.inverse_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

#cmap = cm.get_cmap('gnuplot')
#scatter = pd.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

data_prediction = knn.predict([["Allopathic & Osteopathic Physicians" ]])
lookup_data[data_prediction[0]]
#data_prediction = knn.predict([[]])
#lookup_data[data_prediction[0]]
