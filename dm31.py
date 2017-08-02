import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#data=pd.read_excel('C:\Users\umer\Documents\Visual Studio 2015\Projects\datamining22\datamining22\Book23.xlsx')
#data.to_excel('C:\Users\umer\Documents\Visual Studio 2015\Projects\datamining22\datamining22\Book23.xlsx')
data = pd.read_excel('Book23.xlsx')
#data.head()
#print(data)

lookup_data =dict(zip(data.Physician_Primary_Type.unique(), data.Physician_Specialty.unique()))

print(lookup_data)



X = data[['Total_Amount_Invested_USDollars','Value_of_Interest']].values
Y = data['Physician_Specialty'].values


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

#cmap = cm.get_cmap('gnuplot')
#scatter = pd.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

data_prediction = knn.predict([[2500, 733 ]])
lookup_data[data_prediction[0]]

