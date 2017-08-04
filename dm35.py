#this is test code contaions errors and need modification

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 20:57:19 2017

@author: umer
"""

from sklearn import linear_model
import numpy as np
reg = linear_model.Ridge (alpha = .5)
reg.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1]) 
array = array.astype(np.float64)


#reg.coef_

#reg.intercept_

# Split the data into training/testing sets
diabetes_X_train = reg.fit
diabetes_X_test = reg.fit

# Split the targets into training/testing sets
diabetes_y_train = reg.fit
diabetes_y_test = reg.fit

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"    % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))
