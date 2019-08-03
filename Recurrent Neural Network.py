# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 23:38:21 2019

@author: Jagan Mohan
"""

# Part 1 - Importing & Data Preprocessing 

#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Datasets
training_data = pd.read_csv('Google_Stock_Price_Train.csv')
training_data = training_data.iloc[:,1:2].values

#Let us normalize our data #Since we are using sigmoid activation functions normalization may give better results than standardization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_transform = sc.fit_transform(training_data)

#Since we are predicting the stock price of the next day based on the previous stock price results, Let us build x & y samples accordingly
X_train = training_transform[0:1257,:]
y_train = training_transform[1:1258,:] 

#Converting our training #To include the time step
X_train = X_train.reshape(1257,1,1)


# Part 2 - Building th RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#Initializing the RNN
regressor = Sequential()

#Adding the LSTM Layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

#Adding the Output Layer
regressor.add(Dense(units = 1))

#Compile the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fit the RNN
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)


# Part 3 - Make Predictions & Evaluate

#Importing the Real Stock Price(Test Data)
test_data = pd.read_csv('Google_Stock_Price_Test.csv')
test_data = test_data.iloc[:,1:2].values
real_stock_price = test_data

#Feature Scaling
test_data = sc.transform(test_data)

#Reshaping
X_test = test_data.reshape(20,1,1)

#Predictions
predicted_stock_price = regressor.predict(X_test)

#Reverse Transform
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualizing the Results
plt.plot(real_stock_price, color = 'green', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.legend()
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.show()
#As you can see the plot that we have predicted the stock price almost matches the trend of predicted stock price

#Now let us save our RNN
regressor.save_weights('RNN_Basic_Weights.hdf5')






