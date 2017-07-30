import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

df = pd.read_csv('Google_Stock_Price_Train.csv')

input_data= df.iloc[:,1:2].values

sc = MinMaxScaler()
input_data = sc.fit_transform(input_data)

X_train = input_data[0:1257]
y_train = input_data[1:1258]

X_train = np.reshape(X_train,(1257,1,1))

regressor = Sequential()
regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))
regressor.add(Dense(units=1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)

test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values

inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20, 1, 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print "Error: "+str(rmse)