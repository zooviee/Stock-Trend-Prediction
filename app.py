import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model



start = '2015-01-01'
end = '2024-12-31'

st.title("Stock Trend Prediction")


user_input = st.text_input("Enter Stock Ticker", "AAPL")
df = yf.download(user_input, start, end)

# Describing the data
st.subheader('Data from 2015 - 2024')
st.write(df.describe())


# Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
st.pyplot(fig)



st.subheader('Closing Price vs Time Chart with 100 Moving Average')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(df['Close'])
st.pyplot(fig)



st.subheader('Closing Price vs Time Chart with 100 & 200 Moving Average')
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'b')
plt.plot(ma200, 'r')
plt.plot(df['Close'], 'g')
st.pyplot(fig)


# Training & Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

scaler = MinMaxScaler(feature_range = (0, 1))

data_training_array = scaler.fit_transform(data_training)


# Splitting in X_train and y_train
# X_train contains the past 100 days' prices, y_train contains the next day's price
# This is used to train the LSTM model
# The model will be trained on the past 100 days' prices to predict the next day's price
# The model will learn the patterns in the data and make predictions based on that

    
# Loading the model
model = load_model('stock_dl_model.h5')

# Testing Part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df) 


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test  = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, label = 'Original Price', linewidth = 1)
plt.plot(y_predicted, label = 'Predicted Price', linewidth = 1)
plt.legend()
plt.show()
st.pyplot(fig2)