# Stock Trend Prediction Web App (LSTM + Streamlit)

This project is an interactive web application that predicts stock price trends using a deep learning model (LSTM). It allows users to input any stock ticker, visualize historical price behavior, and compare actual prices against model predictions in real time.

The application is built with:

* **Streamlit** for the UI
* **Yahoo Finance (yfinance)** for live market data
* **Keras (TensorFlow backend)** for the trained LSTM model
* **Scikit-learn** for data normalization
* **Matplotlib** for visualizations

---

## Objective

The main goal is to:

* Predict future stock closing prices based on historical trends
* Demonstrate how deep learning can be applied to time-series financial data
* Provide an interactive interface for non-technical users to explore stock behavior

This project focuses on **trend prediction**, not exact price forecasting, which is more realistic in financial modeling.

---

## Model Overview

The model is a pre-trained **LSTM (Long Short-Term Memory)** neural network designed to learn temporal patterns in stock price movements.

Key characteristics:

* Input: Last **100 days of closing prices**
* Output: Predicted closing price for the next day
* Data normalization using **MinMaxScaler**
* Trained on historical stock data before deployment

The trained model is loaded from:

```python
model = load_model('stock_dl_model.h5')
```

---

## Data Pipeline

1. **User Input**

   * User enters a stock ticker (default: `AAPL`)
   * Example: `TSLA`, `MSFT`, `GOOGL`

2. **Data Download**

   * Historical price data is fetched from Yahoo Finance:

   ```python
   df = yf.download(user_input, start, end)
   ```

   * Time range: **2015 – 2024**

3. **Exploratory Statistics**

   * Summary statistics of stock data:

   ```python
   st.write(df.describe())
   ```

4. **Visual Analysis**

   * Closing price vs time
   * Closing price with:

     * 100-day moving average
     * 100 & 200-day moving averages

These help identify:

* Long-term trends
* Market momentum
* Support and resistance behavior

---

## Model Testing Process

1. **Train/Test Split**

   * 70% for training
   * 30% for testing

2. **Scaling**

   * Prices normalized between 0 and 1 using `MinMaxScaler`

3. **Window Creation**

   * Each prediction uses the previous **100 days** of price data

4. **Prediction**

   ```python
   y_predicted = model.predict(x_test)
   ```

5. **Inverse Scaling**

   * Convert predictions back to real stock prices

---

## Output Visualization

The final plot shows:

* **Original Price** (Actual closing price)
* **Predicted Price** (Model output)

This allows users to:

* Visually evaluate prediction accuracy
* Observe how closely the model follows market trends

Example graph:

```
Original Price  → Real stock movement  
Predicted Price → Model trend estimation  
```

<img width="953" height="463" alt="image" src="https://github.com/user-attachments/assets/a5bbef7c-2c27-4a3d-a620-7c8c4ab9341e" />


---

## Application Features

* 🔎 User-defined stock ticker input
* 📉 Historical price visualization
* 📊 Moving average trend analysis
* 🤖 Deep learning prediction using LSTM
* 📈 Actual vs Predicted price comparison
* ⚡ Real-time interactive experience via Streamlit

---

##  Disclaimer

This project is for **educational and research purposes only**.
It is not intended to be used as financial advice or a trading recommendation system.

Stock markets are highly volatile, and predictions are based solely on historical trends.

---

## Tech Stack

| Component       | Technology                |
| --------------- | ------------------------- |
| Data Source     | Yahoo Finance (yfinance)  |
| Model           | LSTM (Keras / TensorFlow) |
| Frontend        | Streamlit                 |
| Visualization   | Matplotlib                |
| Data Processing | NumPy, Pandas             |
| Scaling         | Scikit-learn              |

---

## Future Improvements

* Add multi-feature modeling (Volume, Open, High, Low)
* Enable multi-day forecasting
* Improve model accuracy with hyperparameter tuning
* Add evaluation metrics (RMSE, MAE)
* Deploy using Docker or Streamlit Cloud
* Support comparison of multiple stocks

---

