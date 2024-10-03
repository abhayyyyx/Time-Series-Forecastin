import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf  # For candlestick chart
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from io import BytesIO

# Custom CSS for background image
css = """
<style>
body {
    background-image: url("download.jfif");
    background-size: cover;
    background-attachment: fixed;
    color: white;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

def get_stock_data(symbol, start_date, end_date):
    # Fetch stock data from Yahoo Finance
    data = yf.download(symbol, start=start_date, end=end_date)
    # Calculate average price
    data['Avg Price'] = data[['Open', 'Low', 'High', 'Adj Close']].mean(axis=1)
    # Calculate the difference in average price
    data['Differenced'] = data['Avg Price'].diff()
    # Drop missing values
    data.dropna(subset=['Differenced'], inplace=True)
    return data

def sarimax_model(train, test, steps):
    # Train the SARIMAX model
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    fitted_model = model.fit(disp=False)
    # Generate predictions
    predictions = fitted_model.forecast(steps=len(test))
    future_predictions = fitted_model.forecast(steps=steps)

    # Calculate error metrics
    mae = mean_absolute_error(test, predictions)
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)

    return predictions, future_predictions, mae, mse, rmse

st.title('Stock Price Prediction with SARIMAX')

# Inputs for stock symbol and date range
symbol = st.text_input('Stock Symbol', 'RELIANCE.NS')
start_date = st.date_input('Start Date', pd.to_datetime('2014-10-01'))
end_date = st.date_input('End Date', pd.to_datetime('2024-09-30'))
future_steps = st.number_input('Future Steps', min_value=1, max_value=100, value=30)

# Fetch the stock data
data = get_stock_data(symbol, start_date, end_date)
if data.empty:
    st.error('No data fetched. Please check the stock symbol and date range.')
else:
    st.sidebar.title('Navigation')
    section = st.sidebar.radio('Go to', ['Data Details', 'Visualizations', 'Predictions'])

    if section == 'Data Details':
        st.header('Data Details')
        st.write('The data fetched from Yahoo Finance includes the stock prices for the selected symbol within the specified date range. Below is the raw data along with key summary statistics that provide an overview of the stock’s performance.')
        st.write(data)
        st.write('## Summary Statistics')
        st.write(data.describe())

    elif section == 'Visualizations':
        st.header('Visualizations')
        st.write('This section provides various visual representations of the stock data to help understand trends, patterns, and anomalies in the stock prices over time.')

        # Historical Average Price
        st.subheader('Historical Average Price')
        st.write('The Historical Average Price plot shows the average price of the stock calculated from the daily open, low, high, and adjusted close prices. This gives a smoothed view of the stock’s performance.')
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(data.index, data['Avg Price'], label='Avg Price')
        ax.set_title('Historical Average Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)

        # Moving Average
        st.subheader('20-Day Moving Average')
        st.write('The Moving Average plot demonstrates the 20-day moving average of the stock’s average price, providing insights into the short-term trends and smoothing out price fluctuations.')
        data['Moving Average'] = data['Avg Price'].rolling(window=20).mean()
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(data.index, data['Avg Price'], label='Avg Price')
        ax.plot(data.index, data['Moving Average'], label='20-Day Moving Average', color='orange')
        ax.set_title('20-Day Moving Average')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)

        # Volume of Shares Traded
        st.subheader('Volume of Shares Traded')
        st.write('The Volume of Shares Traded plot shows the number of shares traded on each day. It’s a good indicator of the stock’s activity and investor interest over time.')
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.bar(data.index, data['Volume'], label='Volume', color='green')
        ax.set_title('Volume of Shares Traded')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volume')
        st.pyplot(fig)

        # Candlestick Chart
        st.subheader('Candlestick Chart (Last 60 Days)')
        st.write('The Candlestick Chart provides a visual representation of the stock’s price movements, including the open, high, low, and close prices. It is particularly useful for identifying patterns in the stock’s behavior over the last 60 days.')
        fig, ax = plt.subplots(figsize=(14, 7))
        mpf.plot(data[-60:], type='candle', ax=ax)  # Show last 60 days of data
        ax.set_title('Candlestick Chart (Last 60 Days)')
        st.pyplot(fig)

    elif section == 'Predictions':
        st.header('Predictions')
        st.write('This section uses the SARIMAX model to predict future stock prices. The model is trained on historical data and forecasts the stock’s average price for the specified number of future days.')

        train_size = int(len(data) * 0.8)
        train = data['Avg Price'][:train_size]
        test = data['Avg Price'][train_size:]

        if st.button('Predict'):
            sarimax_predictions, sarimax_future, sarimax_mae, sarimax_mse, sarimax_rmse = sarimax_model(train, test, future_steps)
            st.write(f"The SARIMAX model predicts future stock prices based on past data. Here are the error metrics that evaluate the model’s performance: MAE (Mean Absolute Error): {sarimax_mae}, MSE (Mean Squared Error): {sarimax_mse}, RMSE (Root Mean Squared Error): {sarimax_rmse}.")

            future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='B')

            st.subheader('Prediction Results')
            st.write('The Prediction Results plot compares the historical data with the predicted values. The orange line represents the SARIMAX model’s future predictions.')
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(data.index, data['Avg Price'], label='Historical Data')
            ax.plot(test.index, test, label='Test Data', color='blue')
            ax.plot(test.index, sarimax_predictions, label='SARIMAX Predictions', color='red')
            ax.plot(future_dates, sarimax_future, label='SARIMAX Future Predictions', color='orange')
            ax.legend()
            st.pyplot(fig)

            st.subheader('Future Predictions')
            future_df = pd.DataFrame({'Date': future_dates, 'Predicted Avg Price': sarimax_future})
            st.write(future_df)

            # Future Predictions
            st.write('The following plot shows the predicted average price of the stock for the specified future period.')
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(future_dates, sarimax_future, label='SARIMAX Future Predictions', color='cyan', linestyle='--')
            ax.legend()
            st.pyplot(fig)
