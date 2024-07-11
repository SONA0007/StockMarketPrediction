from flask import Flask, request, render_template
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import datetime

app = Flask(__name__)
model = load_model('nse_stock_price_prediction_model.keras')

def fetch_data(stock, start, end):
    data = yf.download(stock, start=start, end=end)
    data.fillna(method='ffill', inplace=True)
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    data.dropna(inplace=True)
    return data

def predict_next_day(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close', 'MA_50', 'MA_200']])
    X_test = np.array([scaled_data[-60:]])
    prediction = model.predict(X_test)
    predicted_price = scaler.inverse_transform(np.concatenate((prediction, np.zeros((prediction.shape[0], 2))), axis=1))[:, 0]
    return predicted_price[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        data = fetch_data(stock_symbol, start_date, end_date)
        if not data.empty:
            predicted_price = predict_next_day(data)
            return render_template('index.html', predicted_price=predicted_price, stock_symbol=stock_symbol)
        else:
            return render_template('index.html', error="Failed to fetch data for the given stock ticker symbol. Please check the symbol and try again.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
