import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import datetime

# Step 1: Web Scraping - Gather real-time stock data
def scrape_stock_data(stock_symbol):
    url = f'https://finance.yahoo.com/quote/{stock_symbol}/history?p={stock_symbol}'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')
    rows = table.find_all('tr')

    data = []
    for row in rows[1:]:
        cols = row.find_all('td')
        if len(cols) == 7:
            date = cols[0].text
            close_price = cols[4].text
            data.append({'Date': date, 'Close Price': close_price})

    df = pd.DataFrame(data)
    return df

# Step 2: Preprocess Data
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Close Price'] = pd.to_numeric(df['Close Price'].str.replace(',', ''), errors='coerce')
    df = df.dropna()
    return df

# Step 3: Train a Machine Learning model
def train_model(df):
    X = df['Date'].astype(int).values.reshape(-1, 1)
    y = df['Close Price'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)

    y_test_pred = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)

    return model, mse_train, mse_test

# Step 4: Make Predictions
def make_predictions(model, df):
    last_date = df['Date'].iloc[-1]
    next_date = last_date + datetime.timedelta(days=1)
    next_date = next_date.date()

    next_date_int = next_date.astype(int)
    next_date_int = next_date_int.reshape(-1, 1)

    prediction = model.predict(next_date_int)

    return next_date, prediction

# Step 5: Execute the Workflow
stock_symbol = 'AAPL'  # Replace with your desired stock symbol
df = scrape_stock_data(stock_symbol)
df = preprocess_data(df)
model, mse_train, mse_test = train_model(df)
next_date, prediction = make_predictions(model, df)

# Step 6: Print Results
print(f"Stock Symbol: {stock_symbol}")
print(f"Last Date: {df['Date'].iloc[-1]}")
print(f"Next Date: {next_date}")
print(f"Close Price Prediction: {prediction[0][0]}")