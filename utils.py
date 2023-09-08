# utils.py

import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np


def gather_real_time_data(tickers):
    """
    Gather real-time stock market data using the yfinance library.
    
    Parameters:
    - tickers: list of strings, representing the stock tickers
    
    Returns:
    - pandas DataFrame object with the gathered data
    """

    # Fetch data for each ticker
    data = yf.download(tickers=tickers, period="1d", interval="1m", progress=False)

    # Extract only the closing prices
    data = data['Close'].reset_index()

    return data


def preprocess_data(data):
    """
    Preprocess the stock market data.
    
    Parameters:
    - data: pandas DataFrame object with stock market data
    
    Returns:
    - processed_data: numpy array with preprocessed data
    """

    # Convert data to numpy array
    data = data.to_numpy()

    # Scale the data between 0 and 1
    scaler = MinMaxScaler()
    processed_data = scaler.fit_transform(data)

    return processed_data


def split_data(processed_data, test_size=0.2):
    """
    Split the preprocessed data into training and testing sets.
    
    Parameters:
    - processed_data: numpy array with preprocessed data
    - test_size: float, representing the proportion of data to use for testing
    
    Returns:
    - X_train: numpy array with training features
    - X_test: numpy array with testing features
    - y_train: numpy array with training target
    - y_test: numpy array with testing target
    """

    # Split data into features (X) and target (y)
    X = processed_data[:, :-1]
    y = processed_data[:, -1]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train the machine learning model on the training data.
    
    Parameters:
    - X_train: numpy array with training features
    - y_train: numpy array with training target
    
    Returns:
    - model: trained machine learning model
    """

    # Create and train the Support Vector Regression model
    model = SVR()
    model.fit(X_train, y_train)

    return model


def make_predictions(model, X_test):
    """
    Make predictions using the trained model and testing data.
    
    Parameters:
    - model: trained machine learning model
    - X_test: numpy array with testing features
    
    Returns:
    - y_pred: numpy array with predicted values
    """

    # Make predictions using the trained model
    y_pred = model.predict(X_test)

    return y_pred


def evaluate_model(y_test, y_pred):
    """
    Evaluate the trained model using mean squared error.
    
    Parameters:
    - y_test: numpy array with testing target
    - y_pred: numpy array with predicted values
    """

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)


### Example Usage ###

# Gather real-time stock market data for ticker 'AAPL'
data = gather_real_time_data(['AAPL'])

# Preprocess the data
processed_data = preprocess_data(data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = split_data(processed_data, test_size=0.2)

# Train the machine learning model
model = train_model(X_train, y_train)

# Make predictions using the trained model
y_pred = make_predictions(model, X_test)

# Evaluate the model
evaluate_model(y_test, y_pred)
