# Stock Market Prediction

## Overview
This project applies machine learning and deep learning techniques to predict stock prices based on historical market data. The system aims to assist traders and investors by forecasting future stock trends.

## Features
- Data collection from financial APIs (Yahoo Finance, Alpha Vantage, etc.)
- Time series forecasting using machine learning models
- Deep learning models such as LSTMs and Transformers
- Performance evaluation using RMSE, MAE, and R-squared
- Interactive visualization of stock trends and predictions

## Technologies Used
- Python
- TensorFlow/Keras (for deep learning models)
- Scikit-learn (for machine learning models)
- Pandas & NumPy (for data preprocessing)
- Matplotlib & Seaborn (for data visualization)
- Financial data APIs (Yahoo Finance, Alpha Vantage)
- Statsmodels (for ARIMA modeling)

## Dataset
The dataset is obtained from **Yahoo Finance** or other financial data sources. It includes historical stock prices, volume, and technical indicators. Store the dataset in the `data/` directory.

## Model Performance
- Models tested: **Linear Regression, Random Forest, LSTMs, ARIMA, and Transformers**
- Best model achieved **low RMSE and high R-squared scores**

## Required Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
