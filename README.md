# Gold Price Forecasting AI

An intelligent desktop application that utilizes a Long Short-Term Memory (LSTM) neural network to predict gold prices. The system automatically fetches financial data, calculates technical indicators, and provides users with an intuitive graphical interface to visualize historical trends and future forecasts.

## Project Preview

### Application Interface

Here is how the application interface looks in practice. The uncomplicated GUI provides an intuitive way to interact with the complex machine learning model running in the background. You can generate new predictions and view market trends with a single click.

![Application (Main Window)](images/app_main.png)

### Dynamic Prediction Charts

The application integrates `matplotlib` directly into the Tkinter window. The charts display historical closing prices alongside the future trajectory predicted by the AI model, making the data easy to read and analyze. Tooltips allow users to check the exact price on a specific day.

![Application (Forecast Chart)](images/app_chart.png)

### Dataset & Feature Engineering

Before the neural network can learn, the raw market data is heavily processed. Below is a preview of the preprocessed dataset, enriched with calculated technical indicators (such as RSI, MACD, and Bollinger Bands) which help the model understand market momentum and volatility.

![Dataset Preview](images/data_preview.png)

## Main Features

This project demonstrates a complete Machine Learning pipeline, from data gathering to deployment. The most important features are:

* **Automated Data Fetching:** The system uses the `yfinance` API to download the latest market data automatically, ensuring the model always works with up-to-date information.
* **Advanced Feature Engineering:** Calculation of various technical indicators (RSI, MACD, EMA, ATR) using the `ta` library to improve the neural network's accuracy.
* **Deep Learning Model:** A well-structured LSTM architecture built with TensorFlow/Keras, specifically designed for sequential time-series forecasting.
* **Interactive Visualization:** Dynamic, embedded charts that allow users to visually compare real market data with AI predictions.
* **Safe & Automated Deployment:** A custom Windows batch script (`launch.bat`) automates the entire setup process, including virtual environment creation and dependency installation.

## Code Examples

### Building the Neural Network

The model uses an LSTM architecture optimized for financial time series. It includes a `BatchNormalization` layer to act as a shock absorber for sudden price spikes, and a `Dropout` layer to prevent the model from overfitting to historical data.

```python
model = Sequential()

# A compact LSTM layer designed to capture strong trends and ignore market noise
model.add(LSTM(
    units=16, 
    return_sequences=False, 
    input_shape=(X_train.shape[1], X_train.shape[2])
))

# Batch Normalization acts as a shock absorber for sudden price spikes
model.add(BatchNormalization())

# Dropout randomly disables a portion of neurons to prevent overfitting
model.add(Dropout(0.3))

model.add(Dense(16, activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")
```

### Feature Engineering (Technical Indicators)

To help the neural network understand the market context, the raw price data is enriched with established financial indicators using the `ta` library.

```python
# --- MOMENTUM & TREND INDICATORS ---

# RSI (Relative Strength Index)
rsi_indicator = RSIIndicator(close=close_prices, window=14)
df['RSI'] = rsi_indicator.rsi()

# MACD (Moving Average Convergence Divergence)
macd = MACD(close=close_prices, window_slow=26, window_fast=12, window_sign=9)
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()

# Bollinger Bands - Shows if the price is unusually high or low
bb_indicator = BollingerBands(close=close_prices, window=20, window_dev=2)
df['BB_High'] = bb_indicator.bollinger_hband()
df['BB_Low'] = bb_indicator.bollinger_lband()
```

### Autoregressive Forecasting Loop

The application generates future predictions step-by-step. It takes the last known 30 days of data, predicts tomorrow's price, adds it to the sequence, and drops the oldest day to predict the day after tomorrow.

```python
# Execute the autoregressive prediction loop for future days
for _ in range(forecast_days):
    input_data = last_sequence.reshape(1, TIME_STEPS, len(FEATURE_COLUMNS))
    predicted_scaled_close = model.predict(input_data, verbose=0)[0][0]

    # Convert scaled value back to real USD
    predicted_close = target_scaler.inverse_transform([[predicted_scaled_close]])[0][0]
    future_predictions.append(predicted_close)

    # Update the sequence window with the new prediction
    new_sequence = np.roll(last_sequence, -1, axis=0)
    new_sequence[-1, 0] = predicted_scaled_close
    last_sequence = new_sequence
```

## Technologies & Tools

* **Core & GUI:** Python 3.10, Tkinter
* **Machine Learning:** TensorFlow, Keras, Scikit-Learn
* **Data Handling & APIs:** Pandas, NumPy, Yahoo Finance (`yfinance`), Technical Analysis (`ta`)
* **Data Visualization:** Matplotlib
* **Development & Prototyping:** Jupyter Notebook

## Project Structure

* `src/desktop_app.py`: Main entry point containing the Tkinter user interface and chart rendering logic.
* `src/data_prep.py`: Script responsible for downloading data and feature engineering.
* `notebooks/lstm_gold_model.ipynb`: Jupyter Notebook used for model prototyping, training, and evaluation.
* `data/best_gold_model.keras`: Saved weights of the trained neural network.
* `launch.bat`: Automated deployment script for Windows environments.

## How to Run?

Follow these steps to safely set up and run the project on your local machine:

1. **System Requirements:**
   * Ensure you are running **Windows 10/11**.
   * You must have exactly **Python 3.10.x (64-bit)** installed on your machine. Newer versions (3.11+) may lack full TensorFlow support on Windows.
   * *If needed, download the official installer here:* [Python 3.10.11 (64-bit)](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe). 
   * **Important:** Make sure to check the **"Add Python to PATH"** box during installation.

2. **Quick Start (Automated Setup):**
   * Navigate to the root directory of the project.
   * Double-click the **`launch.bat`** file.
   * The script will automatically verify your Python version, create a `.venv` folder, install all libraries from `requirements.txt`, and launch the application GUI.

3. **Manual Setup (Alternative):**
   * If you prefer to use the command line, open your terminal in the project directory.
   * Create a virtual environment:
     ```bash
     python -m venv .venv
     ```
   * Activate the virtual environment:
     ```bash
     .venv\Scripts\activate
     ```
   * Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   * Run the application:
     ```bash
     python src/desktop_app.py
     ```

---
<p align="center">
  Made with 💻 by <strong>Antoni Jackowski</strong>
</p>
