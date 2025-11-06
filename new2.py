import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from plotly import graph_objs as go
import pandas as pd
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
from datetime import datetime, timedelta
import time

# --- Configuration ---
st.set_page_config(page_title="Stock Forecast Comparison App", layout="wide")
st.title("Stock Forecast : Prophet ")

st.sidebar.header("App Configuration")

sector_to_stocks = {
    "Technology ": ["AAPL", "MSFT", "GOOGL", "NVDA"],
    "Financial services ": ["AXP", "JPM", "MA", "V","YESBANK.BO"],
    "Industrials ": ["FDX", "WM", "MMM", "GE"],
    "Healthcare ": ["UNH", "LLY", "VRTX", "ABT"]
}
# Select stock sector
selected_sector = st.sidebar.selectbox("Select stock sectors", list(sector_to_stocks.keys()))

# Select stock based on the selected sector
selected_stock = st.sidebar.selectbox("Select stock", sector_to_stocks[selected_sector])

# Fixed 1 year prediction period (252 trading days)
period = 252

# --- Dynamic Date Range ---
TODAY = "2024-08-20"
START_DATE = "2018-01-01"
TRAIN_END_DATE = "2023-12-31"

# --- Data Loading with Retry Logic ---
@st.cache_data
def load_data(ticker, start, end, retries=3, delay=5):
    with st.status(f"Loading data for {ticker}...", expanded=True) as status:
        for attempt in range(retries):
            try:
                data = yf.download(ticker, start=start, end=end, progress=False)
                if data.empty:
                    status.update(label=f"No data found for {ticker}.", state="error")
                    st.error(f"No data found for {ticker}. Please check the ticker or date range.")
                    return None
                
                data.reset_index(inplace=True)
                
                # Flatten and standardize column names
                new_columns = [col[0] if isinstance(col, tuple) and col[0] else col for col in data.columns]
                data.columns = new_columns
                data.rename(columns={
                    'Date': 'Date', 'date': 'Date',
                    'Close': 'Close', 'close': 'Close',
                    'Open': 'Open', 'open': 'Open',
                    'High': 'High', 'high': 'High',
                    'Low': 'Low', 'low': 'Low',
                    'Volume': 'Volume', 'volume': 'Volume',
                    'Adj Close': 'Adj Close', 'adj close': 'Adj Close'
                }, inplace=True)
                
                if 'Date' not in data.columns or 'Close' not in data.columns:
                    status.update(label=f"Missing required columns for {ticker}.", state="error")
                    st.error(f"Required columns 'Date' or 'Close' not found. Available: {data.columns.tolist()}")
                    return None
                
                status.update(label=f"Data loaded for {ticker}!", state="complete", expanded=False)
                return data
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
                status.update(label=f"Error loading data for {ticker}.", state="error")
                st.error(f"Error loading data for {ticker}: {e}")
                return None

data = load_data(selected_stock, START_DATE, TODAY)
if data is None:
    st.stop()

# --- Data Cleaning ---
if not data.empty:
    data['Date'] = pd.to_datetime(data['Date'])
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data.fillna(method='ffill')  
    initial_rows = len(data)
    data.dropna(subset=['Close'], inplace=True)
    rows_after_cleaning = len(data)
    if initial_rows != rows_after_cleaning:
        st.warning(f"Removed {initial_rows - rows_after_cleaning} rows due to missing or non-numeric 'Close' values.")
    if data.empty:
        st.error("No valid data remains after cleaning. Try a different stock or date range.")
        st.stop()

# --- Add MACD Feature ---
def add_macd_feature(df):
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

data = add_macd_feature(data)

# --- Display Raw Data ---
st.subheader(f"Raw Data for {selected_stock} (Last 5 Rows)")
st.write(data.tail().style.format({
    'Open': '{:.2f}', 'High': '{:.2f}', 'Low': '{:.2f}', 'Close': '{:.2f}',
    'Adj Close': '{:.2f}', 'Volume': '{:,.0f}', 'MACD': '{:.2f}', 'Signal': '{:.2f}'
}))
st.write(f"Data from {START_DATE} to {TODAY} ({len(data)} trading days)")

# --- Plot Raw Data with Volume ---
st.subheader("Historical Stock Prices with Volume")
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price", mode='lines', line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Open Price", mode='lines', line=dict(color="#ff7f0e", dash='dash')))
    fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name="Volume", yaxis="y2", opacity=0.3, marker=dict(color="#2ca02c")))
    fig.layout.update(
        title_text=f"{selected_stock} Stock Prices and Volume",
        xaxis_rangeslider_visible=True,
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        yaxis2=dict(title="Volume", overlaying="y", side="right"),
        hovermode="x unified",
        legend=dict(x=0, y=1.1, orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

# --- Define training and testing periods ---
train_data = data[data['Date'] <= TRAIN_END_DATE]
test_data = data[data['Date'] > TRAIN_END_DATE]

if test_data.empty:
    st.warning("No data available for the test period (2024-today). Models will not be evaluated.")
    st.stop()

# --- Prophet Model with MACD ---
with st.status("Training and evaluating Prophet model...", expanded=True) as status:
    try:
        df_train_prophet = train_data[['Date', 'Close', 'MACD']].copy().rename(columns={
            "Date": "ds", "Close": "y"
        })

        m = Prophet(
            seasonality_mode="multiplicative",
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        m.add_seasonality(name='quarterly', period=90.25, fourier_order=10)
        m.add_country_holidays(country_name='US')

        # Add MACD as regressor
        m.add_regressor('MACD')

        m.fit(df_train_prophet)

        # Future dataframe with MACD for test period
        prophet_future = test_data[['Date', 'MACD']].copy().rename(columns={"Date": "ds"})
        prophet_forecast = m.predict(prophet_future)

        # Cross-validation for better metrics
        df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days')
        df_p = performance_metrics(df_cv)
        prophet_cv_mae = df_p['mae'].mean()

        # Training metrics
        st.subheader("Performance on Training Data")
        train_forecast = m.predict(df_train_prophet[['ds', 'MACD']])
        train_metrics = {
            'MAE': mean_absolute_error(df_train_prophet['y'], train_forecast['yhat']),
            'RMSE': np.sqrt(mean_squared_error(df_train_prophet['y'], train_forecast['yhat'])),
            'MAPE': mean_absolute_percentage_error(df_train_prophet['y'] + 1e-10, train_forecast['yhat'])
        }
        train_accuracy = 100 * (1 - train_metrics['MAPE'])
        train_metrics_data = {
            'Metric': ['MAE', 'RMSE', 'MAPE', 'Accuracy'],
            'Value': [
                f"{train_metrics['MAE']:.2f}",
                f"{train_metrics['RMSE']:.2f}",
                f"{train_metrics['MAPE']:.2%}",
                f"{train_accuracy:.2f}%"
            ]
        }
        train_metrics_df = pd.DataFrame(train_metrics_data)
        st.table(train_metrics_df.set_index('Metric'))

        # Test metrics
        st.subheader("Performance on Test Data")
        prophet_metrics = {
            'MAE': mean_absolute_error(test_data['Close'], prophet_forecast['yhat']),
            'RMSE': np.sqrt(mean_squared_error(test_data['Close'], prophet_forecast['yhat'])),
            'MAPE': mean_absolute_percentage_error(test_data['Close'] + 1e-10, prophet_forecast['yhat'])
        }
        accuracy = 100 * (1 - prophet_metrics['MAPE'])
        metrics_data = {
            'Metric': ['MAE', 'RMSE', 'MAPE', 'Accuracy'],
            'Value': [
                f"{prophet_metrics['MAE']:.2f}",
                f"{prophet_metrics['RMSE']:.2f}",
                f"{prophet_metrics['MAPE']:.2%}",
                f"{accuracy:.2f}%"
            ]
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.table(metrics_df.set_index('Metric'))

        st.write(f"Cross-Validation MAE (over rolling 2-year windows): {prophet_cv_mae:.2f}")
        st.write("""
        **Metric Explanations:**
        * **MAE (Mean Absolute Error):** The average absolute difference between the predicted and actual values. A lower value is better.
        * **RMSE (Root Mean Squared Error):** Similar to MAE, but penalizes larger errors more. A lower value is better.
        * **MAPE (Mean Absolute Percentage Error):** The percentage error of the forecast. A lower percentage is better.
        * **Accuracy:** Calculated as $100 - MAPE$. While commonly used, for a regression task like this, MAPE provides a more direct measure of forecast quality.
        """)
        
        status.update(state="complete", expanded=False)
    except Exception as e:
        status.update(label=f"Error in Prophet model: {e}", state="error")
        st.error(f"Prophet model failed: {e}")
        prophet_forecast = pd.DataFrame()

# --- Prophet Plot with Historical + Actual + Predicted ---
if not prophet_forecast.empty:
    st.subheader("Prophet Model Performance")
    fig_prophet = go.Figure()

    # Historical training data
    fig_prophet.add_trace(go.Scatter(
        x=train_data['Date'], y=train_data['Close'],
        name="Historical (Train)", mode='lines',
        line=dict(color="gray")
    ))

    # Actual test data
    fig_prophet.add_trace(go.Scatter(
        x=test_data['Date'], y=test_data['Close'],
        name="Actual Price", mode='lines'
    ))

    # Prophet predictions
    fig_prophet.add_trace(go.Scatter(
        x=prophet_forecast['ds'], y=prophet_forecast['yhat'],
        name="Prophet Predicted", mode='lines'
    ))

    fig_prophet.layout.update(
        title_text=f"Prophet: Historical + Actual vs. Predicted {selected_stock} (with MACD)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=True,
        hovermode="x unified",
        legend=dict(x=0, y=1.1, orientation="h")
    )
    st.plotly_chart(fig_prophet, use_container_width=True)
