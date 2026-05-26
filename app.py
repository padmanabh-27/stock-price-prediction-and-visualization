import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Stock Prediction",
    page_icon="📈",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.metric-card {
    background-color:#1e222d;
    padding:15px;
    border-radius:12px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("📈 AI Stock Trend Prediction")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙ Stock Settings")

stock_list = {
    "ITC": "ITC.NS",
    "TCS": "TCS.NS",
    "Reliance": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS"
}

selected_stock = st.sidebar.selectbox(
    "Choose Stock",
    list(stock_list.keys())
)

default_ticker = stock_list[selected_stock]

user_input = st.sidebar.text_input(
    "Or Enter Custom Ticker",
    default_ticker
)

start = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime("2020-01-01")
)

end = st.sidebar.date_input(
    "End Date",
    pd.to_datetime("today")
)

# ---------------- LOAD DATA ----------------
with st.spinner("📡 Fetching stock data..."):

    stock = yf.Ticker(user_input)
    ts = stock.history(start=start, end=end)
    df = pd.DataFrame(ts)

if df.empty:
    st.error("❌ Invalid ticker or no stock data found.")
    st.stop()

# ---------------- METRICS ----------------
current_price = round(df.Close.iloc[-1], 2)
high_52 = round(df.High.max(), 2)
low_52 = round(df.Low.min(), 2)

col1, col2, col3 = st.columns(3)

col1.metric("💰 Current Price", f"₹ {current_price}")
col2.metric("📈 Highest", f"₹ {high_52}")
col3.metric("📉 Lowest", f"₹ {low_52}")

# ---------------- DATA ----------------
st.subheader("📊 Stock Data Summary")
st.write(df.describe())

# ---------------- CHART 1 ----------------
# ---------------- CHART 1 ----------------
st.subheader("Closing Price vs Time")

plt.style.use('dark_background')

fig1, ax = plt.subplots(figsize=(12,6))

ax.plot(df.Close, color='cyan', linewidth=2)

ax.set_xlabel("Date", color='white', fontsize=12)
ax.set_ylabel("Price (₹)", color='white', fontsize=12)

ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

ax.grid(True, alpha=0.3)

st.pyplot(fig1)

# ---------------- CHART 2 ----------------
st.subheader("Closing Price with 100 MA")

ma100 = df.Close.rolling(100).mean()

fig2 = plt.figure(figsize=(12,6))
plt.style.use('dark_background')
plt.plot(df.Close, label="Closing Price", color='cyan')
plt.plot(ma100, label="100 MA", color='orange')
plt.legend()
plt.grid(True)
st.pyplot(fig2)

# ---------------- CHART 3 ----------------
st.subheader("Closing Price with 100 MA & 200 MA")

ma200 = df.Close.rolling(200).mean()

fig3 = plt.figure(figsize=(12,6))
plt.style.use('dark_background')
plt.plot(df.Close, label="Close", color='cyan')
plt.plot(ma100, label="100 MA", color='orange')
plt.plot(ma200, label="200 MA", color='magenta')
plt.legend()
plt.grid(True)
st.pyplot(fig3)

# ---------------- TRAIN TEST ----------------
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# ---------------- LOAD MODEL ----------------
model = load_model('keras_model.h5')

# ---------------- TESTING ----------------
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test = np.array(x_test)
y_test = np.array(y_test)

# ---------------- PREDICT ----------------
with st.spinner("🧠 AI is predicting..."):
    y_predicted = model.predict(x_test)

scale_factor = 1/scaler.scale_[0]

y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# ---------------- NEXT DAY PREDICTION ----------------
st.subheader("🔮 Next Day Prediction")

# Prepare last 100 days for prediction
last_100 = input_data[-100:]
last_100 = np.reshape(last_100, (1, 100, 1))

# Predict next day
next_day = model.predict(last_100)

# Reverse scaling
next_day_price = next_day[0][0] * scale_factor

# Dates
last_date = df.index[-1]
next_date = last_date + pd.offsets.BDay(1)

# Display
st.metric(
    f"Predicted Price ({next_date.strftime('%d %b %Y')})",
    f"₹ {round(next_day_price,2)}"
)

st.caption(
    f"Based on last market data: "
    f"{last_date.strftime('%d %b %Y')}"
)

# ---------------- 7 DAY FORECAST ----------------
st.subheader("📅 Next 7 Trading Days Forecast")

future_days = 7
future_predictions = []

# Last 100 scaled values
temp_input = input_data[-100:].flatten().tolist()

for i in range(future_days):

    x_input = np.array(temp_input[-100:])
    x_input = x_input.reshape(1,100,1)

    yhat = model.predict(x_input, verbose=0)

    # Save prediction
    temp_input.append(yhat[0][0])

    # Reverse scaling
    predicted_price = yhat[0][0] * scale_factor
    future_predictions.append(predicted_price)

# Future business dates
future_dates = pd.bdate_range(
    start=df.index[-1] + pd.offsets.BDay(1),
    periods=future_days
)

# Create dataframe
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": np.round(future_predictions,2)
})

st.dataframe(forecast_df)

# ---------------- 7 DAY FORECAST GRAPH ----------------
import plotly.graph_objects as go

fig_future = go.Figure()

fig_future.add_trace(
    go.Scatter(
        x=future_dates,
        y=future_predictions,
        mode='lines+markers',
        name='7 Day Forecast',
        line=dict(color='lime', width=3)
    )
)

fig_future.update_layout(
    template='plotly_dark',
    title='📈 Next 7 Day Forecast',
    xaxis_title='Date',
    yaxis_title='Predicted Price (₹)',
    hovermode='x unified'
)

st.plotly_chart(
    fig_future,
    use_container_width=True
)

# ---------------- FINAL INTERACTIVE GRAPH ----------------
import plotly.graph_objects as go

st.subheader("📉 Prediction vs Original")

dates = df.index[-len(y_test):]

fig4 = go.Figure()

fig4.add_trace(
    go.Scatter(
        x=dates,
        y=y_test,
        mode='lines',
        name='Original Price',
        line=dict(color='cyan', width=2),

        hovertemplate=
        "<b>Date:</b> %{x|%d %b %Y}<br>" +
        "<b>Price:</b> ₹ %{y:.2f}<extra></extra>"
    )
)

fig4.add_trace(
    go.Scatter(
        x=dates,
        y=y_predicted.flatten(),
        mode='lines',
        name='Predicted Price',
        line=dict(color='red', width=2),

        hovertemplate=
        "<b>Date:</b> %{x|%d %b %Y}<br>" +
        "<b>Predicted:</b> ₹ %{y:.2f}<extra></extra>"
    )
)

fig4.update_layout(
    template='plotly_dark',
    xaxis_title='Date',
    yaxis_title='Price (₹)',
    hovermode='x unified',
    height=600,

    xaxis=dict(
        tickformat="%b %Y",
        dtick="M1",
        tickangle=-45,
        showgrid=True
    )
)

st.plotly_chart(fig4, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + LSTM + TensorFlow")