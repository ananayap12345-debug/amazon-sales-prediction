import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("Amazon Sales Dashboard & Prediction")

# ==========================
# SINGLE PREDICTION SECTION
# ==========================

st.header("Predict Order Status")

fulfilment = st.selectbox(
    "Fulfilment",
    ["Amazon", "Merchant"]
)

sales_channel = st.selectbox(
    "Sales Channel",
    ["Amazon.in"]
)

service_level = st.selectbox(
    "Service Level",
    ["Standard", "Expedited"]
)

category = st.selectbox(
    "Category",
    ["T-shirt", "Shirt", "Blazer", "Trousers"]
)

size = st.selectbox(
    "Size",
    ["S", "M", "L", "XL", "XXL", "3XL"]
)

currency = st.selectbox(
    "Currency",
    ["INR"]
)

amount = st.number_input(
    "Amount",
    min_value=0.0
)

ship_city = st.text_input("City")
ship_state = st.text_input("State")
ship_country = st.text_input("Country")

b2b = st.selectbox(
    "B2B",
    [True, False]
)

if st.button("Predict Status"):

    sample = pd.DataFrame({
        "fulfilment": [fulfilment],
        "sales channel": [sales_channel],
        "ship-service-level": [service_level],
        "category": [category],
        "size": [size],
        "currency": [currency],
        "amount": [amount],
        "ship-city": [ship_city],
        "ship-state": [ship_state],
        "ship-country": [ship_country],
        "b2b": [b2b]
    })

    prediction = model.predict(sample)

    st.success(f"Predicted Status: {prediction[0]}")

# ==========================
# DASHBOARD SECTION
# ==========================

st.sidebar.title("Upload Sales Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.header("Dataset Preview")
    st.dataframe(df.head())

    st.header("Sales Summary")

    if "Amount" in df.columns:
        st.metric(
            "Total Sales",
            f"₹ {df['Amount'].sum():,.2f}"
        )

    if "Category" in df.columns:
        st.subheader("Category Distribution")
        st.bar_chart(df["Category"].value_counts())