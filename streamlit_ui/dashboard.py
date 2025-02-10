import streamlit as st
import pandas as pd
import requests  # To make API calls
import json

# Set page configuration
st.set_page_config(
    page_title="Sticker Sales Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Define the API endpoint ---
API_URL = "http://localhost:3000/predict"  # Replace with your actual Bentoml API endpoint
CSV_API_URL = "http://localhost:3000/predict_csv"

# --- Sidebar for Filters ---
st.sidebar.header("Filters")
countries = ["Canada", "USA", "UK", "Germany", "France"]  # Example countries - replace with your actual list
stores = ["Discount Stickers", "Premium Stickers", "Wholesale Stickers"]  # Example stores - replace with your actual list
products = ["Holographic Goose", "Sparkly Unicorn", "Matte Moose"]  # Example products - replace with your actual list

selected_country = st.sidebar.selectbox("Country", options=countries)
selected_store = st.sidebar.selectbox("Store", options=stores)
selected_product = st.sidebar.selectbox("Product", options=products)

# --- Function to make API requests ---
def get_prediction(date, country, store, product):
    """
    Makes a POST request to the prediction API.
    """
    data = {
        "data": [
            {
                "id": "0",  # Placeholder - the API might not need this
                "date": date,
                "country": country,
                "store": store,
                "product": product
            }
        ]
    }
    try:
        response = requests.post(API_URL, json=data)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {e}")
        return None

def get_prediction_from_csv(csv_file):
    """
    Makes a POST request to the batch prediction API.
    """
    try:
        files = {"csv": csv_file}
        response = requests.post(CSV_API_URL, files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling CSV API: {e}")
        return None

# --- Main Content Area ---
st.title("Sticker Sales Forecast Dashboard")

# Date Range Selection
start_date = st.date_input("Start Date", value=pd.to_datetime("2017-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2017-01-10"))

if start_date <= end_date:
    dates = pd.date_range(start_date, end_date)
else:
    st.error("Error: End date must fall after start date.")
    dates = pd.date_range(start = pd.to_datetime("2017-01-01"),end = pd.to_datetime("2016-12-31"))

# --- Fetch Predictions and Prepare Data ---
if not dates.empty:
    all_predictions = []
    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        prediction_data = get_prediction(date_str, selected_country, selected_store, selected_product)
        if prediction_data:
            try:
                num_sold = prediction_data["num_sold"][0]  # Extract the prediction
                all_predictions.append({"date": date_str, "num_sold": num_sold})
            except (KeyError, IndexError) as e:
                st.error(f"Error extracting data from API response: {e}")
                all_predictions.append({"date": date_str, "num_sold": None})  # Append None to maintain date sequence
        else:
            all_predictions.append({"date": date_str, "num_sold": None})  # Ensure data point exists even if API fails

    # Create DataFrame
    df = pd.DataFrame(all_predictions)
    df['date'] = pd.to_datetime(df['date'])  # Ensure correct datetime format
    df = df.set_index('date')

    # --- Display the chart ---
    st.header("Sales Forecast")
    if df['num_sold'].isnull().all():
        st.warning("No sales data to display for the selected filters and date range.")
    else:
        st.line_chart(df)

# CSV Upload and Batch Prediction
st.header("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_predictions = get_prediction_from_csv(uploaded_file)
    if batch_predictions:
        batch_df = pd.DataFrame(batch_predictions)
        batch_df['date'] = pd.to_datetime(batch_df['date'])
        batch_df = batch_df.set_index('date')

        st.subheader("Batch Prediction Results")
        st.line_chart(batch_df)
    else:
        st.error("Failed to process the uploaded CSV file.")
