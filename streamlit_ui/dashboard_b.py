# Import required libraries
import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
from datetime import datetime
import traceback # Import traceback for detailed error info

def load_and_predict_data(csv_path):
    """
    Sends the test CSV to the Modal API endpoint and gets predictions,
    with added error handling and logging.
    """
    st.info(f"Attempting to send '{csv_path}' to the prediction API...") # Log start

    files = {'csv': open(csv_path, 'rb')}
    api_url = "https://flexible-functions-ai--sticker-sales-api-health.modal.run" # Use the correct Modal endpoint

    try:
        st.write(f"Posting data to: {api_url}")
        response = requests.post(
            api_url,
            files=files,
            timeout=90 # Increased timeout for potentially long API calls
        )

        # --- DEBUG: Show API response details ---
        st.write(f"API Response Status Code: {response.status_code}")
        # Use st.text_area for potentially long responses
        st.text_area("Raw API Response Text:", response.text, height=150)

        # --- Check for HTTP errors (e.g., 404 Not Found, 500 Server Error) ---
        response.raise_for_status() # This will raise an HTTPError if the status code indicates an error

        # --- Try to parse the JSON response ---
        try:
            predictions = response.json()
            st.write("Successfully parsed JSON response.")
            # Display a sample of what was received
            st.write(f"Parsed Predictions (type: {type(predictions)}, first 10): {str(predictions[:10]) if isinstance(predictions, list) else str(predictions)}")
        except json.JSONDecodeError:
            st.error("Error: Failed to decode JSON from the API response.")
            st.error("The API might have returned HTML (like an error page) or non-JSON text.")
            # Return an empty DataFrame to prevent downstream errors
            return pd.DataFrame()

        # --- Validate the predictions format ---
        # Check if the response is actually a list, as expected for direct assignment
        if not isinstance(predictions, list):
            st.error(f"Error: API response is not a list as expected. Received type: {type(predictions)}")
            st.warning("Please check the API's return format. If predictions are nested (e.g., {'predictions': [...]}), adjust the code here.")
            # Example adjustment if nested:
            # if isinstance(predictions, dict) and 'predictions' in predictions:
            #     predictions = predictions['predictions']
            # else:
            #     return pd.DataFrame() # Return empty if still not a list
            # For now, assume it should be a list and return empty if not:
            return pd.DataFrame()

        # --- Load the original test data ---
        test_df = pd.read_csv(csv_path)
        st.write(f"Loaded test CSV with {len(test_df)} rows.")

        # --- Check for length mismatch ---
        if len(predictions) != len(test_df):
            st.error(f"Error: Mismatch between number of predictions received ({len(predictions)}) and number of rows in CSV ({len(test_df)}).")
            st.warning("Cannot reliably assign predictions. Check API logic or the uploaded CSV.")
            # Return the DataFrame without predictions, or an empty one
            # Let's return it without predictions for now, so user can see the input data
            test_df['predicted_sales'] = None # Add the column but fill with None
            return test_df
            # return pd.DataFrame() # Alternative: return empty

        # --- Add predictions to the dataframe ---
        test_df['predicted_sales'] = predictions
        st.success("Successfully added 'predicted_sales' column.")

        # Check if predictions are actually numbers (handle potential strings, None, etc.)
        try:
            test_df['predicted_sales'] = pd.to_numeric(test_df['predicted_sales'], errors='coerce')
            if test_df['predicted_sales'].isnull().any():
                 st.warning("Some predicted sales values were non-numeric and have been set to NaN.")
        except Exception as e:
            st.warning(f"Could not convert predicted_sales column to numeric: {e}")


        # --- Convert date column to datetime ---
        try:
            test_df['date'] = pd.to_datetime(test_df['date'])
        except Exception as e:
            st.error(f"Error converting 'date' column to datetime: {e}")
            st.warning("Date conversion failed. Date filtering and time-based plots might not work correctly.")
            # Proceeding without date conversion if it fails

        return test_df

    except requests.exceptions.RequestException as e:
        st.error(f"Network or Request Error contacting the API: {e}")
        st.error(traceback.format_exc()) # Show detailed traceback
        return pd.DataFrame() # Return empty DataFrame
    except requests.exceptions.HTTPError as e:
        # Error already logged above from response.text, but add context
        st.error(f"HTTP Error from API: Status Code {e.response.status_code}")
        return pd.DataFrame() # Return empty DataFrame
    except Exception as e:
        st.error(f"An unexpected error occurred in load_and_predict_data: {e}")
        st.error(traceback.format_exc()) # Show detailed traceback
        return pd.DataFrame() # Return empty DataFrame
    finally:
        # Ensure the file handle is closed
        if 'files' in locals() and 'csv' in files:
             files['csv'].close()


def create_dashboard():
    """
    Creates the Streamlit dashboard with enhanced filters, KPI cards, and visualizations
    """
    st.title("Sales Prediction Dashboard")

    # Add custom CSS for dark theme cards (keep as is)
    st.markdown("""
        <style>
        /* ... your CSS here ... */
        </style>
    """, unsafe_allow_html=True)

    # File uploader for the test CSV
    uploaded_file = st.file_uploader("Upload test CSV file", type=['csv'])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_csv_path = 'temp_test.csv'
        try:
            with open(temp_csv_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
        except Exception as e:
            st.error(f"Failed to save uploaded file: {e}")
            return # Stop if we can't save the file

        # Load data and get predictions using the robust function
        df = load_and_predict_data(temp_csv_path)

        # --- Check if DataFrame is valid and has predictions ---
        if df.empty:
            st.warning("Could not load data or retrieve predictions. Cannot display dashboard.")
            # Optionally display the raw uploaded data if needed for debugging
            try:
                raw_df = pd.read_csv(temp_csv_path)
                st.subheader("Uploaded Data (for debugging)")
                st.dataframe(raw_df.head())
            except Exception as e:
                st.error(f"Could not read the uploaded CSV for debugging: {e}")
            return # Stop processing

        if 'predicted_sales' not in df.columns:
             st.error("Processing Error: 'predicted_sales' column is missing after API call.")
             st.subheader("Data Received (Partial)")
             st.dataframe(df.head())
             return # Stop processing

        if df['predicted_sales'].isnull().all():
             st.warning("The 'predicted_sales' column exists but contains no valid prediction values (all null/NaN).")
             st.info("Please check the API Response Text above and ensure the API returns valid numeric predictions.")
             # Decide whether to proceed. Let's show the data but maybe skip plots/KPIs that rely on numbers.
             # For now, we'll try to continue, but add checks below.


        # --- Sidebar Filters ---
        st.sidebar.header("Filters")

        # Ensure columns exist before creating filters
        countries = ['All'] + sorted(df['country'].unique().tolist()) if 'country' in df.columns else ['All']
        selected_country = st.sidebar.selectbox('Select Country', countries)

        stores = ['All'] + sorted(df['store'].unique().tolist()) if 'store' in df.columns else ['All']
        selected_store = st.sidebar.selectbox('Select Store', stores)

        products = ['All'] + sorted(df['product'].unique().tolist()) if 'product' in df.columns else ['All']
        selected_product = st.sidebar.selectbox('Select Product', products)

        # Time period filter - Check if 'date' column exists and is datetime
        time_periods = { 'All Time': None, 'Last Month': 30, 'Last 3 Months': 90, 'Last Year': 365 }
        selected_period = 'All Time' # Default
        if 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
             selected_period = st.sidebar.selectbox('Select Time Period', list(time_periods.keys()))
        else:
            st.sidebar.warning("Date column missing or not in datetime format. Time filter disabled.")


        # --- Apply Filters ---
        filtered_df = df.copy()

        # Apply time filter safely
        if 'date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['date']) and time_periods[selected_period]:
            try:
                max_date = filtered_df['date'].max()
                cutoff_date = max_date - pd.Timedelta(days=time_periods[selected_period])
                filtered_df = filtered_df[filtered_df['date'] >= cutoff_date]
            except Exception as e:
                 st.warning(f"Could not apply time filter: {e}")

        if selected_country != 'All' and 'country' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['country'] == selected_country]
        if selected_store != 'All' and 'store' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['store'] == selected_store]
        if selected_product != 'All' and 'product' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['product'] == selected_product]

        # --- Check if data remains after filtering ---
        if filtered_df.empty:
            st.warning("No data matches the selected filters.")
            return

        # --- Calculate Metrics and KPIs (with checks) ---
        total_sales = 0
        avg_daily_sales = 0
        sales_change = 0
        top_store = "N/A"
        store_sales = 0
        top_product = "N/A"
        product_sales = 0

        # Check if predicted_sales column is numeric and not all NaN before calculating
        if 'predicted_sales' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['predicted_sales']) and not filtered_df['predicted_sales'].isnull().all():
            try:
                total_sales = filtered_df['predicted_sales'].sum()

                # Calculate daily sales only if date column is valid
                if 'date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['date']):
                    daily_sum = filtered_df.groupby('date')['predicted_sales'].sum()
                    if not daily_sum.empty:
                         avg_daily_sales = daily_sum.mean()

                # Calculate period-over-period change (simplified, needs robust date handling)
                # ... (Your existing sales_change logic can go here, potentially wrapped in try-except)

                # Calculate top performers
                if 'store' in filtered_df.columns:
                     store_groups = filtered_df.groupby('store')['predicted_sales'].sum()
                     if not store_groups.empty:
                         top_store = store_groups.idxmax()
                         store_sales = store_groups.max()

                if 'product' in filtered_df.columns:
                     product_groups = filtered_df.groupby('product')['predicted_sales'].sum()
                     if not product_groups.empty:
                         top_product = product_groups.idxmax()
                         product_sales = product_groups.max()

            except Exception as e:
                 st.error(f"Error calculating KPIs: {e}")
        else:
             st.warning("Predicted sales data is missing or non-numeric. KPIs cannot be calculated accurately.")


        # Display KPI cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
             # ... (display total_sales, sales_change)
             st.metric("Total Predicted Sales", f"${total_sales:,.0f}", f"{sales_change:+.1f}%" if sales_change != 0 else None)
        with col2:
             # ... (display avg_daily_sales)
             st.metric("Average Daily Sales", f"${avg_daily_sales:,.0f}")
        with col3:
             # ... (display top_store, store_sales)
             st.metric(f"Top Store: {top_store}", f"${store_sales:,.0f}")
        with col4:
             # ... (display top_product, product_sales)
              st.metric(f"Top Product: {top_product}", f"${product_sales:,.0f}")


        # --- Create Plot (with checks) ---
        if 'date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['date']) and \
           'predicted_sales' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['predicted_sales']) and \
           not filtered_df['predicted_sales'].isnull().all():
            try:
                daily_sales_df = filtered_df.groupby('date')['predicted_sales'].sum().reset_index()

                if not daily_sales_df.empty:
                    fig = px.line(
                        daily_sales_df, x='date', y='predicted_sales',
                        title='Predicted Daily Sales Over Time'
                    )
                    fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    # Add trend line etc.
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No daily sales data available to plot for the selected filters.")
            except Exception as e:
                 st.error(f"Error creating plot: {e}")
        else:
            st.warning("Cannot create plot. Requires valid 'date' and numeric 'predicted_sales' columns.")


        # --- Display Detailed Data View ---
        st.subheader("Detailed Data View (Filtered)")
        st.dataframe(
            filtered_df.sort_values('date') if 'date' in filtered_df.columns else filtered_df,
            hide_index=True
        )


if __name__ == "__main__":
    # Set page configuration at the very beginning
    st.set_page_config(
        page_title="Sales Prediction Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    create_dashboard()