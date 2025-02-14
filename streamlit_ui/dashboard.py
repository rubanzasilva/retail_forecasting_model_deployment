import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
from datetime import datetime

def load_and_predict_data(csv_path):
    """
    Sends the test CSV to the BentoML API endpoint and gets predictions
    """
    # Making a POST request to the BentoML predict_csv endpoint
    files = {'csv': open(csv_path, 'rb')}
    response = requests.post(
        #"http://localhost:3000/predict_csv",
        "https://sticker-sales-predictor-63072676.mt-guc1.bentoml.ai/predict_csv",
        files=files
    )
    predictions = response.json()
    
    # Load the original test data
    test_df = pd.read_csv(csv_path)
    
    # Add predictions to the dataframe
    test_df['predicted_sales'] = predictions
    
    # Convert date column to datetime
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    return test_df

def create_dashboard():
    """
    Creates the Streamlit dashboard with filters and visualizations
    """
    st.title("Sales Prediction Dashboard")
    
    # File uploader for the test CSV
    uploaded_file = st.file_uploader("Upload test CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open('temp_test.csv', 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # Load data and get predictions
        df = load_and_predict_data('temp_test.csv')
        
        # Creating filters in a sidebar
        st.sidebar.header("Filters")
        
        # Country filter
        countries = ['All'] + sorted(df['country'].unique().tolist())
        selected_country = st.sidebar.selectbox('Select Country', countries)
        
        # Store filter
        stores = ['All'] + sorted(df['store'].unique().tolist())
        selected_store = st.sidebar.selectbox('Select Store', stores)
        
        # Product filter
        products = ['All'] + sorted(df['product'].unique().tolist())
        selected_product = st.sidebar.selectbox('Select Product', products)
        
        # Apply filters
        filtered_df = df.copy()
        if selected_country != 'All':
            filtered_df = filtered_df[filtered_df['country'] == selected_country]
        if selected_store != 'All':
            filtered_df = filtered_df[filtered_df['store'] == selected_store]
        if selected_product != 'All':
            filtered_df = filtered_df[filtered_df['product'] == selected_product]
        
        # Group by date and calculate daily total predicted sales
        daily_sales = filtered_df.groupby('date')['predicted_sales'].sum().reset_index()
        
        # Create the line chart using Plotly
        fig = px.line(
            daily_sales,
            x='date',
            y='predicted_sales',
            title='Predicted Daily Sales Over Time'
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Predicted Sales",
            hovermode='x'
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Predicted Sales",
                f"{filtered_df['predicted_sales'].sum():,.0f}"
            )
        
        with col2:
            st.metric(
                "Average Daily Sales",
                f"{filtered_df.groupby('date')['predicted_sales'].sum().mean():,.0f}"
            )
            
        with col3:
            st.metric(
                "Number of Predictions",
                f"{len(filtered_df):,}"
            )
        
        # Display the filtered dataframe
        st.subheader("Detailed Data View")
        st.dataframe(
            filtered_df.sort_values('date'),
            hide_index=True
        )

if __name__ == "__main__":
    st.set_page_config(
        page_title="Sales Prediction Dashboard",
        page_icon="📊",
        layout="wide"
    )
    create_dashboard()