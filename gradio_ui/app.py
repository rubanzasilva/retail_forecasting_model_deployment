import gradio as gr
import requests
import pandas as pd
import plotly.express as px
import tempfile
import os
import json

def clean_data(df):
    """
    Cleans and preprocesses the input data to ensure consistent formatting.
    """
    # Clean column names - remove spaces and standardize
    df.columns = df.columns.str.strip().str.lower()
    
    # Handle potential line breaks in string columns
    string_columns = ['country', 'store', 'product']
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].str.replace('\n', ' ').str.strip()
    
    # Ensure date is in correct format
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    return df

def create_sales_plot(data, predictions):
    """
    Creates an interactive line plot of sales predictions over time using plotly.
    """
    # Combine data and predictions
    plot_df = data.copy()
    plot_df['Predicted_Sales'] = predictions
    
    # Convert date to datetime for plotting
    plot_df['date'] = pd.to_datetime(plot_df['date'])
    
    # Create the main line plot
    fig = px.line(
        plot_df,
        x='date',
        y='Predicted_Sales',
        color='product',  # Color lines by product
        title='Predicted Sticker Sales Over Time',
        labels={
            'Predicted_Sales': 'Predicted Number of Sales',
            'date': 'Date',
            'product': 'Product'
        }
    )
    
    # Enhance the layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Predicted Sales",
        hovermode='x unified',
        template='plotly_white',
        legend_title="Products",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add hover data
    fig.update_traces(
        hovertemplate="<br>".join([
            "Date: %{x|%Y-%m-%d}",
            "Sales: %{y:,.0f}",
            "Product: %{customdata[0]}",
            "Store: %{customdata[1]}",
            "Country: %{customdata[2]}"
        ])
    )
    
    return fig

def predict_single_record(date, country, store, product):
    """
    Makes a prediction for a single record using the form input.
    """
    try:
        # Create input data structure
        data = {
            "data": [{
                "date": date,
                "country": country,
                "store": store,
                "product": product
            }]
        }
        
        # Make prediction request to BentoML service
        response = requests.post(
            "http://localhost:3000/predict",  # Update with your BentoML service URL
            json=data,
            headers={"content-type": "application/json"}
        )
        
        if response.status_code == 200:
            prediction = response.json()
            # Create a single-row DataFrame for plotting
            df = pd.DataFrame([data["data"][0]])
            return create_sales_plot(df, prediction)
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.RequestException as e:
        return f"Error connecting to service: {str(e)}"

def predict_csv_file(file_obj):
    """
    Makes predictions for multiple records using CSV input.
    """
    try:
        # Check if file was uploaded
        if file_obj is None:
            return "Please upload a CSV file"
            
        # Read the uploaded CSV file and clean the data
        df = pd.read_csv(file_obj.name)
        df = clean_data(df)
        
        # Create multipart form-data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
            df.to_csv(tmp.name, index=False)
            files = {
                'csv': ('input.csv', open(tmp.name, 'rb'), 'text/csv')
            }
            
            # Make prediction request to BentoML service
            response = requests.post(
                "http://localhost:3000/predict_csv",  # Update with your BentoML service URL
                files=files
            )
        
        # Clean up temporary file
        os.unlink(tmp.name)
        
        if response.status_code == 200:
            predictions = response.json()
            
            # Create and return the plot
            return create_sales_plot(df, predictions)
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error processing file: {str(e)}"

# Create the Gradio interface with tabs
with gr.Blocks() as demo:
    gr.Markdown("# Sticker Sales Forecasting")
    
    with gr.Tabs():
        # Single Record Prediction Tab
        with gr.TabItem("Single Prediction"):
            with gr.Row():
                with gr.Column():
                    date_input = gr.Textbox(
                        label="Date (YYYY-MM-DD)",
                        placeholder="2017-01-01"
                    )
                    country_input = gr.Dropdown(
                        choices=["Canada"],  # Add more countries as needed
                        label="Country"
                    )
                    store_input = gr.Dropdown(
                        choices=["Discount Stickers"],  # Add more stores as needed
                        label="Store"
                    )
                    product_input = gr.Dropdown(
                        choices=[
                            "Holographic Goose",
                            "Kaggle",
                            "Kaggle Tiers",
                            "Kerneler",
                            "Kerneler Dark Mode"
                        ],
                        label="Product"
                    )
                    
                    predict_btn = gr.Button("Predict")
                
                with gr.Column():
                    plot_output1 = gr.Plot(label="Sales Prediction")
            
            predict_btn.click(
                fn=predict_single_record,
                inputs=[date_input, country_input, store_input, product_input],
                outputs=plot_output1
            )
        
        # Batch Prediction Tab
        with gr.TabItem("Batch Prediction"):
            gr.Markdown("""
            ### CSV File Format Requirements
            
            Upload a CSV file containing multiple records for batch prediction.
            The CSV should contain these columns:
            - date (YYYY-MM-DD)
            - country
            - store
            - product
            
            Example row:
            ```
            date,country,store,product
            2017-01-01,Canada,Discount Stickers,Holographic Goose
            ```
            """)
            
            file_input = gr.File(
                label="Upload CSV File",
                file_types=[".csv"]
            )
            plot_output2 = gr.Plot(label="Sales Predictions")
            
            file_input.change(
                fn=predict_csv_file,
                inputs=file_input,
                outputs=plot_output2
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)