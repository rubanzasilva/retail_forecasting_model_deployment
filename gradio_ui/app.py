import gradio as gr
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta


def generate_table_data(df, predictions):
    """
    Convert DataFrame and predictions into a format suitable for display
   
    Args:
        df: Input DataFrame with sales data
        predictions: List of prediction values
   
    Returns:
        list: List of rows formatted for display
    """
    df['date'] = pd.to_datetime(df['date'])
   
    table_data = []
    for index, row in df.iterrows():
        table_row = [
            row["date"].strftime("%Y-%m-%d"),
            row["country"],
            row["store"],
            row["product"],
            float(predictions[index])
        ]
        table_data.append(table_row)
   
    return table_data


def create_sales_plot(df, start_date, end_date):
    """
    Create a Plotly figure for sales visualization
   
    Args:
        df: DataFrame with sales data
        start_date: Start date for filtering
        end_date: End date for filtering
   
    Returns:
        plotly.Figure: Interactive sales visualization
    """
    # Convert dates and filter data
    df['date'] = pd.to_datetime(df['date'])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    mask = (df['date'] >= start) & (df['date'] <= end)
    filtered_df = df[mask]
   
    # Create plot using Plotly Express
    fig = px.line(
        filtered_df,
        x='date',
        y='predicted_sales',
        color='store',
        title='Predicted Sales Over Time',
        labels={
            'date': 'Date',
            'predicted_sales': 'Predicted Sales',
            'store': 'Store'
        }
    )
   
    # Customize layout
    fig.update_layout(
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
   
    return fig


def predict_and_visualize(file_obj, start_date, end_date):
    """
    Process uploaded file and create visualizations
   
    Args:
        file_obj: Uploaded CSV file
        start_date: Start date for filtering
        end_date: End date for filtering
   
    Returns:
        tuple: (file_path, DataFrame, Plot, error_message)
    """
    try:
        if file_obj is None:
            return None, gr.DataFrame(), None, "Please upload a CSV file"


        # Read the uploaded CSV file
        df = pd.read_csv(file_obj.name)
       
        # Make prediction request to BentoML service
        files = {'csv': ('input.csv', open(file_obj.name, 'rb'), 'text/csv')}
        response = requests.post(
            "http://localhost:3000/predict_csv",
            files=files
        )


        if response.status_code == 200:
            predictions = response.json()
           
            # Create table data
            table_data = generate_table_data(df, predictions)
           
            # Prepare data for plotting
            df['date'] = pd.to_datetime(df['date'])
            df['predicted_sales'] = predictions
           
            # Create the visualization
            fig = create_sales_plot(df, start_date, end_date)
           
            # Save results to CSV
            results_path = "predictions_results.csv"
            df.to_csv(results_path, index=False)


            return (
                results_path,
                gr.DataFrame(
                    headers=["Date", "Country", "Store", "Product", "Predicted Sales"],
                    value=table_data
                ),
                fig,
                None
            )
        else:
            return None, gr.DataFrame(), None, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, gr.DataFrame(), None, f"Error processing file: {str(e)}"


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Sticker Sales Predictions Dashboard")
   
    # Error message display
    error_message = gr.Textbox(label="Status", interactive=False, visible=False)
   
    with gr.Row():
        file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
   
    with gr.Row():
        start_date = gr.Textbox(
            label="Start Date (YYYY-MM-DD)",
            value=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        )
        end_date = gr.Textbox(
            label="End Date (YYYY-MM-DD)",
            value=datetime.now().strftime("%Y-%m-%d")
        )
        predict_button = gr.Button("Predict & Visualize")
   
    with gr.Row():
        predictions_table = gr.DataFrame(
            headers=["Date", "Country", "Store", "Product", "Predicted Sales"],
            label="Sticker Sales Predictions"
        )
   
    with gr.Row():
        sales_plot = gr.Plot(label="Sales Trends")
   
    # Update both table and plot when button is clicked
    predict_button.click(
        predict_and_visualize,
        inputs=[file_input, start_date, end_date],
        outputs=[file_input, predictions_table, sales_plot, error_message]
    )


# Launch the interface
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
