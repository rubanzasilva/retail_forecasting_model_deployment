import gradio as gr
import requests
import pandas as pd

def generate_table_data(df, predictions):
    """
    Generates the table data for the Gradio DataFrame component.
    
    This function takes the original data frame `df` and the list of predicted sticker sales `predictions`,
    and creates a list of tuples representing the table rows. Each tuple contains the date, country, store,
    product, and predicted sales information for a single row.
    
    The function iterates through the rows of the data frame, extracting the relevant information and creating
    a tuple for each row. The tuples are added to the `table_data` list, which is ultimately returned.
    
    Parameters:
    df (pandas.DataFrame): The original data frame.
    predictions (list): The list of predicted sticker sales.
    
    Returns:
    list[tuple]: A list of tuples representing the table rows.
    """
    table_data = []
    for index, row in df.iterrows():
        table_row = (
            row["date"],
            row["country"],
            row["store"],
            row["product"],
            predictions[index]
        )
        table_data.append(table_row)
    return table_data

def predict_csv_file(file_obj):
    """
    Processes the uploaded CSV file, makes predictions using the BentoML service,
    and returns the predictions as a Gradio DataFrame.
    
    This function first checks if a file was uploaded. If no file was uploaded, it returns a message
    asking the user to upload a CSV file.
    
    If a file was uploaded, the function reads the CSV file using pandas. It then creates a multipart
    form-data payload and sends a POST request to the BentoML service to obtain the predicted sticker sales.
    
    After receiving the predictions, the function calls the `generate_table_data` function to create the
    table data. The function then returns the path to the saved predictions CSV file and the Gradio DataFrame
    component containing the table data.
    
    Parameters:
    file_obj (gradio.files.UploadedFile): The uploaded CSV file.
    
    Returns:
    (str, gradio.components.DataFrame): The path to the predictions CSV file and the predictions DataFrame.
    """
    try:
        # Check if file was uploaded
        if file_obj is None:
            return None, "Please upload a CSV file"

        # Read the uploaded CSV file directly using pandas
        df = pd.read_csv(file_obj.name)

        # Create multipart form-data
        files = {'csv': ('input.csv', open(file_obj.name, 'rb'), 'text/csv')}

        # Make prediction request to BentoML service
        response = requests.post(
            "http://localhost:3000/predict_csv",
            files=files
        )

        if response.status_code == 200:
            predictions = response.json()
            table_data = generate_table_data(df, predictions)

            # Save results to a CSV file
            results_path = "predictions_results.csv"
            df['num_sold_predicted'] = predictions
            df.to_csv(results_path, index=False)

            return results_path, gr.DataFrame(table_data, headers=["Date", "Country", "Store", "Product", "Predicted Sales"])
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

with gr.Blocks() as demo:
    """
    The Gradio interface for the sticker sales prediction application.
    
    This interface includes a file input for uploading a CSV file and a button to initiate the prediction
    process. The results are displayed in a Gradio DataFrame component, showing the date, country, store,
    product, and predicted sticker sales.
    
    The goal of this interface is to provide a clear and user-friendly way for the human to visualize the
    predicted sticker sales. The thorough explanations and step-by-step approach aims to help the human
    deeply understand the functionality and logic behind the application.
    """
    with gr.Row():
        file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
        predict_button = gr.Button("Predict")
    with gr.Row():
        predictions_table = gr.DataFrame(label="Sticker Sales Predictions")

    predict_button.click(predict_csv_file, inputs=file_input, outputs=[file_input, predictions_table])

demo.launch(server_name="0.0.0.0", server_port=7861)