# Importing the necessary libraries
import os
import json
import numpy as np
import pandas as pd

# Function to create the paths
def create_paths(input_folder, output_folder):
    input_path = os.path.abspath(input_folder)
    output_path = os.path.abspath(output_folder)
    return input_path, output_path

# Function to create output folders
def create_folders(input_path, output_path):
    folders = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))]

    for folder in folders:
        output_folder_path = os.path.join(output_path, folder)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
            print(f"Created folder: {output_folder_path}")
        else:
            print(f"Folder already exists: {output_folder_path}")
# =============================================================================

# Function to create DataFrame with coordinates from .json files of every patient
def create_dataframe(input_path):
    # Initialize a list to collect each patient's data
    data = []

    # Get folder names (e.g., patient folders)
    folder_names = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))]
    
    for folder in folder_names:
        folder_path = os.path.join(input_path, folder)
        row_data = {'Patient': folder}  # Dictionary to store data for each patient

        # Loop through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                # Extract point name from the filename
                point_name = filename.split('.')[0]
                
                # Load JSON file
                file_path = os.path.join(folder_path, filename)
                with open(file_path) as f:
                    data_json = json.load(f)
                
                # Extract position coordinates
                try:
                    position = data_json["markups"][0]["controlPoints"][0]["position"]
                    row_data[point_name] = position
                except (KeyError, IndexError) as e:
                    print(f"Error in file {filename}: {e}")

        # Append row data to the list
        data.append(row_data)

    # Convert list of dictionaries to a DataFrame and set 'Patient' as index
    df = pd.DataFrame(data).set_index('Patient')
    
    return df

# =============================================================================

# Define the input and output folders

# input_folder = r'Z:\\TM Internships\Dept of CMF\Bram Roumen\Master Thesis - CMF\Thesis\Inference workflow\Predicted patients'
input_folder = r'C:\Users\pb_va\OneDrive\Documents\Technical Medicine\TM2 - Stage 1 - MKA chirurgie\Bram Roumen\Inference workflow\Predicted patients'
output_folder = 'Output'

# Main function
input_path, output_path = create_paths(input_folder, output_folder)
create_folders(input_path, output_path)


df = create_dataframe(input_path)

# Uiteindelijk de bedoeleing dat je hier een input maakt van de input folder en dat je als output de dataframe hebt om te gebruiken voor een andere .py file
# Code kan dan aangeroepen worden voor ground truth en voor de predicted patients





