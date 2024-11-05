# Importing the necessary libraries
import os
import json
import pandas as pd
import numpy as np

# Function to create the paths
def create_paths(input_folder, output_folder):
    input_path = os.path.abspath(input_folder)
    output_path = os.path.abspath(output_folder)
    return input_path, output_path

# =============================================================================

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

def create_plane_3p(df, patient, landmark1, landmark2, landmark3):
    """
    Create a plane defined by three points in 3D space.

    Parameters:
    - df: DataFrame containing patient landmark coordinates.
    - patient: The identifier for the patient in the DataFrame.
    - landmark1: The first landmark (point) as a string.
    - landmark2: The second landmark (point) as a string.
    - landmark3: The third landmark (point) as a string.

    Returns:
    - A list containing the coefficients of the plane equation in the form 
      [A, B, C, D], representing the equation Ax + By + Cz + D = 0.
    """
    # Retrieve landmark coordinates for the patient
    p1 = np.array(df.loc[patient, landmark1])  
    p2 = np.array(df.loc[patient, landmark2])  
    p3 = np.array(df.loc[patient, landmark3])
    
    # Create vectors from p2 to p1 and p2 to p3
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Compute the cross product to find the normal vector of the plane
    normal = np.cross(v1, v2)
    
    # Compute the coefficients of the plane equation
    a, b, c = normal
    d = -np.dot(normal, p2)
    
    # Return the coefficients as a list
    plane = [a, b, c, d]

    return plane

def create_plane_4p(df, patient, landmark1, landmark2, landmark3, landmark4):
    """
    Create a plane defined by two points and the midpoint of two other points in 3D space.

    Parameters:
    - df: DataFrame containing patient landmark coordinates.
    - patient: The identifier for the patient in the DataFrame.
    - landmark1: The first landmark (point) as a string.
    - landmark2: The second landmark (point) as a string.
    - landmark3: The third landmark (point) as a string.
    - landmark4: The fourth landmark (point) as a string.

    Returns:
    - A list containing the coefficients of the plane equation in the form 
      [A, B, C, D], representing the equation Ax + By + Cz + D = 0.
    """
    # Retrieve landmark coordinates for the patient
    p1 = np.array(df.loc[patient, landmark1])  
    p2 = np.array(df.loc[patient, landmark2])  
    p3 = np.array(df.loc[patient, landmark3])  
    p4 = np.array(df.loc[patient, landmark4])
    
    # Calculate the midpoint between landmark 3 and landmark 4
    midpoint = (p3 + p4) / 2

    # Create vectors from p2 to p1 and p2 to midpoint
    v1 = p1 - p2
    v2 = midpoint - p2
    
    # Compute the cross product to find the normal vector of the plane
    normal = np.cross(v1, v2)
    
    # Compute the coefficients of the plane equation
    a, b, c = normal
    d = -np.dot(normal, p2)
    
    # Return the coefficients as a list
    plane = [a, b, c, d]

    return plane

def create_dataframe(input_path, output_path):
    """
    Create a DataFrame with coordinates from .json files of every patient
    and calculate the mandibular plane coefficients.

    Parameters:
    - input_path: Path to the directory containing patient folders with .json files.
    - output_path: Path where the resulting CSV file will be saved.
    - landmark1: The first landmark (point) as a string.
    - landmark2: The second landmark (point) as a string.
    - landmark3: The third landmark (point) as a string.

    Returns:
    - The resulting DataFrame with a new column 'Mandibular plane' containing the plane coefficients.
    """
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

    # Calculate the mandibular plane coefficients for each patient and add to the DataFrame

    # Initialize the columns for the plane coefficients
    df['Mandibular plane'] = None
    df['Maxillary plane'] = None
    df['Occlusal plane'] = None
    df['Facial midplane'] = None
    df['FHP'] = None 


    for patient in df.index:
        try:
            # Calculate the plane coefficients using the specified landmarks
            df.at[patient, 'Mandibular plane'] = create_plane_3p(df, patient, 'Menton', 'r-Gonion', 'l-Gonion')
#            df.at[patient, 'Maxillary plane'] = create_plane_3p(df, patient, 'Nasion', 'r-Pterygoid', 'l-Pterygoid')
#            df.at[patient, 'Occlusal plane'] = create_plane_3p(df, patient, 'r-Molar', 'l-Molar', 'Incisor')
            df.at[patient, 'Facial midplane'] = create_plane_3p(df, patient, 'Sella', 'Nasion', 'Menton')
            df.at[patient, 'FHP'] = create_plane_4p(df, patient, 'Infraorbitale L', 'Infraorbitale R', 'Porion L', 'Porion R')
        except KeyError as e:
            print(f"Error: Landmark '{e.args[0]}' does not exist for patient '{patient}'.")
        

    # Save the DataFrame as a CSV file
    df.to_csv(os.path.join(output_path, 'patients_coordinates.csv'))
    return df


# =============================================================================

# Temporary function to load the CSV file during testing
def loadcsv(output_path, file_name):
    df = pd.read_csv(os.path.join(output_path, file_name), index_col=0)
    # Convert columns with string representations of lists back to lists
    for col in df.columns:
        df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    
    return df

