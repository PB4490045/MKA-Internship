# Importing the necessary libraries
import os
import json
import pandas as pd
import numpy as np
from scipy import linalg

# Function to create the paths
def create_paths(groundtruth_folder, predicted_folder, output_folder):
    groundtruth_path = os.path.abspath(groundtruth_folder)
    predicted_path = os.path.abspath(predicted_folder)
    output_path = os.path.abspath(output_folder)
    return groundtruth_path, predicted_path, output_path

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

def calculate_midpoint(df, patient, landmark1, landmark2):
    """
    Calculate the midpoint between two landmarks for a given patient.

    Parameters:
    - df: DataFrame containing patient landmark coordinates.
    - patient: The identifier for the patient in the DataFrame.
    - landmark1: The first landmark (point) as a string.
    - landmark2: The second landmark (point) as a string.

    Returns:
    - A numpy array representing the midpoint coordinates.
      Returns np.array([np.nan, np.nan, np.nan]) if any landmark is missing.
    """
    try:
        p1 = np.array(df.loc[patient, landmark1])
    except KeyError:
        print(f"Landmark '{landmark1}' is missing for patient {patient} for calculating midpoint.")
        return np.array([np.nan, np.nan, np.nan])
    
    try:
        p2 = np.array(df.loc[patient, landmark2])
    except KeyError:
        print(f"Landmark '{landmark2}' is missing for patient {patient} for calculating midpoint.")
        return np.array([np.nan, np.nan, np.nan])
    
    # Calculate the midpoint between p1 and p2
    midpoint = (p1 + p2) / 2
    return midpoint

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
      Returns [np.nan, np.nan, np.nan, np.nan] if any of the landmarks are missing or invalid.
    """
    # Check each landmark individually
    try:
        p1 = np.array(df.loc[patient, landmark1])
        if p1.shape != (3,):
            raise ValueError(f"Landmark '{landmark1}' does not have 3 coordinates for patient {patient}.")
    except KeyError:
        print(f"Landmark '{landmark1}' is missing for patient {patient} for 3-point plane calculation.")
        return [np.nan, np.nan, np.nan, np.nan]
    except ValueError as e:
        print(e)
        return [np.nan, np.nan, np.nan, np.nan]

    try:
        p2 = np.array(df.loc[patient, landmark2])
        if p2.shape != (3,):
            raise ValueError(f"Landmark '{landmark2}' does not have 3 coordinates for patient {patient}.")
    except KeyError:
        print(f"Landmark '{landmark2}' is missing for patient {patient} for 3-point plane calculation.")
        return [np.nan, np.nan, np.nan, np.nan]
    except ValueError as e:
        print(e)
        return [np.nan, np.nan, np.nan, np.nan]

    try:
        p3 = np.array(df.loc[patient, landmark3])
        if p3.shape != (3,):
            raise ValueError(f"Landmark '{landmark3}' does not have 3 coordinates for patient {patient}.")
    except KeyError:
        print(f"Landmark '{landmark3}' is missing for patient {patient} for 3-point plane calculation.")
        return [np.nan, np.nan, np.nan, np.nan]
    except ValueError as e:
        print(e)
        return [np.nan, np.nan, np.nan, np.nan]
    
    # Create vectors from p2 to p1 and p2 to p3
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Compute the cross product to find the normal vector of the plane
    normal = np.cross(v1, v2)

    # If the normal vector is all zeros, the points are collinear, and a unique plane cannot be defined
    if np.all(normal == 0):
        print(f"The landmarks for patient {patient} are collinear and cannot define a unique plane.")
        return [np.nan, np.nan, np.nan, np.nan]    
    
    # Compute the coefficients of the plane equation
    a, b, c = normal
    d = -np.dot(normal, p2)
    
    return np.array([a, b, c, d])

def create_plane_4p(df, patient, landmark1, landmark2, landmark3, landmark4):
    """
    Create a plane defined by the midpoint of two landmarks and two other landmarks in 3D space.

    Parameters:
    - df: DataFrame containing patient landmark coordinates.
    - patient: The identifier for the patient in the DataFrame.
    - landmark1: The first landmark (point) as a string used to calculate the midpoint.
    - landmark2: The second landmark (point) as a string used to calculate the midpoint.
    - landmark3: The third landmark (point) as a string.
    - landmark4: The fourth landmark (point) as a string.

    Returns:
    - A list containing the coefficients of the plane equation in the form 
      [A, B, C, D], representing the equation Ax + By + Cz + D = 0.
      Returns [np.nan, np.nan, np.nan, np.nan] if any landmarks are missing, 
      or if the midpoint cannot be calculated.
    """
    # Calculate the midpoint
    midpoint = calculate_midpoint(df, patient, landmark1, landmark2)

    # Check if midpoint contains NaN values
    if midpoint is None or np.isnan(midpoint).any():
        print(f"Midpoint between '{landmark1}' and '{landmark2}' could not be calculated for patient {patient}.")
        return [np.nan, np.nan, np.nan, np.nan]

    # Check each landmark individually
    try:
        p3 = np.array(df.loc[patient, landmark3])
        if p3.shape != (3,):
            raise ValueError(f"Landmark '{landmark3}' does not have 3 coordinates for patient {patient}.")
    except KeyError:
        print(f"Landmark '{landmark3}' is missing for patient {patient} for 4-point plane calculation.")
        return [np.nan, np.nan, np.nan, np.nan]
    except ValueError as e:
        print(e)
        return [np.nan, np.nan, np.nan, np.nan]

    try:
        p4 = np.array(df.loc[patient, landmark4])
        if p4.shape != (3,):
            raise ValueError(f"Landmark '{landmark4}' does not have 3 coordinates for patient {patient}.")
    except KeyError:
        print(f"Landmark '{landmark4}' is missing for patient {patient} for 4-point plane calculation.")
        return [np.nan, np.nan, np.nan, np.nan]
    except ValueError as e:
        print(e)
        return [np.nan, np.nan, np.nan, np.nan]

    # Create vectors from the midpoint to p3 and p4
    v1 = p3 - midpoint
    v2 = p4 - midpoint

    # Compute the cross product to find the normal vector of the plane
    normal = np.cross(v1, v2)

    # Check if the normal vector is all zeros (indicating collinear points)
    if np.all(normal == 0):
        print(f"The points for patient {patient} are collinear and cannot define a unique plane.")
        return [np.nan, np.nan, np.nan, np.nan]

    # Compute the coefficients of the plane equation
    a, b, c = normal
    d = -np.dot(normal, midpoint)

    # Return the coefficients as a list
    return np.array([a, b, c, d])


def occlusal_plane(df, patient, landmark1='IsU1', landmark2='IsL1', landmark3='13', landmark4='43', 
                   landmark5='23', landmark6='33', landmark7='16', landmark8='46', landmark9='26', landmark10='36'):
    """
    Calculate the occlusal plane defined by five pairs of landmarks, each pair averaged into a midpoint.

    Parameters:
    - df: DataFrame containing patient landmark coordinates.
    - patient: The identifier for the patient in the DataFrame.
    - landmark1, landmark2, ..., landmark10: Strings representing landmarks to calculate midpoints.

    Returns:
    - A list containing the coefficients of the plane equation [A, B, C, D].
      Returns [np.nan, np.nan, np.nan, np.nan] if any landmark is missing or if points are collinear.
    """
    # Retrieve the coordinates of the landmarks from the DataFrame
    try:
        points = np.array([
            calculate_midpoint(df, patient, landmark1, landmark2),
            calculate_midpoint(df, patient, landmark3, landmark4),
            calculate_midpoint(df, patient, landmark5, landmark6),
            calculate_midpoint(df, patient, landmark7, landmark8),
            calculate_midpoint(df, patient, landmark9, landmark10)
        ])
    except KeyError as e:
        print(f"Landmark '{e.args[0]}' is missing for patient {patient}.")
        return [np.nan, np.nan, np.nan, np.nan]
    
    # Check if any midpoint contains NaN values
    if np.isnan(points).any():
        print(f"One or more landmarks contain NaN values for patient {patient}.")
        return [np.nan, np.nan, np.nan, np.nan]
    
    # Create the design matrix for the least squares solution
    # The design matrix will be [x, y, z, 1] for each point
    A = np.c_[points[:, 0], points[:, 1], points[:, 2], np.ones(points.shape[0])]
    
    # Check if the points are collinear by evaluating rank
    if np.linalg.matrix_rank(A) < 3:
        print(f"The points for patient {patient} are collinear and cannot define a unique plane.")
        return [np.nan, np.nan, np.nan, np.nan]
    
    # Perform the least squares solution
    _, _, Vt = linalg.svd(A)
    
    # The last row of Vt is the solution for [A, B, C, D]
    plane_coefficients = Vt[-1, :]
    
    # Return the coefficients as [A, B, C, D]
    return np.array(plane_coefficients)

def create_dataframe(input_path, output_path, df_name):
    """
    Create a DataFrame with coordinates from .json files of every patient
    and calculate the mandibular plane coefficients.

    Parameters:
    - input_path: Path to the directory containing patient folders with .json files.
    - output_path: Path where the resulting CSV file will be saved.

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
    df['Occlusal plane'] = None
    df['FHP'] = None
    df['Facial midplane'] = None
    # df['Maxillary plane'] = None 

    for patient in df.index:
        # Calculate the plane coefficients using the specified landmarks
        df.at[patient, 'Mandibular plane'] = create_plane_3p(df, patient, 'Menton', 'r-Gonion', 'l-Gonion')
        df.at[patient, 'Occlusal plane'] = occlusal_plane(df, patient)
        df.at[patient, 'FHP'] = create_plane_4p(df, patient, 'Porion L', 'Porion R', 'Infraorbitale L', 'Infraorbitale R')
        df.at[patient, 'Facial midplane'] = create_plane_3p(df, patient, 'Sella', 'Nasion', 'Menton')
        # df.at[patient, 'Maxillary plane'] = create_plane_3p(df, patient,')

    # Save the DataFrame as a CSV file
    df.to_csv(os.path.join(output_path, f'{df_name}.csv'))
    return df

# =============================================================================

def loadcsv(output_path, file_name):
    df = pd.read_csv(os.path.join(output_path, file_name), index_col=0)
    
    # Convert columns with string representations of lists back to lists, or handle other types
    for col in df.columns:
        def safe_load(x):
            if isinstance(x, str):
                # Attempt to decode JSON if it looks like JSON
                try:
                    # Handle the case where the string is valid JSON (list, dict, etc.)
                    return json.loads(x)
                except json.JSONDecodeError:
                    # If it fails, return the string as is
                    return x
            elif isinstance(x, (int, float, bool)):  # Return numbers or booleans as is
                return x
            else:
                # For other types, simply return the value
                return x
        
        df[col] = df[col].apply(safe_load)
    
    return df
