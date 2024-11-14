"""
preprocessing.py

This module provides functions for preparing, processing, and analyzing cephalometric
data from JSON files. It includes utilities for directory management, DataFrame
creation, plane calculations, and file I/O operations.

Functions:
- create_paths(groundtruth_folder, predicted_folder, output_folder):
    Generates absolute paths for folders used in the analysis.
- create_folders(input_path, output_path):
    Creates output folders for each patient based on the input path structure.
- calculate_midpoint(df, patient, landmark1, landmark2):
    Calculates the midpoint between two landmarks for a specified patient.
- create_plane_3p(df, patient, landmark1, landmark2, landmark3):
    Defines a 3-point plane based on three landmarks.
- create_plane_4p(df, patient, landmark1, landmark2, landmark3, landmark4):
    Defines a plane using a midpoint and two additional landmarks.
- occlusal_plane(df, patient, landmark1, ..., landmark10):
    Calculates the occlusal plane using up to five landmark pairs.
- perpendicular_plane(df, patient, plane, midpoint, landmark1):
    Computes a plane perpendicular to a specified plane based on a direction vector.
- create_dataframe(input_path):
    Creates a DataFrame by extracting landmark coordinates from JSON files within
    patient folders.
- create_dataframe_from_folders(input_path, folder_names):
    Creates a DataFrame from specified patient folders, excluding duplicate landmarks.
- create_planes(df):
    Calculates and adds specific planes (e.g., Mandibular, Occlusal, FHP, MSP)
    as new columns to the DataFrame.
- save_df(df, output_path, df_name):
    Saves the DataFrame to a CSV file.
- loadcsv(output_path, file_name):
    Loads a DataFrame from a CSV file, processing JSON-encoded columns back to
    their original data types.

This module is used for pre-processing anatomical landmarks for cephalometric analysis,
preparing data for further analysis and model evaluation.
"""

import os
import json

import pandas as pd
import numpy as np
from scipy import linalg

def create_paths(groundtruth_folder, predicted_folder, output_folder):
    """
    Generate absolute paths for the ground truth, predicted, and output folders.

    Parameters:
    - groundtruth_folder: Folder path for ground truth data.
    - predicted_folder: Folder path for predicted data.
    - output_folder: Folder path for saving outputs.

    Returns:
    - Tuple of absolute paths: (groundtruth_path, predicted_path, output_path).
    """
    groundtruth_path = os.path.abspath(groundtruth_folder)
    predicted_path = os.path.abspath(predicted_folder)
    output_path = os.path.abspath(output_folder)

    return groundtruth_path, predicted_path, output_path

def create_folders(input_path, output_path):
    """
    Create output folders for each subfolder found in the input path.

    Parameters:
    - input_path: Path containing the input folders.
    - output_path: Path where the corresponding output folders should be created.

    Returns:
    - None. Prints a message for each folder created or already existing.
    """
    folders = [
        f for f in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, f))
    ]

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
    - landmark1: The first landmark (point) for calculating the midpoint.
    - landmark2: The second landmark (point) for calculating the midpoint.

    Returns:
    - A numpy array representing the midpoint coordinates.
      Returns np.array([np.nan, np.nan, np.nan]) if any landmark is missing.
    """
    try:
        p1 = np.array(df.loc[patient, landmark1])
    except KeyError:
        print(
            f"Landmark '{landmark1}' is missing for patient {patient} "
            "for calculating midpoint."
        )
        return np.array([np.nan, np.nan, np.nan])

    try:
        p2 = np.array(df.loc[patient, landmark2])
    except KeyError:
        print(
            f"Landmark '{landmark2}' is missing for patient {patient} "
            "for calculating midpoint."
        )
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
    - landmark1: The first landmark (point) to define the plane.
    - landmark2: The second landmark (point) to define the plane.
    - landmark3: The third landmark (point) to define the plane.

    Returns:
    - A numpy array containing the normalized coefficients of the plane equation
      in the form [A, B, C, D], representing the equation Ax + By + Cz + D = 0.
      Returns np.array([np.nan, np.nan, np.nan, np.nan]) if any of the landmarks
      are missing or invalid.
    """
    # Check each landmark individually
    try:
        p1 = np.array(df.loc[patient, landmark1])
        if p1.shape != (3,):
            raise ValueError(
                f"Landmark '{landmark1}' does not have 3 coordinates for patient {patient}."
            )
    except KeyError:
        print(f"Landmark '{landmark1}' is missing for patient {patient} for 3-point plane calculation.")
        return np.array([np.nan, np.nan, np.nan, np.nan])
    except ValueError as e:
        print(e)
        return np.array([np.nan, np.nan, np.nan, np.nan])

    try:
        p2 = np.array(df.loc[patient, landmark2])
        if p2.shape != (3,):
            raise ValueError(
                f"Landmark '{landmark2}' does not have 3 coordinates for patient {patient}."
            )
    except KeyError:
        print(f"Landmark '{landmark2}' is missing for patient {patient} for 3-point plane calculation.")
        return np.array([np.nan, np.nan, np.nan, np.nan])
    except ValueError as e:
        print(e)
        return np.array([np.nan, np.nan, np.nan, np.nan])

    try:
        p3 = np.array(df.loc[patient, landmark3])
        if p3.shape != (3,):
            raise ValueError(
                f"Landmark '{landmark3}' does not have 3 coordinates for patient {patient}."
            )
    except KeyError:
        print(f"Landmark '{landmark3}' is missing for patient {patient} for 3-point plane calculation.")
        return np.array([np.nan, np.nan, np.nan, np.nan])
    except ValueError as e:
        print(e)
        return np.array([np.nan, np.nan, np.nan, np.nan])

    # Create vectors from p2 to p1 and p2 to p3
    v1 = p1 - p2
    v2 = p3 - p2

    # Compute the cross product to find the normal vector of the plane
    normal = np.cross(v1, v2)

    # If the normal vector is all zeros, the points are collinear, and a unique plane cannot be defined
    if np.all(normal == 0):
        print(f"The landmarks for patient {patient} are collinear and cannot define a unique plane.")
        return np.array([np.nan, np.nan, np.nan, np.nan])

    # Compute the coefficients of the plane equation
    a, b, c = normal
    d = -np.dot(normal, p2)

    # Normalize the coefficients if the normal vector norm is not zero
    norm = np.linalg.norm([a, b, c])
    if norm != 0:
        a, b, c, d = a / norm, b / norm, c / norm, d / norm

    return np.array([a, b, c, d])

def create_plane_4p(df, patient, landmark1, landmark2, landmark3, landmark4):
    """
    Create a plane defined by the midpoint of two landmarks and two additional landmarks in 3D space.

    Parameters:
    - df: DataFrame containing patient landmark coordinates.
    - patient: The identifier for the patient in the DataFrame.
    - landmark1: The first landmark for calculating the midpoint.
    - landmark2: The second landmark for calculating the midpoint.
    - landmark3: The third landmark for defining the plane.
    - landmark4: The fourth landmark for defining the plane.

    Returns:
    - A numpy array containing the normalized coefficients of the plane equation
      in the form [A, B, C, D], representing the equation Ax + By + Cz + D = 0.
      Returns np.array([np.nan, np.nan, np.nan, np.nan]) if any landmarks are missing 
      or if the midpoint cannot be calculated.
    """
    # Calculate the midpoint
    midpoint = calculate_midpoint(df, patient, landmark1, landmark2)

    # Check if midpoint contains NaN values
    if midpoint is None or np.isnan(midpoint).any():
        print(
            f"Midpoint between '{landmark1}' and '{landmark2}' could not be calculated "
            f"for patient {patient}."
        )
        return np.array([np.nan, np.nan, np.nan, np.nan])

    # Check each landmark individually
    try:
        p3 = np.array(df.loc[patient, landmark3])
        if p3.shape != (3,):
            raise ValueError(
                f"Landmark '{landmark3}' does not have 3 coordinates for patient {patient}."
            )
    except KeyError:
        print(
            f"Landmark '{landmark3}' is missing for patient {patient} for 4-point plane calculation."
        )
        return np.array([np.nan, np.nan, np.nan, np.nan])
    except ValueError as e:
        print(e)
        return np.array([np.nan, np.nan, np.nan, np.nan])

    try:
        p4 = np.array(df.loc[patient, landmark4])
        if p4.shape != (3,):
            raise ValueError(
                f"Landmark '{landmark4}' does not have 3 coordinates for patient {patient}."
            )
    except KeyError:
        print(
            f"Landmark '{landmark4}' is missing for patient {patient} for 4-point plane calculation."
        )
        return np.array([np.nan, np.nan, np.nan, np.nan])
    except ValueError as e:
        print(e)
        return np.array([np.nan, np.nan, np.nan, np.nan])

    # Create vectors from the midpoint to p3 and p4
    v1 = p3 - midpoint
    v2 = p4 - midpoint

    # Compute the cross product to find the normal vector of the plane
    normal = np.cross(v1, v2)

    # Check if the normal vector is all zeros (indicating collinear points)
    if np.all(normal == 0):
        print(
            f"The points for patient {patient} are collinear and cannot define a unique plane."
        )
        return np.array([np.nan, np.nan, np.nan, np.nan])

    # Compute the coefficients of the plane equation
    a, b, c = normal
    d = -np.dot(normal, midpoint)

    # Normalize the coefficients if the normal vector norm is not zero
    norm = np.linalg.norm([a, b, c])
    if norm != 0:
        a, b, c, d = a / norm, b / norm, c / norm, d / norm

    return np.array([a, b, c, d])


def occlusal_plane(
    df, patient, landmark1='IsU1', landmark2='IsL1', landmark3='13', landmark4='43',
    landmark5='23', landmark6='33', landmark7='16', landmark8='46', landmark9='26', landmark10='36'
):
    """
    Calculate the occlusal plane defined by up to five pairs of landmarks, each pair averaged into a midpoint.
    If fewer than five pairs are available, at least three pairs are required to define the plane.

    Parameters:
    - df: DataFrame containing patient landmark coordinates.
    - patient: The identifier for the patient in the DataFrame.
    - landmark1, landmark2, ..., landmark10: Landmarks to calculate midpoints.

    Returns:
    - A list containing the coefficients of the plane equation [A, B, C, D].
      Returns [np.nan, np.nan, np.nan, np.nan] if fewer than three valid midpoints are available.
    """
    # Retrieve the coordinates of the landmarks from the DataFrame
    points = []
    for lm1, lm2 in [
        (landmark1, landmark2), (landmark3, landmark4),
        (landmark5, landmark6), (landmark7, landmark8), (landmark9, landmark10)
    ]:
        midpoint = calculate_midpoint(df, patient, lm1, lm2)
        # Only add valid midpoints (no NaN values) to the points list
        if not np.isnan(midpoint).any():
            points.append(midpoint)

    # Check if we have enough points to define a plane (at least 3)
    if len(points) < 3:
        print(
            f"Not enough valid midpoints to calculate the occlusal plane for patient {patient}."
        )
        return [np.nan, np.nan, np.nan, np.nan]

    points = np.array(points)  # Convert list of valid points to a NumPy array

    # Create the design matrix for the least squares solution
    a = np.c_[points[:, 0], points[:, 1], points[:, 2], np.ones(points.shape[0])]

    # Check if the points are collinear by evaluating rank
    if np.linalg.matrix_rank(a) < 3:
        print(
            f"The points for patient {patient} are collinear and cannot define a unique plane."
        )
        return [np.nan, np.nan, np.nan, np.nan]

    # Perform the least squares solution
    _, _, vt = linalg.svd(a)

    # The last row of Vt is the solution for [a, b, c, d]
    plane_coefficients = vt[-1, :]

    # Return the coefficients as [a, b, c, d]
    return plane_coefficients

def perpendicular_plane(df, patient, plane, midpoint, landmark1):
    """
    Calculate a plane perpendicular to a given plane, defined by a direction vector
    from a midpoint to another landmark.

    Parameters:
    - df: DataFrame containing patient plane and landmark coordinates.
    - patient: The identifier for the patient in the DataFrame.
    - plane: The plane from which to derive the perpendicular plane's normal vector.
    - midpoint: The midpoint to define the direction vector for the perpendicular plane.
    - landmark1: A point used to calculate the direction vector from the midpoint.

    Returns:
    - A numpy array containing the coefficients of the perpendicular plane equation [A, B, C, D].
      Returns [np.nan, np.nan, np.nan, np.nan] if the direction vector or normal vector
      contains NaN values or normalization fails.
    """
    a, b, c, _ = np.array(df.loc[patient, plane])
    landmark1 = np.array(df.loc[patient, landmark1])

    # Define the normal vector of the plane
    normal_vector1 = [a, b, c]

    # Calculate the direction vector from the midpoint to landmark1
    direction_vector = midpoint - landmark1

    # Check if any part of the direction vector or normal vector is NaN
    if np.isnan(direction_vector).any() or np.isnan(normal_vector1).any():
        return [np.nan, np.nan, np.nan, np.nan]

    # Calculate the normal vector of the new plane (cross product)
    normal_vector2 = np.cross(normal_vector1, direction_vector)

    # Normalize the normal vector of the new plane
    if np.linalg.norm(normal_vector2) != 0:
        normal_vector2 = normal_vector2 / np.linalg.norm(normal_vector2)
    else:
        return [np.nan, np.nan, np.nan, np.nan]

    # Calculate the d coefficient of the new plane (Ax + By + Cz + D = 0)
    d = -np.dot(normal_vector2, landmark1)

    # Return the coefficients of the new plane
    return np.array([normal_vector2[0], normal_vector2[1], normal_vector2[2], d])


def create_dataframe(input_path):
    """
    Create a DataFrame from JSON files in patient folders, extracting landmark coordinates.

    Parameters:
    - input_path: Path to the directory containing patient folders.

    Returns:
    - A tuple containing:
      - DataFrame with patient identifiers as the index and landmarks as columns.
      - List of folder names (patient identifiers).
    """
    data = []  # Initialize a list to collect each patient's data

    # Get folder names (e.g., patient folders)
    folder_names = [
        f for f in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, f))
    ]

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
                with open(file_path, encoding='utf-8') as f:
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

    return df, folder_names


def create_dataframe_from_folders(input_path, folder_names):
    """
    Create a DataFrame from JSON files in specific patient folders, excluding duplicate landmarks.

    Parameters:
    - input_path: Path to the directory containing patient folders.
    - folder_names: List of folder names (patient identifiers) to process.

    Returns:
    - DataFrame with patient identifiers as the index and landmarks as columns.
    """
    data = []  # Initialize a list to collect each patient's data

    for folder in folder_names:
        folder_path = os.path.join(input_path, folder)
        row_data = {'Patient': folder}  # Dictionary to store data for each patient

        # Check if the folder actually exists to avoid errors
        if not os.path.isdir(folder_path):
            print(f"Folder {folder} does not exist in the specified path.")
            continue

        # Loop through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                # Extract point name from the filename
                point_name = filename.split('.')[0]

                # Exclude landmarks that end with '_1' or '_2'
                if point_name.endswith('_1') or point_name.endswith('_2'):
                    continue  # Skip this landmark if it's a duplicate

                # Load JSON file
                file_path = os.path.join(folder_path, filename)
                with open(file_path, encoding='utf-8') as f:
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

def create_planes(df):
    """
    Calculate specific planes for each patient in the DataFrame and add the coefficients 
    as new columns for each plane.

    Parameters:
    - df: DataFrame containing patient data and landmarks.

    Returns:
    - DataFrame with additional columns for each plane's coefficients: 
      'Mandibular plane', 'Occlusal plane', 'FHP', and 'MSP'.
    """
    # Initialize the columns for the plane coefficients
    df['Mandibular plane'] = None
    df['Occlusal plane'] = None
    df['FHP'] = None
    df['MSP'] = None

    for patient in df.index:
        # Calculate the plane coefficients using the specified landmarks
        df.at[patient, 'Mandibular plane'] = create_plane_3p(
            df, patient, 'Menton', 'r-Gonion', 'l-Gonion'
        )
        df.at[patient, 'Occlusal plane'] = occlusal_plane(df, patient)
        df.at[patient, 'FHP'] = create_plane_4p(
            df, patient, 'Porion L', 'Porion R', 'Infraorbitale L', 'Infraorbitale R'
        )
        df.at[patient, 'MSP'] = perpendicular_plane(
            df, patient, 'FHP', calculate_midpoint(df, patient, 'Infraorbitale L',
                                                   'Infraorbitale R'), 'Sella'
        )  # Nasion missing, this was improvised adaptation

    return df

# =============================================================================

def save_df(df, output_path, df_name):
    """
    Save the DataFrame to a CSV file.

    Parameters:
    - df: DataFrame to be saved.
    - output_path: Path to the directory where the file will be saved.
    - df_name: Name of the output CSV file (without '.csv' extension).

    Returns:
    - None.
    """
    df.to_csv(os.path.join(output_path, f'{df_name}.csv'))


def loadcsv(output_path, file_name):
    """
    Load a DataFrame from a CSV file and process columns containing lists or other 
    non-standard data types represented as strings.

    Parameters:
    - output_path: Path to the directory containing the CSV file.
    - file_name: Name of the CSV file to load.

    Returns:
    - DataFrame with columns processed to handle JSON-encoded strings.
    """
    df = pd.read_csv(os.path.join(output_path, file_name), index_col=0)

    # Convert columns with string representations of lists back to lists, or handle other types
    for col in df.columns:
        def safe_load(x):
            if isinstance(x, str):
                try:
                    # Handle the case where the string is valid JSON (list, dict, etc.)
                    return json.loads(x)
                except json.JSONDecodeError:
                    # If JSON decoding fails, return the string as is
                    return x
            elif isinstance(x, (int, float, bool)):  # Return numbers or booleans as is
                return x
            else:
                # For other types, simply return the value
                return x

        df[col] = df[col].apply(safe_load)

    return df
