"""
ceph_analysis.py

This module provides functions for cephalometric analysis based on anatomical landmarks. It includes
functions to calculate distances and angles between landmarks, project points onto planes, and
compute distances from points to planes. Additionally, the module performs cephalometric analysis
for each patient and stores results in a new DataFrame.

Functions:
- calculate_distance: Calculate the Euclidean distance between two landmarks for a patient.
- calculate_angle_3p: Calculate the angle formed by three landmarks in 3D space.
- calculate_angle_4p: Calculate the angle between two lines formed by four landmarks in 3D space.
- angle_between_planes: Calculate the angle between two planes given their coefficients.
- project_point_to_plane: Project a 3D point onto a specified plane.
- point_to_plane_distance: Calculate the perpendicular distance from a point to a plane.
- cephalometric_analysis: Perform a cephalometric analysis for each patient and store results.
"""

import numpy as np
import pandas as pd


def calculate_distance(df, patient, landmark1, landmark2):
    """
    Calculate the distance between two landmarks for a specific patient.

    Parameters:
    - df (pd.DataFrame): DataFrame containing patient landmark coordinates.
    - patient (str or int): The identifier for the patient in the DataFrame.
    - landmark1 (str): The first landmark identifier.
    - landmark2 (str): The second landmark identifier.

    Returns:
    - float: The Euclidean distance between the two landmarks.
      If a landmark is missing, returns NaN.
    """
    try:
        point1 = np.array(df.loc[patient, landmark1])
        point2 = np.array(df.loc[patient, landmark2])

        if pd.isna(point1).any() or pd.isna(point2).any():
            print(f"Error: One or both landmarks '{landmark1}', '{landmark2}' for patient "
                  f"'{patient}' have missing data.")
            return np.nan

        if point1.shape[0] != 3 or point2.shape[0] != 3:
            print(f"Error: Landmark coordinates for '{landmark1}' or '{landmark2}' are invalid "
                  f"for patient '{patient}'.")
            return np.nan

    except (KeyError, ValueError) as e:
        print(f"Error: {str(e)} for patient '{patient}'. Returning NaN.")
        return np.nan

    distance = np.sqrt((point1[0] - point2[0])**2 +
                       (point1[1] - point2[1])**2 +
                       (point1[2] - point2[2])**2)

    return distance


def calculate_angle_3p(df, patient, landmark1, landmark2, landmark3):
    """
    Calculate the angle formed by three landmarks in 3D space.

    Parameters:
    - df (pd.DataFrame): DataFrame containing patient landmark coordinates.
    - patient (str or int): The identifier for the patient in the DataFrame.
    - landmark1: The first landmark identifier.
    - landmark2: The second landmark identifier.
    - landmark3: The third landmark identifier.

    Returns:
    - float: The angle in degrees between vectors formed by the landmarks.
      Returns NaN if any landmark is missing.
    """
    try:
        p1 = np.array(df.loc[patient, landmark1])
        p2 = np.array(df.loc[patient, landmark2])
        p3 = np.array(df.loc[patient, landmark3])
    except KeyError as e:
        print(f"Error: Landmark '{e.args[0]}' missing for patient '{patient}'. Returning NaN.")
        return np.nan

    v1 = p1 - p2
    v2 = p3 - p2

    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    if magnitude_v1 == 0 or magnitude_v2 == 0:
        raise ValueError("One or more vectors have zero length; cannot compute angle.")

    angle_radians = np.arccos(np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0))
    return np.degrees(angle_radians)


def calculate_angle_4p(df, patient, landmark1, landmark2, landmark3, landmark4):
    """
    Calculate the angle formed by the intersection of two lines, each defined by two landmarks.

    Parameters:
    - df (pd.DataFrame): DataFrame containing patient landmark coordinates.
    - patient (str or int): The identifier for the patient in the DataFrame.
    - landmark1: The first landmark for the first line.
    - landmark2: The second landmark for the first line.
    - landmark3: The first landmark for the second line.
    - landmark4: The second landmark for the second line.

    Returns:
    - float: The angle in degrees between the two lines.
      Returns NaN if any landmark is missing.
    """
    try:
        p1 = np.array(df.loc[patient, landmark1])
        p2 = np.array(df.loc[patient, landmark2])
        p3 = np.array(df.loc[patient, landmark3])
        p4 = np.array(df.loc[patient, landmark4])
    except KeyError as e:
        print(f"Error: Landmark '{e.args[0]}' missing for patient '{patient}'. Returning NaN.")
        return np.nan

    v1 = p2 - p1
    v2 = p4 - p3

    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    if magnitude_v1 == 0 or magnitude_v2 == 0:
        raise ValueError("One or more vectors have zero length; cannot compute angle.")

    angle_radians = np.arccos(np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0))
    return np.degrees(angle_radians)


def angle_between_planes(df, patient, plane1_coefficients, plane2_coefficients):
    """
    Calculate the angle between two planes given their coefficients.

    Parameters:
    - df (pd.DataFrame): DataFrame containing patient data.
    - patient (str or int): Identifier for the patient in the DataFrame.
    - plane1_coefficients: First plane's coefficients column name.
    - plane2_coefficients: Second plane's coefficients column name.

    Returns:
    - float: The angle between the two planes in degrees.
    """
    a, b, c, _ = np.array(df.loc[patient, plane1_coefficients])
    e, f, g, _ = np.array(df.loc[patient, plane2_coefficients])
    normal1 = [a, b, c]
    normal2 = [e, f, g]

    cos_angle = np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2))
    angle_radians = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle_radians)


def project_point_to_plane(df, patient, landmark, plane):
    """
    Project a 3D point onto a plane.

    Parameters:
    - df (pd.DataFrame): DataFrame containing patient data.
    - patient (str or int): Identifier for the patient in the DataFrame.
    - landmark: Landmark identifier for the point.
    - plane: Plane coefficients column name.

    Returns:
    - np.array: Coordinates of the projected point on the plane.
    """
    a, b, c, d = np.array(df.loc[patient, plane])
    landmark = np.array(df.loc[patient, landmark])

    normal_vector = [a, b, c]

    numerator = a * landmark[0] + b * landmark[1] + c * landmark[2] + d
    denominator = np.linalg.norm(normal_vector)
    distance = numerator / denominator

    return np.array(landmark) - distance * (normal_vector / denominator)


def point_to_plane_distance(df, patient, landmark, plane):
    """
    Calculate the perpendicular distance from a point to a plane in 3D space.

    Parameters:
    - df (pd.DataFrame): DataFrame containing patient data.
    - patient (str or int): Identifier for the patient in the DataFrame.
    - landmark: Landmark identifier for the point.
    - plane: Plane coefficients column name.

    Returns:
    - np.array: Perpendicular distance from the point to the plane.
    """
    point = np.array(df.loc[patient, landmark])
    a, b, c, d = np.array(df.loc[patient, plane])
    normal_vector = [a, b, c]

    numerator = np.abs(a * point[0] + b * point[1] + c * point[2] + d)
    denominator = np.linalg.norm(normal_vector)
    distance = numerator / denominator

    return np.array([distance])


def cephalometric_analysis(df):
    """
    Perform cephalometric analysis for each patient and store results in a new DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing patient landmark coordinates.

    Returns:
    - pd.DataFrame: DataFrame containing cephalometric measurements for each patient.
    """
    df_ceph = pd.DataFrame(index=df.index.unique(), columns=[
        'Condylar width', 'Coronoidal width', 'Zygomatic width', 'Anterior facial height', 
        'Anterior upper facial height', 'Anterior midfacial height', 'Anterior lower facial height', 
        'Posterior facial height', 'SNA', 'SNB', 'ANB'
    ])

    for patient in df.index.unique():
        df_ceph.loc[patient, 'Condylar width'] = calculate_distance(df, patient, 'r-Condyle', 'l-Condyle')
        df_ceph.loc[patient, 'Coronoidal width'] = calculate_distance(df, patient, 'r-Coronoid', 'l-Coronoid')
        df_ceph.loc[patient, 'Zygomatic width'] = calculate_distance(df, patient, 'Zygomatic Process R',
                                                                     'Zygomatic Process L')
        df_ceph.loc[patient, 'Anterior facial height'] = calculate_distance(df, patient, 'Nasion', 'Menton')
        df_ceph.loc[patient, 'Anterior upper facial height'] = calculate_distance(df, patient, 'Nasion', 'Sella')

    return df_ceph
