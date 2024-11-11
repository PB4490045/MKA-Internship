# Importing the necessary libraries
import numpy as np
import pandas as pd
import preprocessing as pp
import scipy

def calculate_distance(df, patient, landmark1, landmark2): 
    """
    Calculate the distance between two landmarks for a specific patient.

    Parameters:
    - df: DataFrame containing patient landmark coordinates.
    - patient: The identifier for the patient in the DataFrame.
    - landmark1: The first landmark (point) as a string.
    - landmark2: The second landmark (point) as a string.

    Returns:
    - The Euclidean distance between the two landmarks as a float.
      If a landmark does not exist or has missing data, returns NaN.
    """
    try:
        # Retrieve the coordinates for both landmarks
        point1 = np.array(df.loc[patient, landmark1])
        point2 = np.array(df.loc[patient, landmark2])

        # Check if the landmarks exist (i.e., if they are not NaN)
        if pd.isna(point1).any() or pd.isna(point2).any():
            print(f"Error: One or both landmarks ('{landmark1}', '{landmark2}') for patient '{patient}' are missing data. Returning NaN.")
            return np.nan

        # Check if the landmarks are valid 3D coordinates
        if point1.ndim != 1 or point2.ndim != 1 or point1.shape[0] != 3 or point2.shape[0] != 3:
            print(f"Error: Landmark coordinates for '{landmark1}' or '{landmark2}' are not in the correct 3D format for patient '{patient}'. Returning NaN.")
            return np.nan

    except (KeyError, ValueError) as e:
        print(f"Error: {str(e)} for patient '{patient}'. Returning NaN.")
        return np.nan  # Return NaN if a landmark doesn't exist or coordinates are invalid

    # Calculate the Euclidean distance
    distance = np.sqrt((point1[0] - point2[0])**2 + 
                       (point1[1] - point2[1])**2 + 
                       (point1[2] - point2[2])**2)
    
    return distance

def calculate_angle_3p(df, patient, landmark1, landmark2, landmark3):
    """
    Calculate the angle formed by three landmarks in 3D space.

    Parameters:
    - df: DataFrame containing patient landmark coordinates.
    - patient: The identifier for the patient in the DataFrame.
    - landmark1: The first landmark (point) as a string.
    - landmark2: The second landmark (point) as a string.
    - landmark3: The third landmark (point) as a string.

    Returns:
    - The angle in degrees between the vectors formed by the landmarks.
      If any landmark does not exist, returns NaN.
    """
    try:
        # Retrieve coordinates for the landmarks
        p1 = np.array(df.loc[patient, landmark1])  
        p2 = np.array(df.loc[patient, landmark2])  
        p3 = np.array(df.loc[patient, landmark3])
    except KeyError as e:
        print(f"Error: Landmark '{e.args[0]}' does not exist for patient '{patient}'. Returning NaN.")
        return np.nan  # Return NaN if a landmark doesn't exist

    # Create vectors from p2 to p1 and p2 to p3
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Compute the dot product and magnitudes of the vectors
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    # Check for zero-length vectors to avoid division by zero
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        raise ValueError("One or more vectors have zero length; cannot compute angle.")
    
    # Compute the angle in radians and convert to degrees
    angle_radians = np.arccos(np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

def calculate_angle_4p(df, patient, landmark1, landmark2, landmark3, landmark4):
    """
    Calculate the angle formed by the intersection of two lines, each defined by two landmarks, in 3D space.
    
    Parameters:
    - df: DataFrame containing patient landmark coordinates.
    - patient: The identifier for the patient in the DataFrame.
    - landmark1: The first landmark (point) for the first line as a string.
    - landmark2: The second landmark (point) for the first line as a string.
    - landmark3: The first landmark (point) for the second line as a string.
    - landmark4: The second landmark (point) for the second line as a string.
    
    Returns:
    - The angle in degrees between the two lines formed by the landmarks.
      If any landmark does not exist, returns NaN.
    """
    try:
        # Retrieve coordinates for the landmarks
        p1 = np.array(df.loc[patient, landmark1])  
        p2 = np.array(df.loc[patient, landmark2])  
        p3 = np.array(df.loc[patient, landmark3])  
        p4 = np.array(df.loc[patient, landmark4])  
    except KeyError as e:
        print(f"Error: Landmark '{e.args[0]}' does not exist for patient '{patient}'. Returning NaN.")
        return np.nan  # Return NaN if a landmark doesn't exist

    # Create vectors for the two lines
    v1 = p2 - p1  # Vector from landmark1 to landmark2
    v2 = p4 - p3  # Vector from landmark3 to landmark4
    
    # Compute the dot product and magnitudes of the vectors
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    # Check for zero-length vectors to avoid division by zero
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        raise ValueError("One or more vectors have zero length; cannot compute angle.")
    
    # Compute the angle in radians and convert to degrees
    angle_radians = np.arccos(np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

def angle_between_planes(df, patient, plane1_coefficients, plane2_coefficients):
    """
    Calculate the angle between two planes given their coefficients.

    Parameters:
    - plane1_coefficients: List or array of the plane coefficients [A, B, C, D].
    - plane2_coefficients: List or array of the plane coefficients [A, B, C, D].

    Returns:
    - The angle between the two planes in degrees.
    """
    # Extract the normal vectors from the plane coefficients
    a, b, c, d = np.array(df.loc[patient, plane1_coefficients])  # [A, B, C] from the first plane
    e, f, g, h = np.array(df.loc[patient, plane2_coefficients])  # [A, B, C] from the second plane
    normal1 = [a, b, c]
    normal2 = [e, f, g]
    
    print(normal1)
    print(normal2)

    # Calculate the cosine of the angle between the normal vectors
    cos_angle = np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2))
    
    # Use arccos to find the angle in radians, then convert to degrees
    angle_radians = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    return np.degrees(angle_radians)

def project_point_to_plane(df, patient, landmark, plane):
    """
    Projects a point onto a plane in 3D space.

    Parameters:
    - landmark: A list or array representing the 3D coordinates of the point (x, y, z).
    - plane: A list or array containing the coefficients of the plane equation 
      in the form [A, B, C, D], where the plane equation is Ax + By + Cz + D = 0.

    Returns:
    - A numpy array representing the coordinates of the projected point on the plane.

    The function computes the perpendicular distance from the point to the plane 
    and moves the point by this distance along the normal vector of the plane.
    """
    # Extract plane coefficients
    A, B, C, D = np.array(df.loc([patient, plane]))
    landmark = np.array(df.loc([patient, landmark]))
    
    # Define the normal vector of the plane
    normal_vector = [A, B, C]
    
    # Compute the signed distance from the point to the plane
    numerator = A * landmark[0] + B * landmark[1] + C * landmark[2] + D
    denominator = np.linalg.norm(normal_vector)
    distance = numerator / denominator
    
    # Project the point onto the plane by moving it by the distance along the normal vector
    projected_point = np.array(landmark) - distance * (normal_vector / denominator)
    
    return projected_point

def point_to_plane_distance(df, patient, landmark, plane):
    """
    Calculates the perpendicular distance from a point to a plane in 3D space and returns it as a numpy array.

    Parameters:
    - point: A list or array representing the 3D coordinates of the point (x, y, z).
    - plane: A list or array containing the coefficients of the plane equation 
      in the form [A, B, C, D], where the plane equation is Ax + By + Cz + D = 0.

    Returns:
    - A numpy array representing the perpendicular distance from the point to the plane.

    The function computes the signed distance from the point to the plane using 
    the plane equation and the normal vector of the plane.
    """
    # Ensure the point is a numpy array
    point = np.array(df.loc([patient, landmark]))
    
    # Extract plane coefficients
    A, B, C, D = np.array(df.loc([patient, plane]))
    
    # Define the normal vector of the plane
    normal_vector = [A, B, C]
    
    # Compute the numerator of the distance formula: Ax + By + Cz + D
    numerator = np.abs(A * point[0] + B * point[1] + C * point[2] + D)
    
    # Compute the denominator of the distance formula, which is the magnitude of the normal vector
    denominator = np.linalg.norm(normal_vector)
    
    # Calculate the distance using the formula
    distance = numerator / denominator
    
    # Return the distance as a numpy array
    return np.array([distance])

# =============================================================================

def cephalometric_analysis(df):

    # Pre-allocate the DataFrame with NaN values    
    df_ceph = pd.DataFrame(index=df.index.unique(), columns=[
    'Condylar width', 'Coronoidal width', 'Zygomatic width', 
    'Anterior facial height', 'Anterior upper facial height', 
    'Anterior midfacial height', 'Anterior lower facial height', 
    'Posterior facial height', 'SNA', 'SNB', 'ANB'
    ])

    # Loop through each patient based on the DataFrame index
    for patient in df.index.unique():    

        # Cephalometric distances
        df_ceph.loc[patient, 'Condylar width'] = calculate_distance(df, patient, 'r-Condyle', 'l-Condyl')
        df_ceph.loc[patient, 'Coronoidal width'] = calculate_distance(df, patient, 'r-Coronoid', 'l-Coronoid')
        df_ceph.loc[patient, 'Zygomatic width'] = calculate_distance(df, patient, 'Zygomatic Process R', 'Zygomatic Process L')
        df_ceph.loc[patient, 'Left Ramus height'] = calculate_distance(df, patient, 'Porion L', 'l-Gonion')
        df_ceph.loc[patient, 'Right Ramus height'] = calculate_distance(df, patient, 'Porion R', 'r-Gonion')
        df_ceph.loc[patient, 'Ramus length diff'] = (df_ceph.loc[patient, 'Right Ramus height'] - df_ceph.loc[patient, 'Left Ramus height'])
        df_ceph.loc[patient, 'Left Mandibular length'] = calculate_distance(df, patient, 'l-Gonion', 'B-point')
        df_ceph.loc[patient, 'Right Mandibular length'] = calculate_distance(df, patient, 'r-Gonion', 'B-point')
        df_ceph.loc[patient, 'Mandibular length diff'] = (df_ceph.loc[patient, 'Right Mandibular length'] - df_ceph.loc[patient, 'Left Mandibular length'])

        df_ceph.loc[patient, 'Anterior facial height'] = calculate_distance(df, patient, 'Nasion', 'Menton')                     # Maybe with projection
        df_ceph.loc[patient, 'Anterior upper facial height'] = calculate_distance(df, patient, 'Nasion', 'Sella')                # Maybe with projection
        df_ceph.loc[patient, 'Anterior midfacial height'] = calculate_distance(df, patient, 'Nasion', 'Anterior Nasal Spine')    # Maybe with projection
        df_ceph.loc[patient, 'Anterior lower facial height'] = calculate_distance(df, patient, 'Menton', 'Posterior Nasal Spine') # Maybe with projection

        df_ceph.loc[patient, 'Posterior facial height'] = calculate_distance(df, patient, 'Sella', 'r-Gonion')                   # Maybe with projection

        # Cephalometric angles
        df_ceph.loc[patient, 'Mandibular angle L'] = calculate_angle_3p(df, patient, 'Porion L','l-Gonion','Menton')
        df_ceph.loc[patient, 'Mandibular angle R'] = calculate_angle_3p(df, patient, 'Porion R','r-Gonion','Menton')

        # Distances to planes
        df_ceph.loc[patient, ''] = point_to_plane_distance(df, patient, 'Nasion', 'MSP')

        # Cephalometric planes
        # df_ceph.loc[patient, 'FMA'] = angle_between_planes(df, patient, 'FHP', 'Mandibular plane') # Doesnt work yet

    return df_ceph




