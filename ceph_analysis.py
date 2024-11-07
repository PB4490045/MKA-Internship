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
      If a landmark does not exist, returns NaN.
    """
    try:
        # Retrieve the coordinates for both landmarks
        point1 = df.loc[patient, landmark1]
        point2 = df.loc[patient, landmark2]
    except KeyError as e:
        print(f"Error: Landmark '{e.args[0]}' does not exist for patient '{patient}'. Returning NaN.")
        return np.nan  # Return NaN if a landmark doesn't exist

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

def angle_between_planes(plane1_coefficients, plane2_coefficients):
    """
    Calculate the angle between two planes given their coefficients.

    Parameters:
    - plane1_coefficients: List or array of the plane coefficients [A, B, C, D].
    - plane2_coefficients: List or array of the plane coefficients [A, B, C, D].

    Returns:
    - The angle between the two planes in degrees.
    """
    # Extract the normal vectors from the plane coefficients
    normal1 = np.array(plane1_coefficients[:3])  # [A, B, C] from the first plane
    normal2 = np.array(plane2_coefficients[:3])  # [A, B, C] from the second plane

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

def cephalometric_analysis(output_path, filename, patient):

    # For now for quick working
    df = pp.loadcsv(output_path, filename)

    df_ceph = pd.DataFrame(index=[patient])

    # Cephalometric analysis
    df_ceph['cond_w'] = calculate_distance(df, patient, 'r-Condyle', 'l-Condyl')
    df_ceph['cor_w']  = calculate_distance(df, patient, 'r-Coronoid', 'l-Coronoid')
    df_ceph['zyg_w']  = calculate_distance(df, patient, 'Zygomatic Process R', 'Zygomatic Process L')
    print(df_ceph)




