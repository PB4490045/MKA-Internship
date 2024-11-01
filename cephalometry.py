import math
import numpy as np

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points in 3D space.

    Parameters:
    point1 (tuple): A tuple (x1, y1, z1) representing the coordinates of the first point.
    point2 (tuple): A tuple (x2, y2, z2) representing the coordinates of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    return math.sqrt((point1[0] - point2[0])**2 + 
                     (point1[1] - point2[1])**2 + 
                     (point1[2] - point2[2])**2)


def create_plane(p1, p2, p3):
    # Convert points to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    # Create two vectors from the points
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Compute the normal vector to the plane
    normal_vector = np.cross(v1, v2)
    
    # Compute the plane equation coefficients (A, B, C, D)
    A, B, C = normal_vector
    D = -np.dot(normal_vector, p1)
    
    return A, B, C, D

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle between three points in 3D space, with p2 as the vertex.

    Parameters:
    p1 (tuple): A tuple (x1, y1, z1) representing the coordinates of the first point.
    p2 (tuple): A tuple (x2, y2, z2) representing the coordinates of the vertex point.
    p3 (tuple): A tuple (x3, y3, z3) representing the coordinates of the third point.

    Returns:
    float: The angle in degrees between the three points.
    """
    # Convert points to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    # Create vectors from p2 to p1 and p2 to p3
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Compute the dot product and magnitudes of the vectors
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    # Compute the cosine of the angle
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    
    # Compute the angle in radians and then convert to degrees
    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees








# Example usage
point1 = (1.0, 2.0, 3.0)
point2 = (4.0, 5.0, 6.0)
point3 = (7.0, 8.0, 9.0)


plane = create_plane(point1, point2, point3)
print("Plane equation coefficients (A, B, C, D):", plane)

# Example usage
distance = euclidean_distance(point1, point2)
print(f"The distance between the points is: {distance}")

angle = calculate_angle(point1, point2, point3)
print(f"The angle between the points is: {angle} degrees")


