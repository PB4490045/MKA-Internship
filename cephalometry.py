import math

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

# Example usage
landmark1 = (1.0, 2.0, 3.0)
landmark2 = (4.0, 5.0, 6.0)
distance = euclidean_distance(landmark1, landmark2)
print(f"The distance between the landmarks is: {distance}")