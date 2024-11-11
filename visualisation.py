import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd


def landmark_array(df, patient, landmarks):
    """
    Creates a list of NumPy arrays representing points for a single patient,
    excluding any landmarks with missing values.
    
    Parameters:
    df (DataFrame): DataFrame containing coordinate information.
    patient (str or int): The identifier for the patient (index value in the DataFrame).
    landmarks (list): List of landmark identifiers (column names in the DataFrame).
    
    Returns:
    list: A list of NumPy arrays, each representing a 3D point [x, y, z].
    """
    points = []
    for landmark in landmarks:
        # Extract the coordinates for the given patient and landmark
        coords = df.loc[patient, landmark]
        
        # Ensure that coords is a list or array of [x, y, z] values and check for NaNs
        if isinstance(coords, (list, np.ndarray)) and not any(pd.isna(coord) for coord in coords):
            # Convert coordinates to a NumPy array and add to points list
            point = np.array(coords)
            points.append(point)
        else:
            # Print message if the landmark is skipped
            print(f'Skipping landmark "{landmark}" for patient "{patient}": Invalid or incomplete data.')
    
    return points

def plot_3d_points(points):
    """
    Plots 3D points without projections onto the XY, XZ, or YZ planes.

    Parameters:
    points (list of np.array): List of points, each as a numpy array with x, y, and z coordinates.
    """
    # Extract x, y, and z coordinates from the points
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    z = [point[2] for point in points]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for the points
    ax.scatter(x, y, z, color='b', s=50, label="3D Points")

    # Labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Show plot
    plt.show()

def plot_3d_points_and_planes(points, df, patient, planes, x_range=None, y_range=None, z_range=None):
    """
    Plots 3D points and planes in a 3D space with predefined axis ranges, 
    including handling the 'MSP' vertical plane by using y and z ranges.

    Parameters:
    points (list of np.array): List of points, each as a numpy array with x, y, and z coordinates.
    df (DataFrame): DataFrame containing plane coefficients.
    patient (str or int): The identifier for the patient in the DataFrame.
    planes (list): List of plane identifiers (column names in the DataFrame) containing plane coefficients [A, B, C, D].
    x_range, y_range, z_range (tuple): Optional tuples specifying (min, max) for the x, y, and z axes.
    """
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    z = [point[2] for point in points]
    ax.scatter(x, y, z, color='b', s=50, label="3D Points")

    # Set axis ranges using specified ranges or defaults if none provided
    x_min, x_max = x_range if x_range else (-70, 70)
    y_min, y_max = y_range if y_range else (-70, 70)
    z_min, z_max = z_range if z_range else (-70, 70)

    # Apply the axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Add "dummy" points to create an equal aspect ratio
    ax.plot([x_min, x_max], [y_min, y_min], [z_min, z_min], color='w', alpha=0)  # Invisible line
    ax.plot([x_min, x_min], [y_min, y_max], [z_min, z_min], color='w', alpha=0)  # Invisible line
    ax.plot([x_min, x_min], [y_min, y_min], [z_min, z_max], color='w', alpha=0)  # Invisible line

    for plane in planes:
        # Get the plane coefficients [A, B, C, D] for the specified patient
        A, B, C, D = np.array(df.loc[patient, plane])

        # Special handling for the vertical 'MSP' plane, solving for X
        if plane == 'MSP':
            # Use y and z ranges for meshgrid and solve for X
            y_vals = np.linspace(y_min, y_max, 10)
            z_vals = np.linspace(z_min, z_max, 10)
            Y, Z = np.meshgrid(y_vals, z_vals)
            X = (-B * Y - C * Z - D) / A  # Solve for X
        else:
            # Standard handling for other planes using x and y ranges
            x_vals = np.linspace(x_min, x_max, 10)
            y_vals = np.linspace(y_min, y_max, 10)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = (-A * X - B * Y - D) / C  # Solve for Z

        ax.plot_surface(X, Y, Z, alpha=0.5, edgecolor='none')
        ax.plot([], [], [], label=f"{plane}")

    # Labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()

    # Show plot
    plt.show()
