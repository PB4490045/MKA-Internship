"""
visualisation.py

This module provides functions to visualize 3D points and planes based on cephalometric data. 
It includes utilities for plotting landmark points in 3D space and overlaying planes using 
predefined axis ranges.

Functions:
- landmark_array(df, patient, landmarks): Create a list of 3D points as NumPy arrays for a patient, 
  excluding landmarks with missing data.
- plot_3d_points(points): Plot 3D points in space without projections onto the XY, XZ, or YZ planes.
- plot_3d_points_and_planes(points, df, patient, planes, x_range=None, y_range=None, z_range=None): 
  Plot 3D points with planes in 3D space, with optional axis ranges for x, y, and z.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd


def landmark_array(df, patient, landmarks):
    """
    Create a list of NumPy arrays representing points for a single patient, excluding any landmarks 
    with missing values.

    Parameters:
    - df (pd.DataFrame): DataFrame containing coordinate information.
    - patient (str or int): Identifier for the patient (index value in the DataFrame).
    - landmarks (list): List of landmark identifiers (column names in the DataFrame).

    Returns:
    - list: A list of NumPy arrays, each representing a 3D point [x, y, z].
    """
    points = []
    for landmark in landmarks:
        coords = df.loc[patient, landmark]

        if isinstance(coords, (list, np.ndarray)) and not any(pd.isna(coord) for coord in coords):
            point = np.array(coords)
            points.append(point)
        else:
            print(f'Skipping landmark "{landmark}" for patient "{patient}": Invalid or incomplete data.')

    return points


def plot_3d_points(points):
    """
    Plot 3D points without reference planes in a 3D space.

    Parameters:
    - points (list of np.array): List of points, each as a numpy array with x, y, and z coordinates.
    """
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    z = [point[2] for point in points]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color='b', s=50, label="3D Points")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def plot_3d_points_and_planes(points, df, patient, planes, x_range=None, y_range=None, z_range=None):
    """
    Plot 3D points and planes in a 3D space with predefined axis ranges, including handling for a 
    vertical plane 'MSP' using y and z ranges.

    Parameters:
    - points (list of np.array): List of points, each as a numpy array with x, y, and z coordinates.
    - df (pd.DataFrame): DataFrame containing plane coefficients.
    - patient (str or int): Identifier for the patient in the DataFrame.
    - planes (list): List of plane identifiers (column names in the DataFrame) with coefficients 
      [A, B, C, D].
    - x_range (tuple, optional): Tuple (min, max) specifying the range for the x-axis.
      Default is (-70, 70).
    - y_range (tuple, optional): Tuple (min, max) specifying the range for the y-axis. 
      Default is (-70, 70).
    - z_range (tuple, optional): Tuple (min, max) specifying the range for the z-axis. 
      Default is (-70, 70).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = [point[0] for point in points]
    y = [point[1] for point in points]
    z = [point[2] for point in points]
    ax.scatter(x, y, z, color='b', s=50, label="3D Points")

    x_min, x_max = x_range if x_range else (-70, 70)
    y_min, y_max = y_range if y_range else (-70, 70)
    z_min, z_max = z_range if z_range else (-70, 70)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.plot([x_min, x_max], [y_min, y_min], [z_min, z_min], color='w', alpha=0)
    ax.plot([x_min, x_min], [y_min, y_max], [z_min, z_min], color='w', alpha=0)
    ax.plot([x_min, x_min], [y_min, y_min], [z_min, z_max], color='w', alpha=0)

    for plane in planes:
        a, b, c, d = np.array(df.loc[patient, plane])

        if plane == 'MSP':
            y_vals = np.linspace(y_min, y_max, 10)
            z_vals = np.linspace(z_min, z_max, 10)
            Y, Z = np.meshgrid(y_vals, z_vals)
            X = (-b * Y - c * Z - d) / a
        else:
            x_vals = np.linspace(x_min, x_max, 10)
            y_vals = np.linspace(y_min, y_max, 10)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = (-a * X - b * Y - d) / c

        ax.plot_surface(X, Y, Z, alpha=0.5, edgecolor='none', label=f"{plane}")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()

    plt.show()
