"""
main.py

This main module coordinates the overall cephalometric analysis workflow. It handles preprocessing, 
landmark and plane data extraction, cephalometric measurements, and statistical analysis. Results 
are saved to CSV files, and visualizations of 3D landmarks and planes are generated for selected 
patients.

Workflow:
1. Define ground truth and predicted data paths and create output directories.
2. Load and preprocess landmark data from specified folders.
3. Create landmark planes and add them to the ground truth and predicted DataFrames.
4. Save the processed DataFrames to the output folder.
5. Perform cephalometric analysis on ground truth and predicted DataFrames.
6. Conduct statistical analysis to compare cephalometric measurements.
7. Save statistical analysis results and generate 3D visualizations for selected patients.

Modules Required:
- preprocessing.py
- ceph_analysis.py
- stat_analysis.py
- visualisation.py
"""

import preprocessing as pp
import ceph_analysis as ca
import stat_analysis as sa
import visualisation as vis


# Define input and output folders
GROUNDTRUTH_FOLDER = (
    r'Z:\TM Internships\Dept of CMF\Bram Roumen\Master Thesis - CMF\Thesis\Ground Truth database'
    r'\Nifti - Segmentaties met landmarks'
)
PREDICTED_FOLDER = (
    r'Z:\TM Internships\Dept of CMF\Bram Roumen\Master Thesis - CMF\Thesis\Inference workflow'
    r'\Predicted patients'
)
OUTPUT_FOLDER = 'Output'

# Create paths
groundtruth_path, predicted_path, output_path = pp.create_paths(
    GROUNDTRUTH_FOLDER, PREDICTED_FOLDER, OUTPUT_FOLDER
)

# Create DataFrames of landmarks
df_pred, folder_names = pp.create_dataframe(predicted_path)
df_gt = pp.create_dataframe_from_folders(groundtruth_path, folder_names)

# Create planes and add to DataFrames
df_groundtruth = pp.create_planes(df_gt)
df_predicted = pp.create_planes(df_pred)

# Save DataFrames
pp.save_df(df_groundtruth, output_path, 'df_groundtruth')
pp.save_df(df_predicted, output_path, 'df_predicted')

# Perform Cephalometric Analysis
ceph_gt = ca.cephalometric_analysis(df_groundtruth)
ceph_p = ca.cephalometric_analysis(df_predicted)

# Save DataFrames of cephalometric analyses
pp.save_df(ceph_gt, output_path, 'ceph_gt')
pp.save_df(ceph_p, output_path, 'ceph_p')

# Perform statistical analysis
df_stats = sa.combined_analysis(ceph_gt, ceph_p)

# Save DataFrame of statistical analysis
pp.save_df(df_stats, output_path, 'df_stats')

# List of planes to plot landmarks and planes separately
planes = ['Mandibular plane', 'Occlusal plane', 'FHP', 'MSP']

# Use set operations to exclude plane landmarks from points to plot
landmarks_to_plot = [landmark for landmark in df_groundtruth.columns if landmark not in planes]

# Generate points with the filtered landmarks
points_gt = vis.landmark_array(df_groundtruth, 'ma_009', landmarks_to_plot)

# Plot the 3D points with planes
vis.plot_3d_points_and_planes(points_gt, df_groundtruth, 'ma_009', planes)
