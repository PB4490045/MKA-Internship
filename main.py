# Importing the necessary libraries
import preprocessing as pp
import ceph_analysis as ca
import stat_analysis as sa
import pandas as pd
import os

# =============================================================================

# Main function

# Define input and output folders
groundtruth_folder = r'Z:\TM Internships\Dept of CMF\Bram Roumen\Master Thesis - CMF\Thesis\Ground Truth database\Nifti - Segmentaties met landmarks'
predicted_folder   = r'Z:\\TM Internships\Dept of CMF\Bram Roumen\Master Thesis - CMF\Thesis\Inference workflow\Predicted patients'
# input_folder      = r'C:\Users\pb_va\OneDrive\Documents\Technical Medicine\TM2 - Stage 1 - MKA chirurgie\Bram Roumen\Inference workflow\Predicted patients'
output_folder      = 'Output'

# Create paths
groundtruth_path, predicted_path, output_path = pp.create_paths(groundtruth_folder, predicted_folder, output_folder)
# create_folders(input_path, output_path)

# # Create dataframes of landmarks
# df_pred, folder_names = pp.create_dataframe(predicted_path)
# df_gt = pp.create_dataframe_from_folders(groundtruth_path, folder_names)

# df_groundtruth = pp.create_planes(df_gt)
# df_predicted = pp.create_planes(df_pred)

# # # Save dataframes
# pp.save_df(df_groundtruth, output_path, 'df_groundtruth')
# pp.save_df(df_predicted, output_path, 'df_predicted')

# Load dataframes for testing
df_groundtruth = pp.loadcsv(output_path, 'df_groundtruth.csv')
df_predicted = pp.loadcsv(output_path, 'df_predicted.csv')

# # Perform Cephalometric Analysis
ceph_gt = ca.cephalometric_analysis(df_groundtruth)
ceph_p = ca.cephalometric_analysis(df_predicted)
print(ceph_gt)

df_stats = sa.paired_ttest(ceph_gt, ceph_p)
print(df_stats)

