# Importing the necessary libraries
import preprocessing as pp
import ceph_analysis as ca

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

# Create dataframes of landmarks
# df_predicted = pp.create_dataframe(predicted_path, output_path, 'df_predicted')
# df_groundtruth = pp.create_dataframe(groundtruth_path, output_path, 'df_groundtruth')

#Load dataframes for testing
df_groundtruth = pp.loadcsv(output_path, 'df_groundtruth.csv')
df_predicted = pp.loadcsv(output_path, 'df_predicted.csv')

# Perform Cephalometric Analysis
ceph_gt = ca.cephalometric_analysis(output_path, 'df_groundtruth.csv', 'ma_007')
ceph_p = ca.cephalometric_analysis(output_path, 'df_predicted.csv', 'ma_007')
print(ceph_gt)
print(ceph_p)


