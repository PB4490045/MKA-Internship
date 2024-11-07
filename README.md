# MKA-Internship
Repository containing all files and code for the 3D cephalometric analysis performed on predicted landmarks

main.py is the function and is used to call the different files to perform the cephalometric analysis. 

preprocessing.py is used to define all necessary paths, and create dataframes with coordinates of the landmarks. Currently it also creates a csv file of the dataframe to make testing easier. After initial creation of the dataframe the .csv can be loaded in using the designated function to cut processing time significantly. 

ceph_analysis.py is used to perform the cephalometric analysis. The calculated measurements are stored in a dataframe. The dataframe is currently not saved as.csv file, but can be added shortly. 

stat_analyis.py is used to perform a statistical analysis over the performed cephalometric analysis between the ground truth and predicted landmarks. 

All functions contain docstrings to further explain their function. 


