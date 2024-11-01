# Importing the necessary libraries
import os
import json

# Function to create the paths
def create_paths(input_folder, output_folder):
    input_path = os.path.abspath(input_folder)
    output_path = os.path.abspath(output_folder)
    return input_path, output_path

# Function to create output folders
def create_folders(input_path, output_path):
    folders = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))]

    for folder in folders:
        output_folder_path = os.path.join(output_path, folder)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
            print(f"Created folder: {output_folder_path}")
        else:
            print(f"Folder already exists: {output_folder_path}")
# =============================================================================

# Function to get all folder names in the input folder
def get_folder_names(input_path):

    coordinates_dict = {}

    folder_names = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))]
    for folder in folder_names:
        folder_path = os.path.join(input_path, folder)

        # Loop through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                # Extract point name from the filename (assumes the name is before the first underscore or dot)
                point_name = filename.split('.')[0]
                
                # Load JSON file
                file_path = os.path.join(folder_path, filename)
                with open(file_path) as f:
                    data = json.load(f)
                
                # Extract position coordinates
                try:
                    position = data["markups"][0]["controlPoints"][0]["position"]
                    coordinates_dict[point_name] = position
                except (KeyError, IndexError) as e:
                    print(f"Error in file {filename}: {e}")        
        coords_dict[f'{folder}'] = coordinates_dict
    print(coords_dict)
    return coordinates_dict



# =============================================================================


# Define the input and output folders
input_folder = r'Z:\\TM Internships\Dept of CMF\Bram Roumen\Master Thesis - CMF\Thesis\Inference workflow\Predicted patients'
output_folder = 'Output'
input_json = r'Z:\\TM Internships\Dept of CMF\Bram Roumen\Master Thesis - CMF\Thesis\Inference workflow\Predicted patients\ma_006'

# Main function
input_path, output_path = create_paths(input_folder, output_folder)
create_folders(input_path, output_path)


coords_dict = {}

# # Function for creating a dictionary of coordinates of landmarks for all patients in the same dictionary
# def create_dict(input_json):

#     # Dictionary to store coordinates with point names as keys
#     coordinates_dict = {}

#     # Loop through all files in the folder
#     for filename in os.listdir(input_json):
#         if filename.endswith('.json'):
#             # Extract point name from the filename (assumes the name is before the first underscore or dot)
#             point_name = filename.split('.')[0]
            
#             # Load JSON file
#             file_path = os.path.join(input_json, filename)
#             with open(file_path) as f:
#                 data = json.load(f)
            
#             # Extract position coordinates
#             try:
#                 position = data["markups"][0]["controlPoints"][0]["position"]
#                 coordinates_dict[point_name] = position
#             except (KeyError, IndexError) as e:
#                 print(f"Error in file {filename}: {e}")
#     coords_dict['ma_006'] = coordinates_dict
#     #print(coordinates_dict)
#     print(coords_dict)
#     return coordinates_dict

get_folder_names(input_path)





