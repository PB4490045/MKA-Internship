# Cephalometric Analysis Project

This project automates cephalometric analysis by processing anatomical landmarks, calculating distances and angles, performing statistical comparisons, and visualizing results in 3D. It includes modules for data preprocessing, cephalometric measurements, statistical analysis, and visualization. The current fucntions are easily adaptable to add more measurements. The code is free to download for everyone to use.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Modules](#modules)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

Clone the repository and create a Conda environment using the `environment.yml` file:

```bash
git clone https://github.com/PB4490045/MKA-Internship.git
cd cephalometric-analysis
conda env create -f environment.yml
conda activate mkaenv
```

### Prerequisites

This project requires Python 3.6 or higher and the following packages:
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`


## Project Structure

cephalometric-analysis/

│ 

├── main.py                    # Main script to run the analysis pipeline

├── preprocessing.py           # Functions for data preprocessing

├── ceph_analysis.py           # Functions for cephalometric measurements

├── stat_analysis.py           # Functions for statistical analysis

├── visualisation.py           # Functions for 3D visualization

└── README.md                  # Project documentation


## Usage

To run the full analysis pipeline:

1. **Define Input and Output Paths**: Update the paths to your ground truth and predicted datasets in `main.py`.

2. **Run `main.py`**:
```bash
python main.py
```

3. **Results**:
   - DataFrames with calculated measurements will be saved in the output folder.
   - 3D visualizations of landmarks and planes will be generated for selected patients.

## Modules

### `preprocessing.py`
This module processes and organizes landmark data from the ground truth and predicted datasets:
- `create_paths`: Generates absolute paths for the data folders.
- `create_dataframe`: Loads landmark data into a DataFrame.
- `create_planes`: Creates planes based on landmark coordinates for cephalometric analysis.
- `save_df`: Saves DataFrames to CSV files.
- `loadcsv`: Loads DataFrames from CSV files, handling JSON-like strings if necessary.

### `ceph_analysis.py`
This module calculates various cephalometric measurements:
- `calculate_distance`: Calculates the Euclidean distance between two landmarks.
- `calculate_angle_3p`: Calculates the angle formed by three landmarks.
- `calculate_angle_4p`: Calculates the angle between two lines, each defined by two landmarks.
- `angle_between_planes`: Calculates the angle between two planes based on their coefficients.
- `project_point_to_plane`: Projects a point onto a specified plane in 3D space.
- `point_to_plane_distance`: Calculates the perpendicular distance from a point to a plane.
- `cephalometric_analysis`: Executes the full cephalometric analysis for all patients.

### `stat_analysis.py`
This module performs statistical analysis to compare cephalometric measurements:
- `paired_ttest`: Conducts paired t-tests between two DataFrames.
- `wilcoxon_test`: Conducts Wilcoxon signed-rank tests between two DataFrames.
- `shapiro_test`: Performs the Shapiro-Wilk test for normality.
- `calculate_errors`: Calculates systematic and random errors between datasets.
- `combined_analysis`: Combines statistical analysis results for systematic comparison.

### `visualisation.py`
This module generates 3D visualizations of landmarks and planes:
- `landmark_array`: Creates a 3D array of landmarks for plotting.
- `plot_3d_points`: Plots 3D landmarks.
- `plot_3d_points_and_planes`: Plots 3D landmarks and overlays planes.

## Examples

### Running the Analysis

Here’s an example script that demonstrates running the analysis steps manually:

```python
import preprocessing as pp
import ceph_analysis as ca
import stat_analysis as sa
import visualisation as vis

# Set up paths and process data
groundtruth_folder = "path/to/groundtruth"
predicted_folder = "path/to/predicted"
output_folder = "path/to/output"

groundtruth_path, predicted_path, output_path = pp.create_paths(
    groundtruth_folder, predicted_folder, output_folder
)

# Load data and create planes
df_gt = pp.create_dataframe_from_folders(groundtruth_path, folder_names)
df_pred = pp.create_dataframe(predicted_path)

# Run cephalometric analysis
ceph_gt = ca.cephalometric_analysis(df_gt)
ceph_pred = ca.cephalometric_analysis(df_pred)

# Conduct statistical analysis
df_stats = sa.combined_analysis(ceph_gt, ceph_pred)

# Visualize 3D landmarks and planes
planes = ['Mandibular plane', 'Occlusal plane', 'FHP', 'MSP']
landmarks_to_plot = [landmark for landmark in df_gt.columns if landmark not in planes]
points = vis.landmark_array(df_gt, 'sample_patient_id', landmarks_to_plot)
vis.plot_3d_points_and_planes(points, df_gt, 'sample_patient_id', planes)
```

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request to propose improvements, fixes, or new features.

## License

This project is licensed under the MIT License.
```

This combined document provides everything from installation and usage instructions to a detailed overview of each module and an example of how to run the project. It’s organized to help new users understand the purpose of each part of the codebase and how to get started. Let me know if there are any further adjustments you’d like!
