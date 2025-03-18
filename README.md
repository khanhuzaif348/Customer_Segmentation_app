# Customer_Segmentation_app


# Customer Segmentation using Clustering

## Overview
This Streamlit web application allows users to perform customer segmentation using the K-Means clustering algorithm. Users can upload a CSV file, process the data, and visualize the resulting clusters. The clustered data can be downloaded as a CSV file.

## Features
- **Upload CSV Dataset**: Users can upload their own dataset for clustering.
- **Automatic Data Processing**:
  - Selects numerical features automatically.
  - Handles missing values by replacing them with the mean.
  - Standardizes data using `StandardScaler`.
- **K-Means Clustering**:
  - Default number of clusters (`k=3`), which can be adjusted in the code.
  - Assigns each data point to a cluster.
- **Data Visualization**:
  - Displays raw and clustered data.
  - Provides a scatter plot of the clustered data.
- **Download Clustered Data**: Users can download the processed data with cluster labels.

## Installation
To run the application, ensure you have Python installed and then install the required dependencies:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

## How to Run
To launch the application, use the following command:

```bash
streamlit run app.py
```

## Usage
1. **Upload a CSV file** containing customer-related data.
2. Click the **"Process & Cluster Data"** button.
3. View the clustered data and scatter plot visualization.
4. Download the clustered dataset as a CSV file.

## Code Breakdown
### `main()`
- Sets up the Streamlit UI.
- Handles CSV file uploads.
- Calls the `cluster_customers()` function when the button is clicked.

### `cluster_customers(df)`
- Preprocesses the data (handling missing values, standardization).
- Applies K-Means clustering.
- Displays the clustered data and provides a download link.
- Calls `plot_clusters()` for visualization.

### `convert_df_to_csv(df)`
- Converts the DataFrame to a CSV format.

### `plot_clusters(num_df, clusters)`
- Generates a scatter plot to visualize clusters.

## Dependencies
- Python 3.x
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## License
This project is open-source and available under the MIT License.

## Author
Developed by Mohd Huzaif

