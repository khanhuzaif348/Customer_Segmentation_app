import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import io

# Streamlit UI
def main():
    st.title("Customer Segmentation using Clustering")
    st.sidebar.header("Upload Dataset")
    
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Raw Data Preview:")
        st.dataframe(df.head())
        
        if st.sidebar.button("Process & Cluster Data"):
            cluster_customers(df)

# Clustering function
def cluster_customers(df):
    st.write("### Data Processing & Clustering")
    
    # Select numerical features
    num_df = df.select_dtypes(include=[np.number])
    
    # Handle missing values
    num_df = num_df.fillna(num_df.mean())
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(num_df)
    
    # K-Means Clustering
    k = 3  # Default number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    df['Cluster'] = clusters
    st.write("### Clustered Data")
    st.dataframe(df.head())
    
    # Save the clustered data to CSV and provide download link
    csv = convert_df_to_csv(df)
    st.download_button(
        label="Download Clustered Data as CSV",
        data=csv,
        file_name="clustered_data.csv",
        mime="text/csv"
    )
    
    # Visualization
    plot_clusters(num_df, clusters)

# Convert DataFrame to CSV
def convert_df_to_csv(df):
    csv = df.to_csv(index=False)
    return io.StringIO(csv).getvalue()

# Visualization function
def plot_clusters(num_df, clusters):
    st.write("### Cluster Distribution")
    
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=num_df.iloc[:, 0], y=num_df.iloc[:, 1], hue=clusters, palette='viridis')
    plt.xlabel(num_df.columns[0])
    plt.ylabel(num_df.columns[1])
    st.pyplot(plt)

if __name__ == "__main__":
    main()
