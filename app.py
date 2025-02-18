import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from PIL import Image

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
    
    # Add cluster column to original DataFrame
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
    
    # Create and provide a presentation download button
    pdf_report = generate_pdf_report(df)
    st.download_button(
        label="Download Presentation for Stakeholders",
        data=pdf_report,
        file_name="customer_segmentation_report.pdf",
        mime="application/pdf"
    )
    
    # Visualization
    plot_clusters(num_df, clusters)

# Convert DataFrame to CSV
def convert_df_to_csv(df):
    csv = df.to_csv(index=False)
    return io.StringIO(csv).getvalue()

# Generate PDF report using ReportLab
def generate_pdf_report(df):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Customer Segmentation Report")
    
    # Description
    c.setFont("Helvetica", 12)
    text = "This report summarizes the clustering analysis on customer data. Customers have been segmented into clusters based on various features."
    c.drawString(100, 730, text)
    
    # Cluster Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, 700, "Cluster Summary:")
    
    # Ensure only numeric columns are included
    numeric_df = df.select_dtypes(include=[np.number])
    cluster_summary = numeric_df.groupby('Cluster').mean()
    cluster_summary_str = cluster_summary.to_string()
    
    # Write cluster summary in the PDF
    c.setFont("Helvetica", 10)
    c.drawString(100, 680, cluster_summary_str)
    
    # Add some sample visualizations
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, 640, "Cluster Distribution Chart:")
    
    # Create scatter plot for clusters
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df['Cluster'], palette='viridis', ax=ax)
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    plt.tight_layout()
    
    # Save the plot to a byte stream
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    img_stream.close()
    
    # Save image as PIL Image for use in ReportLab
    img = Image.open(img_stream)
    img_path = "/tmp/temp_plot.png"
    img.save(img_path)

    # Add the image to the PDF using the image path
    c.drawImage(img_path, 100, 400, width=400, height=200)
    
    # Save the PDF to the buffer and return it as bytes
    c.save()
    buffer.seek(0)
    return buffer.read()

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
