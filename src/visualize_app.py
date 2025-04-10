import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import chardet
import os

# Set page configuration
st.set_page_config(
    page_title="Customer Reviews Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add title and description
st.title("Customer Reviews Analysis Dashboard")
st.markdown("""
This dashboard visualizes sentiment analysis results and location data from customer reviews.
""")

# File paths
processed_file = "../data/generated/processed_reviews.csv"
locations_file = "../data/generated/extracted_locations.csv"

# Check if files exist
if not os.path.exists(processed_file) or not os.path.exists(locations_file):
    st.error("Data files not found. Please make sure the processed_reviews.csv and extracted_locations.csv files exist in the data/generated directory.")
    st.stop()

# Load data functions
def load_processed_data(processed_file):
    """Load the processed reviews from CSV with encoding detection"""
    with open(processed_file, "rb") as f:
        result = chardet.detect(f.read())
    encoding = result["encoding"]
    st.info(f"Detected encoding for processed reviews: {encoding}")
    return pd.read_csv(processed_file, encoding=encoding)

def load_locations_data(locations_file):
    """Load the extracted locations from CSV with encoding detection"""
    with open(locations_file, "rb") as f:
        result = chardet.detect(f.read())
    encoding = result["encoding"]
    st.info(f"Detected encoding for locations data: {encoding}")
    return pd.read_csv(locations_file, encoding=encoding)

# Load data
with st.spinner("Loading data..."):
    processed_data = load_processed_data(processed_file)
    locations_data = load_locations_data(locations_file)

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Sentiment Analysis", "Satisfaction Scores", "Location Analysis"])

# Tab 1: Sentiment Analysis
with tab1:
    st.header("Sentiment Analysis Metrics")
    
    # Create two columns for the confusion matrix and metrics
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Create confusion matrix
        y_true = [1 if sentiment == "positive" else 0 for sentiment in processed_data["ground_truth"]]
        y_pred = [1 if sentiment == "positive" else 0 for sentiment in processed_data["sentiment"]]
        
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        
        # Display confusion matrix
        fig, ax = plt.subplots(figsize=(3, 2))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Positive", "Negative"])
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title("Sentiment Analysis Confusion Matrix", fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        st.pyplot(fig)
    
    with col2:
        # Calculate metrics
        true_positive = cm[0, 0]
        false_positive = cm[1, 0]
        false_negative = cm[0, 1]
        true_negative = cm[1, 1]
        
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Create metrics bar chart
        metrics = ["Accuracy", "Precision", "Recall", "F1"]
        values = [accuracy, precision, recall, f1]
        
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.bar(metrics, values, color=["blue", "green", "orange", "red"])
        ax.set_ylim(0, 1.0)
        plt.xticks(rotation=45, fontsize=6)
        plt.yticks(fontsize=6)
        plt.title("Performance Metrics", fontsize=8)
        st.pyplot(fig)

# Tab 2: Satisfaction Scores
with tab2:
    st.header("Customer Satisfaction Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Group by detected sentiment and calculate average satisfaction score
        sentiment_groups = processed_data.groupby("sentiment")["satisfaction_score"].mean()
        
        fig, ax = plt.subplots(figsize=(3, 2))
        sentiment_groups.plot(kind="bar", color=["green", "red"], ax=ax)
        ax.set_ylabel("Avg Score", fontsize=6)
        plt.xticks(rotation=45, fontsize=6)
        plt.yticks(fontsize=6)
        plt.title("Satisfaction by Sentiment", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # Display raw data
        st.subheader("Raw Data Sample")
        st.dataframe(processed_data[["sentiment", "satisfaction_score"]].head(5))

# Tab 3: Location Analysis
with tab3:
    st.header("Location Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Count occurrences of each location
        location_counts = locations_data["location"].value_counts().head(5)
        
        fig, ax = plt.subplots(figsize=(3, 2))
        location_counts.plot(kind="bar", ax=ax)
        plt.title("Top 5 Locations", fontsize=8)
        plt.xticks(rotation=45, fontsize=6)
        plt.yticks(fontsize=6)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # Count occurrences of each entity type
        entity_counts = locations_data["entity_type"].value_counts()
        
        fig, ax = plt.subplots(figsize=(3, 2))
        entity_counts.plot(kind="pie", autopct=lambda p: f'{p:.0f}%' if p > 5 else '')
        plt.title("Location Types", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

# Add sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This dashboard provides visualizations for:
    
    - Sentiment analysis results
    - Customer satisfaction scores
    - Location mentions in reviews
    
    The data is loaded from CSV files in the data/generated directory.
    """)
    
    st.header("Data Summary")
    st.write(f"Processed Reviews: {len(processed_data)}")
    st.write(f"Extracted Locations: {len(locations_data)}")
    
    # Add download buttons for the data
    st.header("Download Data")
    
    csv_processed = processed_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Processed Reviews",
        data=csv_processed,
        file_name="processed_reviews.csv",
        mime="text/csv",
    )
    
    csv_locations = locations_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Location Data",
        data=csv_locations,
        file_name="extracted_locations.csv",
        mime="text/csv",
    )