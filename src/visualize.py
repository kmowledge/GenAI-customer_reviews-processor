import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_processed_data(processed_file):
    """Load the processed reviews from CSV"""
    return pd.read_csv(processed_file)


def load_locations_data(locations_file):
    """Load the extracted locations from CSV"""
    return pd.read_csv(locations_file)


def visualize_sentiment_metrics(data):
    """Visualize sentiment analysis metrics"""
    # Create confusion matrix
    y_true = [1 if sentiment == "positive" else 0 for sentiment in data["ground_truth"]]
    y_pred = [1 if sentiment == "positive" else 0 for sentiment in data["sentiment"]]

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    # Display confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Positive", "Negative"]
    )
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Sentiment Analysis Confusion Matrix")
    plt.savefig("../data/confusion_matrix.png")
    plt.close()

    # Calculate metrics
    true_positive = cm[0, 0]
    false_positive = cm[1, 0]
    false_negative = cm[0, 1]
    true_negative = cm[1, 1]

    accuracy = (true_positive + true_negative) / (
        true_positive + true_negative + false_positive + false_negative
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Create metrics bar chart
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    values = [accuracy, precision, recall, f1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(metrics, values, color=["blue", "green", "orange", "red"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Sentiment Analysis Metrics")

    # Add value labels on top of bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.4f}", ha="center")

    plt.savefig("../data/metrics_chart.png")
    plt.close()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def visualize_satisfaction_scores(data):
    """Visualize customer satisfaction scores by sentiment"""
    # Group by detected sentiment and calculate average satisfaction score
    sentiment_groups = data.groupby("sentiment")["satisfaction_score"].mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    sentiment_groups.plot(kind="bar", color=["green", "red"], ax=ax)
    ax.set_ylabel("Average Satisfaction Score")
    ax.set_title("Average Satisfaction Score by Detected Sentiment")

    for i, v in enumerate(sentiment_groups):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center")

    plt.savefig("../data/satisfaction_by_sentiment.png")
    plt.close()


def visualize_locations(locations_data):
    """Visualize extracted locations"""
    # Count occurrences of each location
    location_counts = locations_data["location"].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(12, 8))
    location_counts.plot(kind="bar", ax=ax)
    ax.set_title("Top 10 Mentioned Locations")
    ax.set_ylabel("Count")
    ax.set_xlabel("Location")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("../data/top_locations.png")
    plt.close()

    # Count occurrences of each entity type
    entity_counts = locations_data["entity_type"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 8))
    entity_counts.plot(kind="pie", autopct="%1.1f%%")
    ax.set_title("Distribution of Location Entity Types")
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig("../data/entity_types.png")
    plt.close()


if __name__ == "__main__":
    # File paths
    processed_file = "../data/generated/processed_reviews.csv"
    locations_file = "../data/generated/extracted_locations.csv"

    # Load data
    processed_data = load_processed_data(processed_file)
    locations_data = load_locations_data(locations_file)

    # Generate visualizations
    metrics = visualize_sentiment_metrics(processed_data)
    visualize_satisfaction_scores(processed_data)
    visualize_locations(locations_data)

    print("\nSentiment Analysis Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")

    print("\nVisualizations saved to data directory:")
