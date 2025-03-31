import json
import csv
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableBranch
import spacy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv

# python -m spacy download en_core_web_sm

# Load environment variables from .env file
load_dotenv()

# Initialize LLM with API key from environment
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=api_key)

# Initialize spaCy for NER
nlp = spacy.load("en_core_web_sm")

# Define the sentiment analysis response format
sentiment_parser = JsonOutputParser()

# Sentiment analysis prompt
sentiment_prompt = PromptTemplate(
    template="""You are a sentiment analysis expert. 
Review the following customer review and determine if it's positive or negative.

Review: ```{review}```

Return answer as a valid json object with the following format:
{{"positive_sentiment": boolean, "reasoning": string}}
""",
    input_variables=["review"],
)

# Create sentiment analysis chain
sentiment_chain = sentiment_prompt | llm | sentiment_parser

# Positive review handling
positive_prompt = PromptTemplate(
    template="""You are a customer service representative for a travel company.
A customer has left a positive review about their trip.

Customer Review: {review}

Based on the review, identify what they specifically liked and create a personalized response that:
1. Thanks them for their positive feedback
2. Offers a voucher related to the thing they liked
3. Recommends 3 specific trips based on their interests shown in the review

Return your response as a valid JSON object with the following format:
{{"message": str, "recommended_trips": [{{"destination": str, "description": str}}, {{"destination": str, "description": str}}, {{"destination": str, "description": str}}]}}

Make sure to add appropriate newline after approximately every 80 characters for better readability.
""",
    input_variables=["review"],
)


# Use JsonOutputParser to get properly formatted JSON output
positive_chain = positive_prompt | llm | JsonOutputParser()

# Negative review handling
negative_prompt = PromptTemplate(
    template="""You are a customer service representative for a travel company.
A customer has left a negative review about their trip.

Customer Review: {review}

Based on the review, identify what they specifically disliked and create a personalized response that:
1. Apologizes for their negative experience
2. Addresses the specific issue they mentioned
3. Explains how you'll mitigate this issue in the future
4. Offers a 25% discount on their next visit
5. Thanks them for their feedback

Return your response as a valid JSON object with the following format:
{{"message": str, "discount_code": str}}

Create a personalized discount code that includes a reference to the location they visited 
and a random 4-digit number.

Make sure to add appropriate newline after approximately every 80 characters for better readability.
""",
    input_variables=["review"],
)

negative_chain = negative_prompt | llm | JsonOutputParser()

# Create the branching logic with RunnablePassthrough
full_chain = RunnablePassthrough.assign(
    sentiment_result=sentiment_chain
) | RunnableBranch(
    (
        lambda x: x["sentiment_result"]["positive_sentiment"],
        RunnablePassthrough.assign(review=lambda x: x["review"]) | positive_chain,
    ),
    (
        lambda x: not x["sentiment_result"]["positive_sentiment"],
        RunnablePassthrough.assign(review=lambda x: x["review"]) | negative_chain,
    ),
    # Default fallback
    lambda x: {"message": f"Error: Unable to determine sentiment for: {x['review']}"},
)


# Extract locations using NER
def extract_locations(text):
    doc = nlp(text)
    locations = []

    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC", "FAC"]:
            locations.append({"text": ent.text, "label": ent.label_})

    return locations


# Process customer reviews
def process_customer_reviews(input_file, output_file, locations_file):
    # Load customer reviews
    with open(input_file, "r") as f:
        data = json.load(f)

    # Process each review
    results = []
    locations_data = []
    ground_truth = []
    predictions = []

    for review_data in data:
        review = review_data["review"]

        # Extract sentiment ground truth
        ground_truth_sentiment = (
            True if review_data["survey_sentiment"] == "positive" else False
        )
        ground_truth.append(ground_truth_sentiment)

        # Process review
        result = full_chain.invoke({"review": review})

        # Check if sentiment analysis was successful
        sentiment_result = sentiment_chain.invoke({"review": review})
        predictions.append(sentiment_result["positive_sentiment"])

        # Extract locations
        locations = extract_locations(review)
        for loc in locations:
            locations_data.append(
                {
                    "review_id": len(results),
                    "location": loc["text"],
                    "entity_type": loc["label"],
                }
            )

        # Combine results
        result["review"] = review
        result["sentiment"] = (
            "positive" if sentiment_result["positive_sentiment"] else "negative"
        )
        result["reasoning"] = sentiment_result["reasoning"]
        result["ground_truth"] = review_data["survey_sentiment"]
        result["satisfaction_score"] = review_data["customer_satisfaction_score"]

        results.append(result)

    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    # Save results to CSV
    save_results_to_csv(results, output_file)

    # Save locations to CSV
    save_locations_to_csv(locations_data, locations_file)

    return results, metrics


def save_results_to_csv(results, output_file):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header
        if results and len(results) > 0:
            first_result = results[0]
            headers = [
                "review",
                "sentiment",
                "reasoning",
                "ground_truth",
                "satisfaction_score",
            ]

            if "message" in first_result:
                headers.append("message")

            if "recommended_trips" in first_result:
                headers.extend(
                    ["recommended_trip_1", "recommended_trip_2", "recommended_trip_3"]
                )

            if "discount_code" in first_result:
                headers.append("discount_code")

            writer.writerow(headers)

            # Write data rows
            for result in results:
                row = [
                    result["review"],
                    result["sentiment"],
                    result["reasoning"],
                    result["ground_truth"],
                    result["satisfaction_score"],
                ]

                if "message" in result:
                    row.append(result["message"])

                if "recommended_trips" in result:
                    trips = result["recommended_trips"]
                    for i in range(min(3, len(trips))):
                        row.append(
                            f"{trips[i]['destination']}: {trips[i]['description']}"
                        )

                    # Fill empty trip slots if less than 3 trips
                    for i in range(len(trips), 3):
                        row.append("")

                if "discount_code" in result:
                    row.append(result["discount_code"])

                writer.writerow(row)


def save_locations_to_csv(locations, output_file):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["review_id", "location", "entity_type"])

        for location in locations:
            writer.writerow(
                [location["review_id"], location["location"], location["entity_type"]]
            )


def visualize_metrics(metrics):
    # This function could be expanded to create visualizations of the metrics
    print("\nSentiment Analysis Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    # File paths
    input_file = "../data/customer_surveys.json"
    output_file = "../data/generated/processed_reviews.csv"
    locations_file = "../data/generated/extracted_locations.csv"

    # Process customer reviews
    results, metrics = process_customer_reviews(input_file, output_file, locations_file)

    # Display metrics
    visualize_metrics(metrics)

    print(f"\nProcessed {len(results)} customer reviews")
    print(f"Results saved to {output_file}")
    print(f"Extracted locations saved to {locations_file}")
