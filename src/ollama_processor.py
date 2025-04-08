import concurrent
import csv
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore
from typing import Any, Dict, List

import spacy
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from ollama import Client
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize Ollama client
client = Client(host='http://localhost:11434')
model_name = "llama3:8b"

# spaCy for NER
nlp = spacy.load("en_core_web_sm")

# Templates
sentiment_prompt = PromptTemplate(
    template="""You are a sentiment analysis expert.
Review the following customer review and determine if it's positive or negative.

Review: ```{review}```

Return only valid JSON with this format:
{{"positive_sentiment": boolean, "reasoning": string}}

IMPORTANT: Output ONLY JSON. No extra text.
""",
    input_variables=["review"],
)

positive_prompt = PromptTemplate(
    template="""You are a customer service representative for a travel company.
A customer left a positive review about their trip.

Customer Review: {review}

Based on the review, identify what they specifically liked and create a personalized response that:
1. Thanks them for their feedback
2. Offers a voucher related to what they liked
3. Recommends 3 trips based on their interests

Return ONLY valid JSON:
{{"message": str, "recommended_trips": [{{"destination": str, "description": str}}, ...]}}

IMPORTANT: Output ONLY JSON. No other text.
""",
    input_variables=["review"],
)

negative_prompt = PromptTemplate(
    template="""You are a customer service representative for a travel company.
A customer left a negative review about their trip.

Customer Review: {review}

Identify what they disliked and create a personalized response that:
1. Apologizes
2. Addresses the issue
3. Offers a 25% discount with a custom discount code
4. Thanks for feedback

Return ONLY valid JSON:
{{"message": str, "discount_code": str}}

IMPORTANT: Output ONLY JSON. No other text.
""",
    input_variables=["review"],
)
# Pawel Zmudzki Sprawdzanie pliku .json
def fix_apostrophes_in_json_string(json_string: str) -> str:
    """Popraw apostrofy na cudzysłowy w surowym stringu JSON, jeśli to konieczne."""
    import re

    # Zamienia apostrofy na cudzysłowy TYLKO wewnątrz listy [{...}]
    corrected_string = re.sub(r"\[{'destination'", '[{"destination"', json_string)
    corrected_string = corrected_string.replace("': '", '": "').replace("', '", '", "')
    corrected_string = corrected_string.replace("'}]", '"}]')
    
    return corrected_string


# Helper functions
def ask_ollama(prompt: str) -> str:
    try:
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"timeout": 75}
        )
        content = response['message']['content']
        print(f"\n===== PROMPT =====\n{prompt}\n===== ODPOWIEDŹ MODELU =====\n{repr(content)}\n===================\n")
        if not content.strip():
            raise ValueError("OLLAMA zwróciła pustą odpowiedź!")
        return content
    except Exception as e:
        print(f"Błąd podczas zapytania do Ollama: {e}")
        return "{}"

def safe_json_loads(response: str) -> Dict:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print(f"Błąd dekodowania JSON-a:\n{response}")
        raise

def analyze_sentiment(review_text):
    prompt_text = sentiment_prompt.format(review=review_text)
    response = ask_ollama(prompt_text)
    return safe_json_loads(response)

def handle_positive_review(review_text):
    prompt_text = positive_prompt.format(review=review_text)
    response = ask_ollama(prompt_text)
    return safe_json_loads(response)

def handle_negative_review(review_text):
    prompt_text = negative_prompt.format(review=review_text)
    response = ask_ollama(prompt_text)
    return safe_json_loads(response)

def extract_locations(text):
    doc = nlp(text)
    locations = []
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC", "FAC"]:
            locations.append({"text": ent.text, "label": ent.label_})
    return locations

def process_single_review(review_data: Dict, review_id: int, semaphore: Semaphore) -> Any | None:
    try:
        with semaphore:
            review = review_data["review"]
            if len(review) > 2000:
                review = review[:2000]

            sentiment_result = analyze_sentiment(review)

            if sentiment_result.get("positive_sentiment"):
                response = handle_positive_review(review)
            else:
                response = handle_negative_review(review)

            locations = extract_locations(review)

            result = {
                "review": review,
                "sentiment": "positive" if sentiment_result.get("positive_sentiment") else "negative",
                "reasoning": sentiment_result.get("reasoning", ""),
                "ground_truth": review_data.get("survey_sentiment", ""),
                "satisfaction_score": review_data.get("customer_satisfaction_score", 0),
                "review_id": review_id,
                "locations": locations,
            }

            result.update(response)
            return result
    except Exception as e:
        logging.error(f"Error processing review {review_id}: {str(e)}")
        return None

def calculate_metrics(ground_truth: List[bool], predictions: List[bool]) -> Dict:
    return {
        "accuracy": accuracy_score(ground_truth, predictions),
        "precision": precision_score(ground_truth, predictions),
        "recall": recall_score(ground_truth, predictions),
        "f1_score": f1_score(ground_truth, predictions),
    }

# Pawel Zmudzki Modyfikacja dla plik .json
def process_customer_reviews(input_file, output_file, locations_file):
    with open(input_file, "r", encoding="utf-8") as f:
        raw_json = f.read()
        fixed_json = fix_apostrophes_in_json_string(raw_json)
        data = json.loads(fixed_json)

    semaphore = Semaphore(1)
    results = []
    locations_data = []
    ground_truth = []
    predictions = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_review = {
            executor.submit(process_single_review, review_data, idx, semaphore): idx
            for idx, review_data in enumerate(data)
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_review),
            total=len(data),
            desc="Processing reviews",
        ):
            review_id = future_to_review[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)

                    ground_truth_sentiment = (
                        True if data[review_id]["survey_sentiment"] == "positive" else False
                    )
                    ground_truth.append(ground_truth_sentiment)
                    predictions.append(result["sentiment"] == "positive")

                    for loc in result["locations"]:
                        locations_data.append(
                            {
                                "review_id": review_id,
                                "location": loc["text"],
                                "entity_type": loc["label"],
                            }
                        )
            except Exception as e:
                logging.error(f"Error processing result for review {review_id}: {str(e)}")

    results.sort(key=lambda x: x["review_id"])
    metrics = calculate_metrics(ground_truth, predictions)
    save_results_to_csv(results, output_file)
    save_locations_to_csv(locations_data, locations_file)

    return results, metrics

def save_results_to_csv(results, output_file):
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if results:
            first_result = results[0]
            headers = [
                "review", "sentiment", "reasoning", "ground_truth", "satisfaction_score"
            ]
            if "message" in first_result:
                headers.append("message")
            if "recommended_trips" in first_result:
                headers.extend([
                    "recommended_trip_1", "recommended_trip_2", "recommended_trip_3"
                ])
            if "discount_code" in first_result:
                headers.append("discount_code")
            writer.writerow(headers)

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
                        row.append(f"{trips[i]['destination']}: {trips[i]['description']}")
                    for i in range(len(trips), 3):
                        row.append("")
                if "discount_code" in result:
                    row.append(result["discount_code"])
                writer.writerow(row)

def save_locations_to_csv(locations, output_file):
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["review_id", "location", "entity_type"])
        for location in locations:
            writer.writerow([
                location["review_id"], location["location"], location["entity_type"]
            ])

def visualize_metrics(metrics):
    print("\nSentiment Analysis Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    input_file = "../data/customer_surveys.json"
    output_file = "../data/generated/processed_reviews_o.csv"
    locations_file = "../data/generated/extracted_locations_o.csv"

    results, metrics = process_customer_reviews(input_file, output_file, locations_file)

    visualize_metrics(metrics)

    print(f"\nProcessed {len(results)} customer reviews")
    print(f"Results saved to {output_file}")
    print(f"Extracted locations saved to {locations_file}")