import concurrent
import csv
import json
import logging
import os
import time
import re
import spacy
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableBranch, RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"]
)

class GroqLLMRunnable:
    def __init__(self, model="gemma2-9b-it", temperature=0):
        self.model = model
        self.temperature = temperature

    def invoke(self, input_str: str, config=None) -> str:
        if not isinstance(input_str, str) or not input_str.strip():
            raise ValueError("Prompt musi być niepustym ciągiem znaków")

        retries = 5
        for attempt in range(retries):
            try:
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": input_str}],
                    temperature=self.temperature,
                    timeout=30
                )
                result = completion.choices[0].message.content.strip()

                if not result:
                    logging.error(f"Pusta odpowiedź od Groq dla promptu: {input_str}")
                    return '{"positive_sentiment": null, "reasoning": "No response"}'

                return result

            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    wait_time = 5
                    logging.warning(f"Limit przekroczony, próba {attempt + 1}/{retries}. Czekam {wait_time} sekund...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Błąd przy wywołaniu modelu: {str(e)}")
                    raise e

        raise Exception("Próby wysłania promptu zakończone niepowodzeniem (rate limit)")




llm = GroqLLMRunnable()
nlp = spacy.load("en_core_web_sm")
sentiment_parser = JsonOutputParser()

sentiment_prompt = PromptTemplate(
    template="""
You are a sentiment analysis expert.
You MUST respond with VALID JSON ONLY, and nothing else.
DO NOT add any explanation, preamble, or text outside of the JSON.

Customer Review:
{review}

If the sentiment is positive, return exactly:
{{"positive_sentiment": true, "reasoning": "Customer liked XYZ"}}

If the sentiment is negative, return exactly:
{{"positive_sentiment": false, "reasoning": "Customer disliked XYZ"}}
""",
    input_variables=["review"],
)
sentiment_chain = sentiment_prompt | (lambda x: llm.invoke(x.text)) | sentiment_parser

positive_prompt = PromptTemplate(
    template="""
You are a customer service representative for a travel company.
A customer left a positive review.

Customer Review:
{review}

Respond clearly in JSON format:
{{
    "message": "personalized thanks and voucher offer",
    "recommended_trips": [
        {{"destination": "Name", "description": "Short description"}},
        {{"destination": "Name", "description": "Short description"}},
        {{"destination": "Name", "description": "Short description"}}
    ]
}}
""",
    input_variables=["review"],
)
positive_chain = positive_prompt | (lambda x: llm.invoke(x.text)) | JsonOutputParser()

negative_prompt = PromptTemplate(
    template="""
You are a customer service representative for a travel company.
You MUST respond with VALID JSON ONLY.
DO NOT add any explanation or disclaimers.

Review:
{review}

Return EXACTLY this JSON:
{{
    "message": "personalized apology and offer",
    "discount_code": "location + 4-digit number"
}}
""",
    input_variables=["review"],
)
negative_chain = negative_prompt | (lambda x: llm.invoke(x.text)) | JsonOutputParser()

full_chain = RunnablePassthrough.assign(sentiment_result=sentiment_chain) | RunnableBranch(
    (lambda x: x["sentiment_result"]["positive_sentiment"], positive_chain),
    (lambda x: not x["sentiment_result"]["positive_sentiment"], negative_chain),
    lambda x: {"message": "Unable to determine sentiment"},
)


def clean_location_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.strip()
    # Pozwalamy tylko na litery, spacje i myślniki
    name = re.sub(r"[^a-zA-Z\s\-]", "", name)
    return name

def extract_locations(text):
    locations = []
    if not text:
        return locations
    
    doc = nlp(text)
    for ent in doc.ents:
        #if ent.label_ == "GPE":  # Zmiana test 
        if ent.label_ in ["GPE", "LOC", "FAC"]:
            loc_text = clean_location_name(ent.text)
            # Filtrowanie sensownych nazw
            if 3 <= len(loc_text) <= 40:
                locations.append({
                    "location": loc_text,
                    "entity_type": ent.label_
                })

    return locations

def process_single_review(review_data: Dict, review_id: int, semaphore: Semaphore) -> Any | None:
    try:
        with semaphore:
            review = review_data.get("review", "").strip()
            if not review:
                logging.warning(f"Pusta recenzja ID: {review_id}, zawartość: {review_data}")
                return None

            # 1. Najpierw wywołaj sentiment_chain:
            try:
                sentiment_result = sentiment_chain.invoke({"review": review})
            except Exception as e:
                logging.error(f"Błąd analizy sentymentu {review_id}: {e}")
                return None

            # 2. Następnie pełny chain (który wybierze positive/negative gałąź):
            try:
                branch_result = full_chain.invoke({"review": review})
            except Exception as e:
                logging.error(f"Błąd przetwarzania recenzji {review_id}: {e}")
                return None

            # 3. Wyciągasz reasoning z sentiment_result
            reasoning_text = sentiment_result.get("reasoning", "")

            # 4. Ekstrakcja lokalizacji
            locations = extract_locations(review)

            # 5. Składasz końcowe result – łączysz branch_result (message, discount_code,
            # recommended_trips, itp.) z reasoning oraz innymi polami:
            final_result = {
                "review": review,
                # Gałąź positive/negative rozpoznaje to po recommended_trips:
                "sentiment": "positive" if branch_result.get("recommended_trips") else "negative",
                "reasoning": reasoning_text,
                "ground_truth": review_data["survey_sentiment"],
                "satisfaction_score": review_data["customer_satisfaction_score"],
                "review_id": review_id,
                "locations": locations,
            }

            # 6. Przepisz pola z branch_result (np. "message", "discount_code", "recommended_trips"):
            for key, val in branch_result.items():
                final_result[key] = val

            return final_result

    except Exception as e:
        logging.error(f"Critical error processing review {review_id}: {e}")
        return None




def calculate_metrics(ground_truth: List[bool], predictions: List[bool]) -> Dict:
    if not ground_truth or not predictions:
        logging.warning("No predictions or ground truth available to calculate metrics.")
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}

    return {
        "accuracy": accuracy_score(ground_truth, predictions),
        "precision": precision_score(ground_truth, predictions, zero_division=0),
        "recall": recall_score(ground_truth, predictions, zero_division=0),
        "f1_score": f1_score(ground_truth, predictions, zero_division=0),
    }


def process_customer_reviews(input_file, output_file, locations_file):
    with open(input_file) as f:
        data = json.load(f)

    valid_data = [r for r in data if r.get('review', '').strip()]
    if len(valid_data) < len(data):
        logging.warning(f"Skipped {len(data) - len(valid_data)} reviews due to empty text.")
    data = valid_data

    semaphore = Semaphore(10)
    results, locations_data, gt, pred = [], [], [], []

    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_review = {
            executor.submit(process_single_review, d, idx, semaphore): idx
            for idx, d in enumerate(data)
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_review), total=len(data), desc="Processing reviews"):
            review_id = future_to_review[future]
            res = future.result()
            if res:
                results.append(res)
                gt.append(data[review_id]["survey_sentiment"] == "positive")
                pred.append(res["sentiment"] == "positive")
                locations_data += [{"review_id": review_id, **loc} for loc in res["locations"]]

    results.sort(key=lambda x: x["review_id"])
    metrics = calculate_metrics(gt, pred)

    save_results_to_csv(results, output_file)
    save_locations_to_csv(locations_data, locations_file)

    return results, metrics


def save_results_to_csv(results, output_file):
    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        # Nagłówek
        if results:
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
                headers.extend(["recommended_trip_1", "recommended_trip_2", "recommended_trip_3"])

            if "discount_code" in first_result:
                headers.append("discount_code")

            writer.writerow(headers)

            # Dane
            for result in results:
                row = [
                    result.get("review", ""),
                    result.get("sentiment", ""),
                    result.get("reasoning", ""),
                    result.get("ground_truth", ""),
                    result.get("satisfaction_score", ""),
                ]

                if "message" in result:
                    row.append(result.get("message", ""))

                if "recommended_trips" in result:
                    trips = result.get("recommended_trips", [])
                    for i in range(3):
                        if i < len(trips):
                            trip = trips[i]
                            row.append(f"{trip.get('destination', '')}: {trip.get('description', '')}")
                        else:
                            row.append("")

                if "discount_code" in result:
                    row.append(result.get("discount_code", ""))

                writer.writerow(row)


def save_locations_to_csv(locations, output_file):
    #with open(output_file, "w", newline="") as f:
    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["review_id", "location", "entity_type"])

        for location in locations:
            writer.writerow([location["review_id"], location["location"], location["entity_type"]])


def visualize_metrics(metrics):
    """
    Może posłużyć do generowania dodatkowych wizualizacji, 
    w tym przykładzie jedynie wypisujemy metryki.
    """
    print("\nSentiment Analysis Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    # Ścieżki plików
    input_file = "../data/customer_surveys.json"
    output_file = "../data/generated/processed_reviews_g.csv"
    locations_file = "../data/generated/extracted_locations_g.csv"

    # Analiza recenzji
    results, metrics = process_customer_reviews(input_file, output_file, locations_file)

    # Wyświetlamy metryki
    visualize_metrics(metrics)

    print(f"\nProcessed {len(results)} customer reviews")
    print(f"Results saved to {output_file}")
    print(f"Extracted locations saved to {locations_file}")

