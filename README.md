# GenAI-customer_reviews-processor
Customer Satisfaction Processing Bot

## Overview
This tool processes customer satisfaction survey data to:
1. Classify sentiment of customer reviews (positive/negative)
2. Send customized discounts for negative reviews
3. Recommend 3 best trips for positive reviews
4. Extract mentioned locations using Named Entity Recognition (NER)
5. Compare classification metrics against ground truth data
6. Save all output to CSV files

## Features

### Base Features
- Sentiment analysis of customer reviews
- Personalized responses for both positive and negative reviews
- Trip recommendations for positive reviews
- Discount codes for negative reviews
- CSV output of processed reviews

### Extra Features
- Named Entity Recognition (NER) to extract mentioned locations
- Sentiment analysis metrics calculation (accuracy, precision, recall, F1 score)
- Comparison against ground truth sentiment data

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Download spaCy language model:
```
python -m spacy download en_core_web_sm
```

3. Set your OpenAI API key:
```
export OPENAI_API_KEY=your_api_key_here
```

## Usage

### Process Reviews

Run the processor:
```
cd src
python processor.py
```

This will:
1. Process all reviews in `data/customer_surveys.json`
2. Save processed reviews to `data/processed_reviews.csv`
3. Save extracted locations to `data/extracted_locations.csv`
4. Display sentiment analysis metrics

### Generate Visualizations

After processing the reviews, generate visualizations:
```
cd src
python visualize.py
```

This will create:
1. `confusion_matrix.png`: Confusion matrix for sentiment analysis
2. `metrics_chart.png`: Bar chart of accuracy, precision, recall, and F1 score
3. `satisfaction_by_sentiment.png`: Average satisfaction scores by sentiment
4. `top_locations.png`: Top 10 mentioned locations
5. `entity_types.png`: Distribution of entity types

## Output Files

1. `processed_reviews.csv`: Contains all processed reviews with:
   - Original review text
   - Detected sentiment
   - Reasoning for sentiment classification
   - Ground truth sentiment
   - Customer satisfaction score
   - Personalized response message
   - Recommended trips (for positive reviews)
   - Discount code (for negative reviews)

2. `extracted_locations.csv`: Contains all extracted locations with:
   - Review ID
   - Location name
   - Entity type

## Implementation Details

The system uses:
- LangChain for orchestrating the LLM workflow
- OpenAI's GPT-4o for sentiment analysis and response generation
- spaCy for Named Entity Recognition
- scikit-learn for calculating sentiment analysis metrics
- Matplotlib for visualization of metrics and results

### Architecture

1. **Data Processing Pipeline**:
   - Load customer review data from JSON
   - Perform sentiment analysis with LLM
   - Generate personalized responses based on sentiment
   - Extract locations using NER
   - Save processed results to CSV files

2. **Sentiment Analysis Workflow**:
   - Uses a branching chain to route reviews to appropriate handling
   - Positive reviews: Generate thank-you message and trip recommendations
   - Negative reviews: Generate apology, improvement plan, and discount code

3. **NER Implementation**:
   - Uses spaCy's named entity recognition to extract locations
   - Identifies Geopolitical Entities (GPE), Locations (LOC), and Facilities (FAC)
   - Tracks entity types for further analysis

4. **Metrics Evaluation**:
   - Compares predicted sentiment against ground truth
   - Calculates accuracy, precision, recall, and F1 score
   - Generates visualizations to understand model performance
