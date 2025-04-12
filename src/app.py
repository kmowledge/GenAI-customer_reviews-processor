import streamlit as st
import json
import os
from processor import (
    sentiment_chain, 
    extract_locations, 
    positive_chain, 
    negative_chain
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY environment variable is not set. Please set it in the .env file.")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Customer Review Processor",
    page_icon="✈️",
    layout="wide"
)

# Add title and description
st.title("Customer Review Processor")
st.markdown("""
This application processes customer reviews for a travel company using:
- Sentiment Analysis (positive/negative)
- Named Entity Recognition (locations)
- Personalized response generation
""")

# Review input
review = st.text_area(
    "Enter a customer review:",
    height=150,
    placeholder="Write a travel review here..."
)

# Add a slider for user to manually input satisfaction score
user_score = st.slider(
    "Customer Satisfaction Score (1-5):",
    min_value=1,
    max_value=5,
    value=3,
    help="Manually enter a customer satisfaction score from 1 (very dissatisfied) to 5 (very satisfied)"
)

# Process button
if st.button("Process Review"):
    if not review:
        st.warning("Please enter a review to process")
    else:
        with st.spinner("Processing review..."):
            # Create two columns for output
            col1, col2 = st.columns(2)
            
            # Analyze sentiment
            with col1:
                st.subheader("Sentiment Analysis")
                sentiment_result = sentiment_chain.invoke({"review": review})
                
                # Display sentiment result
                sentiment = "Positive" if sentiment_result["positive_sentiment"] else "Negative"
                st.markdown(f"**Sentiment:** {sentiment}")
                
                # Display customer satisfaction score - use user input score
                score = user_score
                
                # Display score as stars and numeric value
                stars = "⭐" * score
                st.markdown(f"**Customer Satisfaction Score:** {score}/5 {stars}")
                
                # Display AI-predicted score for comparison
                ai_score = sentiment_result.get("satisfaction_score", 0)
                ai_stars = "⭐" * ai_score
                st.markdown(f"**AI-Predicted Score:** {ai_score}/5 {ai_stars}")
                
                st.markdown(f"**Reasoning:** {sentiment_result['reasoning']}")
                
                # Extract locations
                st.subheader("Location Extraction")
                locations = extract_locations(review)
                
                if locations:
                    location_data = []
                    for loc in locations:
                        location_data.append(f"- {loc['text']} ({loc['label']})")
                    st.markdown("\n".join(location_data))
                else:
                    st.markdown("No locations found in the review.")
            
            # Generate response
            with col2:
                st.subheader("Generated Response")
                if sentiment_result["positive_sentiment"]:
                    response = positive_chain.invoke({"review": review, "satisfaction_score": score})
                    
                    # Display personalized message
                    st.markdown("**Personalized Response:**")
                    st.info(response["message"])
                    
                    # Display recommended trips
                    st.markdown("**Recommended Trips:**")
                    for trip in response["recommended_trips"]:
                        st.markdown(f"- **{trip['destination']}**: {trip['description']}")
                else:
                    response = negative_chain.invoke({"review": review, "satisfaction_score": score})
                    
                    # Display personalized message
                    st.markdown("**Personalized Response:**")
                    st.info(response["message"])
                    
                    # Display discount code
                    st.markdown(f"**Discount Code:** {response['discount_code']}")
            
            # Display raw JSON
            with st.expander("View Raw JSON Response"):
                if sentiment_result["positive_sentiment"]:
                    st.json(response)
                else:
                    st.json(response)

# Add instructions in the sidebar
with st.sidebar:
    st.subheader("How to Use")
    st.markdown("""
    1. Enter a customer review in the text area
    2. Adjust the satisfaction score slider (1-5)
    3. Click 'Process Review'
    4. View the sentiment analysis, extracted locations, and generated response
    
    **Example Reviews:**
    
    Positive:
    ```
    Our trip to Paris was absolutely magical! The hotel was comfortable, the food was delicious, and the Eiffel Tower view was breathtaking. We'll definitely return next year!
    ```
    
    Negative:
    ```
    Our vacation to Rome was disappointing. The hotel was dirty, the staff was rude, and our tour was canceled without notice. I wouldn't recommend this experience to anyone.
    ```
    """)
    
    st.markdown("---")
    st.markdown("Created with ❤️ using Streamlit and LangChain") 
