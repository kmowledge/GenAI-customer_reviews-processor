import streamlit as st
import json
import os
from groq_processor import (
    sentiment_chain, 
    extract_locations, 
    positive_chain, 
    negative_chain
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API key
if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY environment variable is not set. Please set it in the .env file.")
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
                st.markdown(f"**Reasoning:** {sentiment_result['reasoning']}")
                
                # Extract locations
                st.subheader("Location Extraction")
                locations = extract_locations(review)
                
                if locations:
                    location_data = []
                    for loc in locations:
                        #location_data.append(f"- {loc['text']} ({loc['label']})")
                        location_data.append(f"- {loc['location']} ({loc['entity_type']})")
                    st.markdown("\n".join(location_data))
                else:
                    st.markdown("No locations found in the review.")
            
            # Generate response
            with col2:
                st.subheader("Generated Response")
                if sentiment_result["positive_sentiment"]:
                    response = positive_chain.invoke({"review": review})
                    
                    # Display personalized message
                    st.markdown("**Personalized Response:**")
                    st.info(response["message"])
                    
                    # Display recommended trips
                    st.markdown("**Recommended Trips:**")
                    for trip in response["recommended_trips"]:
                        st.markdown(f"- **{trip['destination']}**: {trip['description']}")
                else:
                    response = negative_chain.invoke({"review": review})
                    
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
    2. Click 'Process Review'
    3. View the sentiment analysis, extracted locations, and generated response
    
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