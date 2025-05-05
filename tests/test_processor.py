import unittest
import os
from unittest.mock import patch, MagicMock
from src.processor import extract_locations, calculate_metrics, process_single_review

class TestProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['TESTING'] = 'true'
        os.environ['OPENAI_API_KEY'] = 'dummy_key_for_testing'

    @classmethod
    def tearDownClass(cls):
        del os.environ['TESTING']
        del os.environ['OPENAI_API_KEY']

    def test_extract_locations(self):
        text = "I visited Paris and the Eiffel Tower last summer."
        locations = extract_locations(text)
        
        self.assertTrue(any(loc["text"] == "Paris" for loc in locations))
        self.assertFalse(any(loc["text"] == "Eiffel Tower" for loc in locations))

    def test_calculate_metrics(self):
        ground_truth = [True, True, False, True]
        predictions = [True, True, False, False]
        
        metrics = calculate_metrics(ground_truth, predictions)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
        self.assertEqual(metrics['accuracy'], 0.75)

    @patch('src.processor.sentiment_chain')
    @patch('src.processor.full_chain')
    def test_process_single_review(self, mock_full_chain, mock_sentiment_chain):
        # Mock the chain responses
        mock_sentiment_chain.invoke.return_value = {
            "positive_sentiment": True,
            "reasoning": "Test reasoning"
        }
        
        mock_full_chain.invoke.return_value = {
            "message": "Thank you for your review",
            "recommended_trips": [
                {"destination": "Paris", "description": "Test description"}
            ]
        }
        
        review_data = {
            "review": "Great experience!",
            "survey_sentiment": "positive",
            "customer_satisfaction_score": 5
        }
        
        result = process_single_review(review_data, 1, MagicMock())
        
        self.assertIsNotNone(result)
        self.assertEqual(result["sentiment"], "positive")
        self.assertEqual(result["ground_truth"], "positive")
        self.assertEqual(result["satisfaction_score"], 5)
        self.assertIn("locations", result)

    def test_process_single_review_error_handling(self):
        review_data = {
            "review": "Test review",
            "survey_sentiment": "positive",
            "customer_satisfaction_score": 5
        }
        
        # Test with None semaphore to trigger exception
        result = process_single_review(review_data, 1, None)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()