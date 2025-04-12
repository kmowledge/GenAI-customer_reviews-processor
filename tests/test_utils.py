import os
import unittest
from src.utils import save_results_to_csv, save_locations_to_csv

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.test_results_file = "test_results.csv"
        self.test_locations_file = "test_locations.csv"

    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.test_results_file):
            os.remove(self.test_results_file)
        if os.path.exists(self.test_locations_file):
            os.remove(self.test_locations_file)

    def test_save_results_to_csv_basic(self):
        test_results = [{
            "review": "Great trip!",
            "sentiment": "positive",
            "reasoning": "Enthusiastic review",
            "ground_truth": "positive",
            "satisfaction_score": 5
        }]
        
        save_results_to_csv(test_results, self.test_results_file)
        self.assertTrue(os.path.exists(self.test_results_file))
        
        with open(self.test_results_file, 'r') as f:
            content = f.read()
            self.assertIn("review,sentiment,reasoning,ground_truth,satisfaction_score", content)
            self.assertIn("Great trip!,positive,Enthusiastic review,positive,5", content)

    def test_save_results_to_csv_with_recommendations(self):
        test_results = [{
            "review": "Amazing vacation!",
            "sentiment": "positive",
            "reasoning": "Very positive",
            "ground_truth": "positive",
            "satisfaction_score": 5,
            "recommended_trips": [
                {"destination": "Paris", "description": "City of Light"},
                {"destination": "Rome", "description": "Eternal City"}
            ]
        }]
        
        save_results_to_csv(test_results, self.test_results_file)
        self.assertTrue(os.path.exists(self.test_results_file))

    def test_save_locations_to_csv(self):
        test_locations = [
            {"review_id": 1, "location": "Paris", "entity_type": "GPE"},
            {"review_id": 2, "location": "Eiffel Tower", "entity_type": "FAC"}
        ]
        
        save_locations_to_csv(test_locations, self.test_locations_file)
        self.assertTrue(os.path.exists(self.test_locations_file))
        
        with open(self.test_locations_file, 'r') as f:
            content = f.read()
            self.assertIn("review_id,location,entity_type", content)
            self.assertIn("1,Paris,GPE", content)
            self.assertIn("2,Eiffel Tower,FAC", content)

if __name__ == '__main__':
    unittest.main()