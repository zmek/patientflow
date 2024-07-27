import unittest
from datetime import datetime
import pandas as pd


class TestCreatePredictions(unittest.TestCase):
    
    def setUp(self):
        self.model_file_path = 'dummy/path/to/model/file'
        self.prediction_moment = datetime(2023, 1, 1, 12, 0)
        self.prediction_window_hrs = 8.0
        self.x1, self.y1, self.x2, self.y2 = 4.0, 0.76, 12.0, 0.99
        self.cdf_cut_points = [0.7, 0.9]
        self.specialties = ['paediatric', 'medical']

    def test_basic_functionality(self):
        prediction_snapshots = pd.DataFrame([
            {'age_on_arrival': 15, 'elapsed_los': 3600, 'consultation_sequence':[],
            {'age_on_arrival': 45, 'elapsed_los': 7200, 'consultation_sequence':[]}
        ])

        special_category_dict = {
            'medical': 0.0,
            'paediatric': 1.0
        }
        
        # Function to determine if the patient is a child
        special_category_func = lambda row: row['age_on_arrival'] < 18 
        
        special_func_map = {
            'paediatric': special_category_func,
            'default': lambda row: True  
        }

        predictions = create_predictions(
            model_file_path=self.model_file_path,
            prediction_moment=self.prediction_moment,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window_hrs=self.prediction_window_hrs,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            special_func_map=special_func_map,
        )

        self.assertIsInstance(predictions, dict)
        self.assertIn('paediatric', predictions)
        self.assertIn('medical', predictions)
        self.assertIn('in_ed', predictions['paediatric'])
        self.assertIn('yet_to_arrive', predictions['paediatric'])

    # def test_empty_prediction_snapshots(self):
    #     prediction_snapshots = pd.DataFrame()

    #     predictions = create_predictions(
    #         model_file_path=self.model_file_path,
    #         prediction_moment=self.prediction_moment,
    #         prediction_snapshots=prediction_snapshots,
    #         specialties=self.specialties,
    #         prediction_window_hrs=self.prediction_window_hrs,
    #         cdf_cut_points=self.cdf_cut_points,
    #         x1=self.x1,
    #         y1=self.y1,
    #         x2=self.x2,
    #         y2=self.y2,
    #     )

    #     self.assertIsInstance(predictions, dict)
    #     for specialty in self.specialties:
    #         self.assertEqual(predictions[specialty]['in_ed'], [])
    #         self.assertEqual(predictions[specialty]['yet_to_arrive'], [])

    # def test_single_row_prediction_snapshots(self):
    #     prediction_snapshots = pd.DataFrame([
    #         {'age_on_arrival': 15, 'elapsed_los': 3600}
    #     ])

    #     special_func_map = {
    #         'paediatric': lambda row: row['age_on_arrival'] < 18,
    #         'default': lambda row: True
    #     }

    #     predictions = create_predictions(
    #         model_file_path=self.model_file_path,
    #         prediction_moment=self.prediction_moment,
    #         prediction_snapshots=prediction_snapshots,
    #         specialties=self.specialties,
    #         prediction_window_hrs=self.prediction_window_hrs,
    #         cdf_cut_points=self.cdf_cut_points,
    #         x1=self.x1,
    #         y1=self.y1,
    #         x2=self.x2,
    #         y2=self.y2,
    #         special_func_map=special_func_map,
    #     )

    #     self.assertIsInstance(predictions, dict)
    #     self.assertIn('paediatric', predictions)
    #     self.assertIn('medical', predictions)
    #     self.assertEqual(len(predictions['paediatric']['in_ed']), len(self.cdf_cut_points))
    #     self.assertEqual(len(predictions['paediatric']['yet_to_arrive']), len(self.cdf_cut_points))

    # def test_without_optional_parameters(self):
    #     prediction_snapshots = pd.DataFrame([
    #         {'age_on_arrival': 15, 'elapsed_los': 3600},
    #         {'age_on_arrival': 45, 'elapsed_los': 7200}
    #     ])

    #     predictions = create_predictions(
    #         model_file_path=self.model_file_path,
    #         prediction_moment=self.prediction_moment,
    #         prediction_snapshots=prediction_snapshots,
    #         specialties=self.specialties,
    #         prediction_window_hrs=self.prediction_window_hrs,
    #         cdf_cut_points=self.cdf_cut_points,
    #         x1=self.x1,
    #         y1=self.y1,
    #         x2=self.x2,
    #         y2=self.y2,
    #     )

    #     self.assertIsInstance(predictions, dict)
    #     self.assertIn('paediatric', predictions)
    #     self.assertIn('medical', predictions)

    # def test_special_category_handling(self):
    #     prediction_snapshots = pd.DataFrame([
    #         {'age_on_arrival': 15, 'elapsed_los': 3600},
    #         {'age_on_arrival': 45, 'elapsed_los': 7200}
    #     ])

    #     special_category_func = lambda row: row['age_on_arrival'] < 18
    #     special_category_dict = {'paediatric': 1.0}

    #     special_func_map = {
    #         'paediatric': special_category_func,
    #         'default': lambda row: True
    #     }

    #     predictions = create_predictions(
    #         model_file_path=self.model_file_path,
    #         prediction_moment=self.prediction_moment,
    #         prediction_snapshots=prediction_snapshots,
    #         specialties=self.specialties,
    #         prediction_window_hrs=self.prediction_window_hrs,
    #         cdf_cut_points=self.cdf_cut_points,
    #         x1=self.x1,
    #         y1=self.y1,
    #         x2=self.x2,
    #         y2=self.y2,
    #         special_category_func=special_category_func,
    #         special_category_dict=special_category_dict,
    #         special_func_map=special_func_map,
    #     )

    #     self.assertIsInstance(predictions, dict)
    #     self.assertIn('paediatric', predictions)
    #     self.assertIn('medical', predictions)
    #     self.assertEqual(len(predictions['paediatric']['in_ed']), len(self.cdf_cut_points))
    #     self.assertEqual(len(predictions['paediatric']['yet_to_arrive']), len(self.cdf_cut_points))

    # def test_prediction_window_extremes(self):
    #     prediction_snapshots = pd.DataFrame([
    #         {'age_on_arrival': 15, 'elapsed_los': 3600},
    #         {'age_on_arrival': 45, 'elapsed_los': 7200}
    #     ])

    #     short_window_hrs = 0.1
    #     long_window_hrs = 100.0

    #     short_window_predictions = create_predictions(
    #         model_file_path=self.model_file_path,
    #         prediction_moment=self.prediction_moment,
    #         prediction_snapshots=prediction_snapshots,
    #         specialties=self.specialties,
    #         prediction_window_hrs=short_window_hrs,
    #         cdf_cut_points=self.cdf_cut_points,
    #         x1=self.x1,
    #         y1=self.y1,
    #         x2=self.x2,
    #         y2=self.y2,
    #     )

    #     long_window_predictions = create_predictions(
    #         model_file_path=self.model_file_path,
    #         prediction_moment=self.prediction_moment,
    #         prediction_snapshots=prediction_snapshots,
    #         specialties=self.specialties,
    #         prediction_window_hrs=long_window_hrs,
    #         cdf_cut_points=self.cdf_cut_points,
    #         x1=self.x1,
    #         y1=self.y1,
    #         x2=self.x2,
    #         y2=self.y2,
    #     )

    #     self.assertIsInstance(short_window_predictions, dict)
    #     self.assertIsInstance(long_window_predictions, dict)

    # def test_large_dataset_performance(self):
    #     prediction_snapshots = pd.DataFrame([{'age_on_arrival': i % 100, 'elapsed_los': i * 3600} for i in range(10000)])

    #     predictions = create_predictions(
    #         model_file_path=self.model_file_path,
    #         prediction_moment=self.prediction_moment,
    #         prediction_snapshots=prediction_snapshots,
    #         specialties=self.specialties,
    #         prediction_window_hrs=self.prediction_window_hrs,
    #         cdf_cut_points=self.cdf_cut_points,
    #         x1=self.x1,
    #         y1=self.y1,
    #         x2=self.x2,
    #         y2=self.y2,
    #     )

    #     self.assertIsInstance(predictions, dict)
       
