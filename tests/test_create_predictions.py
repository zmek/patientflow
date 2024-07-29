import unittest
from datetime import datetime
import pandas as pd
import numpy as np
import sys
import os

from pathlib import Path
import sys
import joblib

# PROJECT_ROOT = Path().home() 
# USER_ROOT = Path().home() / 'work'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../functions')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/patientflow')))

# sys.path.append(str(USER_ROOT / 'patientflow' / 'src' / 'patientflow'))
# sys.path.append(str(USER_ROOT / 'patientflow' / 'functions'))

from ed_admissions_realtime_preds import create_predictions
from ed_admissions_machine_learning import create_column_transformer
from ed_admissions_utils import get_model_name

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import os


# Example usage:
# Assuming you have a dataframe `df` with the necessary columns
# df = pd.read_csv('your_data.csv')
# pipeline = create_pipeline(df)

class ProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities
    
    def predict(self):
        return self.probabilities
        
class PoissonModel:
    def __init__(self, lambdas):
        self.lambdas = lambdas
    
    def predict(self):
        result = {}
        for spec, lam in self.lambdas.items():
            # Generate Poisson distribution
            poisson_dist = np.random.poisson(lam, 1000)
            
            # Create DataFrame
            df = pd.DataFrame(poisson_dist, columns=['agg_prob'])
            df['sum'] = df.index
            
            # Set 'sum' as the index
            df.set_index('sum', inplace=True)
            
            result[spec] = df
        
        return result

class TestCreatePredictions(unittest.TestCase):
    
    def setUp(self):
        self.model_file_path = Path('tmp')
        os.makedirs(self.model_file_path, exist_ok=True)
        print(self.model_file_path)
        self.prediction_time = (7,0)
        self.prediction_window_hrs = 8.0
        self.x1, self.y1, self.x2, self.y2 = 4.0, 0.76, 12.0, 0.99
        self.cdf_cut_points = [0.7, 0.9]
        self.specialties = ['paediatric', 'medical']
        self.create_admissions_model(self.model_file_path)
        self.create_yta_model(self.model_file_path)
        self.create_spec_model(self.model_file_path)

    def create_admissions_model(self, model_file_path):
        # Define the feature columns and target
        feature_columns = ['elapsed_los', 'sex', 'age_on_arrival', 'arrival_method']
        target_column = 'is_admitted'

        df = pd.DataFrame([
            {'age_on_arrival': 15, 'elapsed_los': 3600, 'arrival_method': 'ambulance', 'sex':'M', 'is_admitted' :1},
            {'age_on_arrival': 45, 'elapsed_los': 7200, 'arrival_method': 'walk-in', 'sex':'F', 'is_admitted' : 0},
        ])

        # Split the data into features and target
        X = df[feature_columns]
        y = df[target_column]

        # Define the model
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        column_transformer = create_column_transformer(X)

        # create a pipeline with the feature transformer and the model
        pipeline = Pipeline(
            [("feature_transformer", column_transformer), ("classifier", model)]
        )

        # Fit the pipeline to the data
        pipeline.fit(X, y)

        model_name = get_model_name('ed_admission', self.prediction_time)
        full_path = self.model_file_path /  str(model_name + '.joblib')        
        joblib.dump(pipeline, full_path)

    def create_spec_model(self, model_file_path):
        probabilities = {
            'surgical': 0.3,
            'haem/onc': 0.1,
            'medical': 0.6
        }

        model = ProbabilityModel(probabilities)
        
        full_path = self.model_file_path /  str('ed_specialty.joblib')        
        joblib.dump(model, full_path)
                    
    def create_yta_model(self, model_file_path):
        lambdas = {
            'medical': 5,
            'paediatric': 3
        }
        model = PoissonModel(lambdas)
        
        full_path = self.model_file_path /  str('ed_yet_to_arrive_by_spec_8_hours.joblib')        
        joblib.dump(model, full_path)
        
    def test_basic_functionality(self):
        prediction_snapshots = pd.DataFrame([
            {'age_on_arrival': 15, 'elapsed_los': 3600, 'arrival_method': 'ambulance', 'sex':'M', 'is_admitted' :1, 'consultation_sequence' : []},
            {'age_on_arrival': 45, 'elapsed_los': 7200, 'arrival_method': 'walk-in', 'sex':'F', 'is_admitted' : 0, 'consultation_sequence' : []},
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

        print(self.model_file_path)

        predictions = create_predictions(
            model_file_path=self.model_file_path,
            prediction_time=self.prediction_time,
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
       
if __name__ == '__main__':
    unittest.main()