


from joblib import load
from ed_admissions_utils import get_model_name, preprocess_data, load_saved_model
from ed_admissions_data_retrieval import ed_admissions_get_data

import pandas as pd

def prepare_for_inference(model_file_path, model_name, time_of_day = None, model_only = False, df = None, data_path = None, single_episode_slice_per_visit = True):
    
    # retrieve model trained for this time of day
    model = load_saved_model(model_file_path, model_name, time_of_day)  
    
    if (model_only):
        return(model)
    
    if data_path:
        df = ed_admissions_get_data(data_path)
    elif df is None or df.empty:
        print("Please supply a dataset if not passing a data path")
        return
    
    # print("Prep for inference - df")
    # print(df[(df.training_validation_test == 'test')].index)
    
    if df.index.name != 'episode_slice_id':
        df = df.set_index('episode_slice_id')

    test_df = df[df.training_validation_test == 'test'].drop(columns='training_validation_test').copy()

    # print("Prep for inference - test_df")
    # print(test_df.index)
    
    exclude_from_training_data = [
        "visit_number",
        "horizon_dt",
        "time_of_day"]

    X_test, y_test = preprocess_data(test_df, time_of_day, exclude_from_training_data, single_episode_slice_per_visit)
    
    # print("Prep for inference - X_test")
    # print(X_test.index)
    # print("Prep for inference - y_test")
    # print(y_test.index)
    
    
    return X_test, y_test, model
        
    


def get_specialty_probs(model_file_path, episode_slices_df):

    # Load model for specialty predictions
    specialty_model =  prepare_for_inference(model_file_path, 'ed_specialty', model_only = True)

    # Mark which observations are for children
    episode_slices_df['is_child'] = episode_slices_df['age_group'] == '0-17'

    # For children we assume all admitted to pediatric specialties and will not go to any other place
    child_dict = {
        'medical': 0.0,
        'surgical': 0.0,
        'haem_onc': 0.0,
        'paediatric': 1.0
    }


    # Apply child_dict directly to children and speciality model to all other visits 
    episode_slices_df['specialty_prob'] = episode_slices_df.apply(
        lambda row: specialty_model.predict(row['consultation_sequence']) if not row['is_child'] else child_dict,
        axis=1
    )

    # Ensure each dictionary in 'specialty_prob' contains the key 'paediatric' with a default value of 0
    # This is necessary because, in our local implementation, specialty_model has not been trained to return predictions for paediatric patients
    episode_slices_df['specialty_prob'] = episode_slices_df['specialty_prob'].apply(lambda d: {**d, **{'paediatric': d.get('paediatric', 0)}})
    
    return(episode_slices_df)
