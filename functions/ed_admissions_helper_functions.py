


from joblib import load
from ed_admissions_utils import get_model_name, preprocess_data, load_saved_model
from ed_admissions_data_retrieval import ed_admissions_get_data


def prepare_data_for_inference(PATH_ED, model_file_path, time_of_day, model_only = False, single_episode_slice_per_visit = True):
    
    # retrieve model trained for this time of day
    model = load_saved_model(time_of_day, model_file_path)  
    
    if (model_only):
        return(model)
    
    df = ed_admissions_get_data(PATH_ED)
    test_df = df[df.training_validation_test == 'test'].drop(columns='training_validation_test')
    
    exclude_from_training_data = [
    "visit_number",
    "horizon_dt",
    "time_of_day"]
    
  
    X_test, y_test = preprocess_data(test_df, time_of_day, exclude_from_training_data, single_episode_slice_per_visit)
    
    return X_test, y_test, model
        