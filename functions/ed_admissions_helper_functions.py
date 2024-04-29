


from joblib import load
from ed_admissions_utils import get_model_name, preprocess_data, load_saved_model
from ed_admissions_data_retrieval import ed_admissions_get_data


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
        