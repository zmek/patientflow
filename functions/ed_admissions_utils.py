from joblib import load


def get_model_name(model_name, tod_):
    """
    Create a model name based on the time of day.

    Parameters:
    tod_ (tuple): A tuple representing the time of day (hour, minute).

    Returns:
    str: A string representing the model name based on the time of day.
    """
    hour_, min_ = tod_
    min_ = f"{min_}0" if min_ % 60 == 0 else str(min_)
    model_name = model_name + '_' + f"{hour_:02}" + min_
    return model_name



def preprocess_data(df, tod_, exclude_columns, single_episode_slice_per_visit=True):
    # Filter by the time of day while keeping the original index
    df_tod = df[df['time_of_day'] == tod_].copy()
    
    if single_episode_slice_per_visit:
        # Group by 'visit_number' and get the row with the maximum 'random_number'
        max_indices = df_tod.groupby('visit_number')['random_number'].idxmax()
        df_single = df_tod.loc[max_indices].drop(columns=['random_number'])
        
        # Create label array with the same index
        y = df_single.pop("is_admitted").astype(int)
        
        # Drop specified columns and ensure we do not reset the index
        df_single.drop(columns=exclude_columns, inplace=True)
        
        return df_single, y
    
    else:
        # Directly modify df_tod without resetting the index
        df_tod.drop(columns=['random_number'] + exclude_columns, inplace=True)
        y = df_tod.pop("is_admitted").astype(int)

        return df_tod, y
    
    # include one one episode slice per visit and drop the random number
    
    


def load_saved_model(model_file_path, model_name, time_of_day = None):
    
    if time_of_day:
    # retrieve model based on the time of day it is trained for
        model_name = get_model_name(model_name, time_of_day)
    
    full_path = model_file_path / model_name 
    full_path = full_path.with_suffix('.joblib')   
    model = load(full_path)
    
    return model