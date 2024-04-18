from joblib import load


def get_model_name(tod_):
    """
    Create a model name based on the time of day.

    Parameters:
    tod_ (tuple): A tuple representing the time of day (hour, minute).

    Returns:
    str: A string representing the model name based on the time of day.
    """
    hour_, min_ = tod_
    min_ = f"{min_}0" if min_ % 60 == 0 else str(min_)
    model_name = 'ed_admission_' + f"{hour_:02}" + min_
    return model_name



# Data preprocessing for a specific time of day
def preprocess_data(df, tod_, exclude_columns, single_episode_slice_per_visit = True):
    
    # get visits that were in at the time of day in question
    df_tod = df[df.time_of_day == tod_]
    
    if single_episode_slice_per_visit:
        df_single = df_tod[df_tod.groupby(['visit_number'])['random_number'].transform(max) == df_tod['random_number']].drop(columns=['random_number'])
        
        # create label
        y = df_single.pop("is_admitted").astype(int)
    
        # drop the columns not used
        df_single.drop(columns=exclude_columns, inplace=True)
    
        return df_single, y
    
    df_tod = df_tod.copy()
    df_tod.drop(columns=['random_number'] + exclude_columns, inplace=True)
    y = df_tod.pop("is_admitted").astype(int)

    return df_tod, y 
    
    
    
    # include one one episode slice per visit and drop the random number
    
    


def load_saved_model(tod_, model_file_path):
        
    # retrieve model based on the time of day it is trained for
    MODEL__ED_ADMISSIONS__NAME = get_model_name(tod_)
    
    full_path = model_file_path / MODEL__ED_ADMISSIONS__NAME 
    full_path = full_path.with_suffix('.joblib')   
    model = load(full_path)
    
    return model