'''
Contains functions for loading built-in datasets
'''

import pandas as pd
import os
import ast # to convert tuples to strings

from pathlib import Path

def ed_admissions_get_data(path_ed_data):
    '''
    Loads XXX ED visits

    Returns:
    pd.DataFrame: A dataframe with the ED visits. See data dictionary 
    '''
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(Path().home(), path_ed_data)
    
    # read dataframe
    df = pd.read_csv(path, parse_dates=True)
    
    # sort by visit and date if in dataset
    sort_columns = [col for col in ['visit_number', 'horizon_dt'] if col in df.columns]
    if sort_columns:
        df.sort_values(sort_columns, inplace=True)


    
    # convert strings to tuples
    if 'time_of_day' in df.columns:
        df['time_of_day'] = df['time_of_day'].apply(lambda x: ast.literal_eval(x))
    
    return df
