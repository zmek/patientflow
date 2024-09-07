from dateutil.parser import parse
import pandas as pd
import random
from load_data_utils import get_dict_cols



def is_date(string):
    try: 
        parse(string)
        return True
    except ValueError:
        return False
    
# Function to generate description based on column name
def generate_description(col_name):

    manual_descriptions = get_manual_descriptions()

    # Check if manual description is provided
    if col_name in manual_descriptions:
        return manual_descriptions[col_name]
    
    if col_name.startswith('num') and not col_name.startswith('num_obs') and not col_name.startswith('num_orders'):
        return 'Number of times ' + col_name[4:] + ' has been recorded'
    if col_name.startswith('num_obs'):
        return 'Number of observations of ' + col_name[8:]
    if col_name.startswith('latest_obs'):
        return 'Latest result for ' + col_name[11:]
    if col_name.startswith('latest_lab'):
        return 'Latest result for ' + col_name[19:]
    if col_name.startswith('lab_orders'):
        return 'Request for lab battery ' + col_name[11:] + ' has been placed'
    if col_name.startswith('visited'):
        return 'Patient visited ' + col_name[8:] + ' previously or is there now'
    else:
        return col_name
    
def additional_details(column, col_name):
    
    # Convert to datetime if it's an object but formatted as a date
    if column.dtype == 'object' and all(is_date(str(x)) for x in column.dropna().unique()):
        column = pd.to_datetime(column)
        return f"Date Range: {column.min().strftime('%Y-%m-%d')} - {column.max().strftime('%Y-%m-%d')}"

    if column.dtype in ['object', 'category', 'bool']:
        # Categorical data: Frequency of unique values
        
        if len(column.value_counts()) <= 12:
            value_counts = column.value_counts(dropna=False).to_dict()
            value_counts = dict(sorted(value_counts.items(), key=lambda x: str(x[0])))
            value_counts_formatted = {k: f"{v:,}" for k, v in value_counts.items()}
            return f"Frequencies: {value_counts_formatted}"
        value_counts = column.value_counts(dropna=False)[0:12].to_dict()
        value_counts = dict(sorted(value_counts.items(), key=lambda x: str(x[0])))
        value_counts_formatted = {k: f"{v:,}" for k, v in value_counts.items()}
        return f"Frequencies (highest 12): {value_counts_formatted}"
        
    if pd.api.types.is_float_dtype(column):
        # Float data: Range with rounding
        na_count = column.isna().sum()
        column = column.dropna()
        return f"Range: {column.min():.2f} - {column.max():.2f},  Mean: {column.mean():.2f}, Std Dev: {column.std():.2f}, NA: {na_count}"
    if pd.api.types.is_integer_dtype(column):
        # Float data: Range without rounding
        na_count = column.isna().sum()
        column = column.dropna()
        return f"Range: {column.min()} - {column.max()}, Mean: {column.mean():.2f}, Std Dev: {column.std():.2f}, NA: {na_count}"
    if pd.api.types.is_datetime64_any_dtype(column):
        # Datetime data: Minimum and Maximum dates
        return f"Date Range: {column.min().strftime('%Y-%m-%d %H:%M')} - {column.max().strftime('%Y-%m-%d %H:%M')}"
    else:
        return "N/A"
    
def find_group_for_colname(column, dict_col_groups):
    for key, values_list in dict_col_groups.items():
        if column in values_list:
            return key
    return None
    
def get_manual_descriptions():
    manual_descriptions = {
    'snapshot_id': 'Unique identifier for the visit snapshot (an internal reference field only)',
    'snapshot_date': 'Date of visit, shifted by a random number of days',
    'visit_number': 'Hospital visit number (replaced with fictional number, but consistent across visit snapshots is retained)',
    'arrival_method': 'How the patient arrived at the ED',
    'current_location_type': 'Location in ED currently',
    'sex': 'Sex of patient',
    'age_on_arrival': 'Age in years on arrival at ED',
    'elapsed_los': 'Elapsed time since patient arrived in ED (seconds)',
    'num_obs' : 'Number of observations recorded',
    'num_obs_events': 'Number of unique events when one or more observations have been recorded', 
    'num_obs_types': 'Number of types of observations recorded', 
    'num_lab_batteries_ordered': 'Number of lab batteries ordered (each many contain multiple tests)',
    'has_consult': 'One or more consult request has been made',
    'total_locations_visited': 'Number of ED locations visited',
    'is_admitted': 'Patient was admitted after ED',
    'hour_of_day': 'Hour of day at which visit was sampled',
    'consultation_sequence': 'Consultation sequence at time of snapshot',
    'has_consultation': 'Consultation request made before time of snapshot',
    'final_sequence': 'Consultation sequence at end of visit',
    'observed_specialty': 'Specialty of admission at end of visit',
    'random_number': 'A random number that will be used during model training to sample one visit snapshot per visit',
    'prediction_time': 'The time of day at which the visit was observed',
    'training_validation_test': 'Whether visit snapshot is assigned to training, validation or test set',
    'observed_specialty' : 'Specialty of admission',
    'age_group' : 'Age group',
    'is_child' : 'Is under age of 18 on day of arrival',
    'ed_visit_start_dttm' : 'Timestamp of visit start',
    'random_number': 'A number added to enable sampling of one snapshot per visit'

    
       }
    return(manual_descriptions)
    
def write_data_dict(df, dict_name, dict_path):

    cols_to_exclude = ['snapshot_id', 'visit_number']

    if 'visits' in dict_name :

    
        df_admitted = df[df.is_admitted]
        df_not_admitted = df[df.is_admitted == False]
    
        dict_col_groups = get_dict_cols(df)
    
        data_dict = pd.DataFrame({
            'Variable type': [find_group_for_colname(col, dict_col_groups) for col in df.columns],
            'Column Name': df.columns,
            'Data Type': df.dtypes,
            'Description': [generate_description(col) for col in df.columns], 
            'Whole dataset': [additional_details(df[col], col) if col not in cols_to_exclude else '' for col in df.columns],
            'Admitted': [additional_details(df_admitted[col], col) if col not in cols_to_exclude else '' for col in df_admitted.columns],
            'Not admitted': [additional_details(df_not_admitted[col], col) if col not in cols_to_exclude else '' for col in df_not_admitted.columns]
    
        })
        data_dict["Whole dataset"] = data_dict["Whole dataset"].str.replace("'", "")
        data_dict["Admitted"] = data_dict["Admitted"].str.replace("'", "")
        data_dict["Not admitted"] = data_dict["Not admitted"].str.replace("'", "")


    else:
        
        data_dict = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes,
        'Description': [generate_description(col) for col in df.columns], 
        'Additional Details': [additional_details(df[col], col) if col not in cols_to_exclude else '' for col in df.columns]

    })
        data_dict["Additional Details"] = data_dict["Additional Details"].str.replace("'", "")


    # Export to Markdown and csv for data dictionary
    data_dict.to_markdown(str(dict_path) + '/' + dict_name + '.md', index=False)
    data_dict.to_csv(str(dict_path) + '/' + dict_name + '.csv', index=False)

    
    return(data_dict)

