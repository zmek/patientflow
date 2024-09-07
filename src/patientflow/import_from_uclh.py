import pandas as pd
import numpy as np
from dateutil.parser import parse
import pandas as pd

from prepare import prep_uclh_dataset_for_inference, assign_mrn_to_training_validation_test_set



def prepare_age_and_dates(df):# conversions necessary for each datetime column

    # calculate age on arrival
    df['age_on_arrival'] = ( pd.to_timedelta((pd.to_datetime(df['arrival_datetime']).dt.date - pd.to_datetime(df['date_of_birth']).dt.date)).dt.days / 365.2425).apply(lambda x: np.floor(x) if pd.notna(x) else x)
    # convert to groups
    bins = [-1, 18, 25, 35, 45, 55, 65, 75, 102]
    labels = ['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-102']
    df['age_group'] = pd.cut(df['age_on_arrival'], bins=bins, labels=labels, right=True)
    
    if 'snapshot_datetime' in list(df.columns):

        # df['snapshot_datetime'] = df['snapshot_datetime'].dt.tz_localize  ('UTC') 
        df['prediction_time'] = df['snapshot_datetime'].dt.strftime('%H,%M').apply(lambda x: tuple(map(int, x.split(','))))
        df['snapshot_date'] = pd.to_datetime(df['snapshot_datetime']).dt.date  

        # Calculate elapsed time in ED
        df['elapsed_los'] = df['snapshot_datetime']- df['arrival_datetime']
        df['elapsed_los'] = df['elapsed_los'].dt.total_seconds()

        # remove rows with los < 0
        print("Removing " + str(len(df[df.elapsed_los  < 0])) + " rows with negative elapsed_los")
        df = df[df.elapsed_los  >= 0]
        
    return(df)
    
def shift_dates_into_future(df, yta, seed_path):
    # Adjust all dates to anonymise visits
    print("\nConverting dates to anonymise visits. Current min and max snapshot dates:")
    print(df.snapshot_date.min())
    print(df.snapshot_date.max())
        
    # Read the seed from a saved file
    with open(seed_path, 'r') as file:
        seed = int(file.read().strip())
    # Set the seed for numpy
    np.random.seed(seed=seed)
    n = np.random.randint(1, 10*52)  
    
    # print(new.snapshot_date.min())
    df.loc[:, 'snapshot_date'] = df['snapshot_date'] + pd.Timedelta(days=n*7)
    df.loc[:, 'snapshot_datetime'] = df['snapshot_datetime'] + pd.Timedelta(days=n*7)
    df.loc[:, 'arrival_datetime'] = df['arrival_datetime'] + pd.Timedelta(days=n*7)
    df.loc[:, 'departure_datetime'] = df['departure_datetime'] + pd.Timedelta(days=n*7)

    print("New min and max snapshot dates:")
    print(df.snapshot_date.min())
    print(df.snapshot_date.max())

    yta['arrival_datetime'] = yta['arrival_datetime'] + pd.Timedelta(days=n*7)
    yta['departure_datetime'] = yta['departure_datetime'] + pd.Timedelta(days=n*7)
    return(df, yta)


def map_consultations_to_types(df, name_mapping):
    
    # Create a dictionary to map consultation codes to types
    code_to_type = dict(zip(name_mapping['code'], name_mapping['type']))
    
    # Define a function to map a list of consultation codes to their types
    def map_codes_to_types(codes):
        return [code_to_type.get(code, 'unknown') for code in codes]
    
    # Apply the function to the consultations columns
    df['consultation_sequence'] = df['consultation_sequence'].apply(map_codes_to_types)
    df['final_sequence'] = df['final_sequence'].apply(map_codes_to_types)
   
    return df


def reformat_uclh_data_for_modelling(df, yta, start_training_set, start_validation_set, start_test_set, end_test_set, seed_path, uclh, name_mapping, spec_mapping, remove_bed_requests = True):

    if uclh:
        print("Preparing dataset for UCLH")
        exclude_minority_categories = False
    else:
        print("Preparing dataset for generic use")
        exclude_minority_categories = True

    # map encounters to a new number
    encounter_mapping = {encounter: idx for idx, encounter in enumerate(df['encounter'].unique(), 1)}
    df['visit_number'] = df['encounter'].map(encounter_mapping)

    # convert dates where needed, and create elapsed los 
    df = prepare_age_and_dates(df)
    yta = prepare_age_and_dates(yta)

    # remove any visits with negative length of stay
    print(f"Removing {str(sum(df.elapsed_los <0))} rows with negative length of stay")
    df = df[df.elapsed_los >= 0]

    # shift dates into future to anonymise the data
    df, yta = shift_dates_into_future(df, yta, seed_path)

    # remove dates outside range of training, validation and test sets
    df = df[(df.arrival_datetime.dt.date >= start_training_set) & ( df.arrival_datetime.dt.date <= end_test_set)]
    yta = yta[(yta.arrival_datetime.dt.date >= start_training_set) & ( yta.arrival_datetime.dt.date <= end_test_set)]

    # assign each mrn to only the training, validation or test set
    df, yta = assign_mrn_to_training_validation_test_set(df, start_training_set, start_validation_set, start_test_set, end_test_set, yta = yta)

    # Remove any snapshots that fall outside the start and end dates for the relevant set
    df = df[
        ((df.training_validation_test == 'train') & (df.arrival_datetime.dt.date < start_validation_set)) |
        ((df.training_validation_test == 'valid') & (df.arrival_datetime.dt.date >= start_validation_set) & (df.arrival_datetime.dt.date < start_test_set)) |
        ((df.training_validation_test == 'test') & (df.arrival_datetime.dt.date >= start_test_set) & (df.arrival_datetime.dt.date < end_test_set))
    ]


    # use the prep_dataset_for_inference function to get df ready for training of the ML model
    print("\nPreparing datasets without bed requests for input into ML; note this may take some time")
    visits = prep_uclh_dataset_for_inference(df, uclh, remove_bed_requests, exclude_minority_categories, inference_time = False)
    


    merged_visits = visits.merge(spec_mapping[['SubSpecialty', 'Specialty']].drop_duplicates(), how='left', left_on='specialty', right_on='SubSpecialty').drop(
        ['SubSpecialty', 'specialty'], axis = 1
    ).rename(
        columns = {'consultations' : 'consultation_sequence',
                   'all_consultations' : 'final_sequence',
                  'Specialty' : 'specialty'}
    )

    merged_visits['consultation_sequence'] = merged_visits['consultation_sequence'].apply(lambda x: tuple(x) if x else ())
    merged_visits['final_sequence'] = merged_visits['final_sequence'].apply(lambda x: tuple(x) if x else ())
    
    # for generic dataset, mask the consults data
    if not uclh:
        merged_visits = map_consultations_to_types(merged_visits, name_mapping)
        
    merged_visits.index = visits.index

    yta = yta[
        ((yta.training_validation_test == 'train') & (yta.arrival_datetime.dt.date < start_validation_set)) |
        ((yta.training_validation_test == 'valid') & (yta.arrival_datetime.dt.date >= start_validation_set) & (yta.arrival_datetime.dt.date < start_test_set)) |
        ((yta.training_validation_test == 'test') & (yta.arrival_datetime.dt.date >= start_test_set) & (yta.arrival_datetime.dt.date < end_test_set))
    ]

    yta.loc[:, 'is_child'] = yta['age_group'] == '0-17'

    if not uclh:
        yta = yta[~yta.sex.isin(["U", "I"])]

    merged_yta = yta.merge(spec_mapping[['SubSpecialty', 'Specialty']].drop_duplicates(), how='left', left_on='specialty', right_on='SubSpecialty').drop(
        ['SubSpecialty', 'specialty'], axis = 1
    ).rename(
        columns = {
                  'Specialty' : 'specialty'}
    )
    
    merged_yta.index = yta.index
    merged_yta = merged_yta[['training_validation_test', 'arrival_datetime', 'sex', 'specialty', 'is_child']]
    
    return(merged_visits, merged_yta)
