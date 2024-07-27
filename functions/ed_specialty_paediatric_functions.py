def create_special_category_objects(uclh):

    special_category_dict = {
        'medical': 0.0,
        'surgical': 0.0,
        'haem/onc': 0.0,
        'paediatric': 1.0
    }
    
    # Function to determine if the patient is a child
    if uclh:
        special_category_func = lambda row: row['age_on_arrival'] < 18 
    else:
        special_category_func = lambda row: row['age_group'] =='0-17' 
    
    special_func_map = {
        'paediatric': special_category_func,
        'default': lambda row: True  # Default function for other specialties
    }

    return special_category_dict, special_category_func, special_func_map