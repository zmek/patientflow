from typing import Dict
from errors import MissingKeysError

def create_special_category_objects(uclh):
    special_category_dict = {
        "medical": 0.0,
        "surgical": 0.0,
        "haem/onc": 0.0,
        "paediatric": 1.0,
    }

    # Function to determine if the patient is a child
    def is_paediatric_uclh(row):
        return row["age_on_arrival"] < 18

    def is_paediatric_non_uclh(row):
        return row["age_group"] == "0-17"

    if uclh:
        special_category_func = is_paediatric_uclh
    else:
        special_category_func = is_paediatric_non_uclh

    def default_specialty(row):
        return True  # Default function for other specialties

    special_func_map = {
        "paediatric": special_category_func,
        "default": default_specialty,
    }

    special_params = {
        "special_category_func": special_category_func,
        "special_category_dict": special_category_dict,
        "special_func_map": special_func_map,
    }

    return special_params


def validate_special_category_objects(special_params: Dict[str, any]) -> None:
    required_keys = ["special_category_func", "special_category_dict", "special_func_map"]
    missing_keys = [key for key in required_keys if key not in special_params]

    if missing_keys:
        raise MissingKeysError(missing_keys)