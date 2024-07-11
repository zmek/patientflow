import yaml
from typing import Any, Dict, Tuple, Union, List, Optional


def load_config_file(
    config_file_path: str, return_start_end_dates: bool = False
) -> Optional[
    Union[
        Dict[str, Any],
        Tuple[str, str],
        Tuple[
            List[Tuple[int, int]], str, str, str, str, float, float, float, float, int
        ],
    ]
]:
    try:
        with open(config_file_path, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The file '{config_file_path}' was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None

    try:
        if return_start_end_dates:
            # load the dates used in saved data for uclh versions
            if "file_dates" in config and config["file_dates"]:
                start_date, end_date = [item for item in config["file_dates"]]
            else:
                print(
                    "Error: 'file_dates' key not found or empty in the configuration file."
                )
                return None

        # Convert list of times of day at which predictions will be made (currently stored as lists) to list of tuples
        if "prediction_times" in config:
            prediction_times = [tuple(item) for item in config["prediction_times"]]
        else:
            print("Error: 'prediction_times' key not found in the configuration file.")
            return None

        # Load the dates defining the beginning and end of training, validation and test sets
        if "modelling_dates" in config:
            start_training_set, start_validation_set, start_test_set, end_test_set = [
                item for item in config["modelling_dates"]
            ]
        else:
            print("Error: 'modelling_dates' key not found in the configuration file.")
            return None

        x1 = float(config.get("x1", 0))
        y1 = float(config.get("y1", 0))
        x2 = float(config.get("x2", 0))
        y2 = float(config.get("y2", 0))
        prediction_window = config.get("prediction_window", 480)

        if return_start_end_dates:
            return (start_date, end_date)
        else:
            return (
                prediction_times,
                start_training_set,
                start_validation_set,
                start_test_set,
                end_test_set,
                x1,
                y1,
                x2,
                y2,
                prediction_window,
            )
    except KeyError as e:
        print(f"Error: Missing key in the configuration file: {e}")
        return None
    except ValueError as e:
        print(f"Error: Invalid value found in the configuration file: {e}")
        return None
