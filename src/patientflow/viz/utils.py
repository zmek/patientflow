def clean_title_for_filename(title):
    """
    Clean a title string to make it suitable for use in filenames.

    Args:
        title (str): The title to clean

    Returns:
        str: The cleaned title, safe for use in filenames
    """
    replacements = {" ": "_", "%": "", "\n": "", ",": "", ".": ""}

    clean_title = title
    for old, new in replacements.items():
        clean_title = clean_title.replace(old, new)
    return clean_title


def format_prediction_time(prediction_time):
    """Format prediction time string from 'HHMM' to 'HH:MM' format.

    Args:
        prediction_time (str): Prediction time string, possibly containing underscores

    Returns:
        str: Formatted time string in 'HH:MM' format
    """
    # Split the string by underscores and take the last element
    last_part = prediction_time.split("_")[-1]
    # Add a colon in the middle
    return f"{last_part[:2]}:{last_part[2:]}"
