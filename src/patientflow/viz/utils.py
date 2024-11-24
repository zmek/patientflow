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
