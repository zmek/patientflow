from joblib import dump


def save_model(model, model_name, model_file_path):
    """
    Save trained model(s) to disk.

    Parameters
    ----------
    model : object or dict
        A single model instance or a dictionary of models to save.
    model_name : str
        Base name to use for saving the model(s).
    model_file_path : Path
        Directory path where the model(s) will be saved.

    Returns
    -------
    None
    """

    if isinstance(model, dict):
        # Handle dictionary of models (e.g., admission models)
        for name, m in model.items():
            full_path = model_file_path / name
            full_path = full_path.with_suffix(".joblib")
            dump(m, full_path)
    else:
        # Handle single model (e.g., specialty or yet-to-arrive model)
        full_path = model_file_path / model_name
        full_path = full_path.with_suffix(".joblib")
        dump(model, full_path)
