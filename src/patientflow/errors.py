class ModelLoadError(Exception):
    pass


class MissingKeysError(ValueError):
    def __init__(self, missing_keys):
        super().__init__(f"special_params is missing required keys: {missing_keys}")
        self.missing_keys = missing_keys
