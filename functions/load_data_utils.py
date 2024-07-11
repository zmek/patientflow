import pandas as pd


def convert_set_to_dummies(df, column, prefix):
    # Explode the set into rows
    exploded_df = df[column].explode().dropna().to_frame()

    # Create dummy variables for each unique item with a specified prefix
    dummies = pd.get_dummies(exploded_df[column], prefix=prefix)

    # # Sum the dummies back to the original DataFrame's index
    dummies = dummies.groupby(dummies.index).sum()

    # Convert dummy variables to boolean
    dummies = dummies.astype(bool)

    return dummies


def convert_dict_to_values(df, column, prefix):
    def extract_relevant_value(d):
        if isinstance(d, dict):
            if "value_as_real" in d or "value_as_text" in d:
                return (
                    d.get("value_as_real")
                    if d.get("value_as_real") is not None
                    else d.get("value_as_text")
                )
            else:
                return d  # Return the dictionary as is if it does not contain 'value_as_real' or 'value_as_text'
        return d  # Return the value as is if it is not a dictionary

    # Apply the extraction function to each entry in the dictionary column
    extracted_values = df[column].apply(
        lambda x: {k: extract_relevant_value(v) for k, v in x.items()}
    )

    # Create a DataFrame from the processed dictionary column
    dict_df = extracted_values.apply(pd.Series)

    # Add a prefix to the column names
    dict_df.columns = [f"{prefix}_{col}" for col in dict_df.columns]

    return dict_df
