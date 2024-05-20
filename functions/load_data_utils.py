import pandas as pd

def convert_set_to_dummies(df, column, prefix):
    # Explode the set into rows
    exploded_df = df[column].explode().dropna().to_frame()
    
    # Create dummy variables for each unique item with a specified prefix
    dummies = pd.get_dummies(exploded_df[column], prefix=prefix)
    
    # # Sum the dummies back to the original DataFrame's index
    dummies = dummies.groupby(dummies.index).sum()
    
    # # Join the dummy variables with the original DataFrame
    # # Fill missing values with 0s (where the set did not have the specific item)
    # result_df = df.join(dummies, how='left').fillna(0)
    
    # # Convert float dummies to int (since dummies are 0 or 1)
    # for col in dummies.columns:
    #     result_df[col] = result_df[col].astype(int)
    
    return dummies

def convert_dict_to_dummies(df, column, prefix):
    # Create a DataFrame from the dictionary column
    dict_df = df[column].apply(pd.Series)
    
    # Add a prefix to the column names
    dict_df.columns = [f"{prefix}_{col}" for col in dict_df.columns]
    
    # # Join the new dictionary DataFrame with the original DataFrame
    # # Fill missing values with 0 (where the dictionary did not have the specific key)
    # result_df = df.join(dict_df, how='left').fillna(0)
    
    return dict_df