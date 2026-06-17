import pandas as pd
import ast

def clean_list_column(val):
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

def load_and_preprocess_data(csv_path):
    """Loads the board games dataset and cleans structural string lists."""
    df = pd.read_csv(csv_path)
    df['mechanics_list'] = df['mechanics'].apply(clean_list_column)
    df['categories_list'] = df['categories'].apply(clean_list_column)
    return df
