import pandas as pd


DATA_DIR='data'

def write_dataframe(df, filename, data_dir = DATA_DIR):
    destination = f"{DATA_DIR}/{filename}.parquet"
    df.to_parquet(destination)


def load_dataframe(filename, data_dir = DATA_DIR):
    destination = f"{DATA_DIR}/{filename}.parquet"
    df = pd.read_parquet(destination)
    return df