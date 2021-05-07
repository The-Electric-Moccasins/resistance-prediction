import pandas as pd
import torch
from config import DATA_DIR


def write_dataframe(df, filename, data_dir = DATA_DIR):
    destination = f"{DATA_DIR}/{filename}.parquet"
    df.to_parquet(destination)


def load_dataframe(filename, data_dir = DATA_DIR):
    source = f"{DATA_DIR}/{filename}.parquet"
    df = pd.read_parquet(source)
    return df


def write_serialized_model(model, filename):
    destination = f"{DATA_DIR}/{filename}.pic"
    with open(destination, 'bw') as f:
        torch.save(model, f, pickle_protocol=4)

        
def load_serialized_model(filename):
    source = f"{DATA_DIR}/{filename}.pic"
    with open(source, 'rb') as f:
        model = torch.load(f)
    return model