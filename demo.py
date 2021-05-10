from baseline_model import RandomForest
from dataproc import io
import config


def demo():
    df_dataset = io.load_dataframe('df_cohort', data_dir=config.DATA_DIR)  # 'defaults to './data'
    RandomForest(df_dataset)


if __name__ == '__main__':
    demo()