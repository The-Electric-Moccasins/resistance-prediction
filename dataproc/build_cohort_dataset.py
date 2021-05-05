import numpy as np
from pandas import DataFrame

from dataproc import cohort
from dataproc.io import write_dataframe, load_dataframe
from hyper_params import HyperParams


def build_cohort(params: HyperParams, df_features: DataFrame, datafile='data/fulldata.npy'):
    df_cohort = load_labels(params)
    # df_cohort = load_dataframe('df_cohort')

    # Join the cohort on the features
    df_full_data = df_cohort.set_index(['hadm_id']).join(df_features.set_index(['hadm_id']), how='inner')

    df_full_data = set_target_feature_name(df_full_data)

    print(f"cohort dataset: {df_full_data.shape}")

    write_dataframe(df_full_data, 'df_full_data')
    df_temp = df_full_data.copy()
    if 'hadm_id' in df_temp.columns:
        df_temp = df_temp.drop(columns='hadm_id')

    # df_full_data = load_dataframe('df_full_data')
    np_fulldata = df_temp.to_numpy()
    # Save to a file
    np.save(datafile, np_fulldata)
    print(f"cohort data saved to {datafile}")

    return df_full_data


def build_cohort_bact(params: HyperParams, df_features: DataFrame):
    df_cohort = load_bacteria_labels(params)
    # df_cohort = load_dataframe('df_cohort')

    # Join the cohort on the features
    df_full_data = df_cohort.set_index(['hadm_id']).join(df_features.set_index(['hadm_id']), how='inner')

    df_full_data = set_target_feature_name(df_full_data, 'resistant_label', 'y')

    print(f"cohort dataset: {df_full_data.shape}")

    write_dataframe(df_full_data, 'df_full_data')

    # df_full_data = load_dataframe('df_full_data')
    np_fulldata = df_full_data.to_numpy()
    # Save to a file
    datafile = 'data/fulldata.npy'
    np.save(datafile, np_fulldata)
    print(f"cohort data saved to {datafile}")

    return df_full_data


def set_target_feature_name(df_full_data, original_name = 'RESISTANT_YN', new_name='y'):

    y_col = df_full_data[original_name].to_numpy()
    df_full_data = df_full_data.drop(columns=[original_name])
    df_full_data[new_name] = y_col
    return df_full_data


def load_labels(params):
    df_cohort = cohort.query_esbl_pts(params.observation_window_hours)
    df_cohort = cohort.remove_dups(df_cohort)
    df_cohort = df_cohort[['hadm_id', 'RESISTANT_YN']]
    print(f"df_labels: {df_cohort.shape}")
    write_dataframe(df_cohort, 'df_cohort')
    return df_cohort


def load_bacteria_labels(params):
    df_cohort = cohort.query_esbl_bacteria_label(params.observation_window_hours)
    df_cohort = df_cohort[['hadm_id', 'resistant_label']]
    print(f"df_labels: {df_cohort.shape}")
    write_dataframe(df_cohort, 'df_cohort')
    return df_cohort
