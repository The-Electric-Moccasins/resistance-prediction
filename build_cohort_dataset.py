import numpy as np
from pandas import DataFrame

from dataproc import cohort
from dataproc.io import write_dataframe, load_dataframe
from hyper_params import HyperParams



def build_cohort(params: HyperParams, df_features: DataFrame):

    df_cohort = load_labels(params)
    # df_cohort = load_dataframe('df_cohort')

    # Join the cohort on the features
    df_full_data = df_cohort.set_index(['hadm_id']).join(df_features.reset_index().set_index(['hadm_id']), how='inner')

    df_full_data = set_target_feature_name(df_full_data)

    write_dataframe(df_full_data, 'df_full_data')

    # df_full_data = load_dataframe('df_full_data')
    np_fulldata = df_full_data.to_numpy()
    # Save to a file
    datafile = 'data/fulldata.npy'
    np.save(datafile, np_fulldata)
    print(f"cohort data saved to {datafile}")
    
    return df_full_data


def set_target_feature_name(df_full_data):
    y_col = df_full_data['RESISTANT_YN']
    df_full_data = df_full_data.drop(columns=['RESISTANT_YN'])
    df_full_data['y'] = y_col
    df_full_data = df_full_data.reset_index(drop=True)
    return df_full_data


def load_labels(params):
    df_cohort = cohort.query_esbl_pts(params.observation_window_hours)
    df_cohort = cohort.remove_dups(df_cohort)
    df_cohort = df_cohort[['hadm_id', 'RESISTANT_YN']]
    print(f"df_cohort: {df_cohort.shape}")
    write_dataframe(df_cohort, 'df_cohort')
    return df_cohort

