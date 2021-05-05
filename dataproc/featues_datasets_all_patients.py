import re

import numpy as np
import pandas as pd
from pandas import DataFrame

from dataproc.io import write_dataframe, load_dataframe
from dataproc import cohort
from dataproc import create_dataset
from dataproc.proc_utils import drop_sparse_columns, stanardize_numeric_values, replace_missing_val, bin_numerics

from hyper_params import HyperParams

from config import DATA_DIR


def run(params :HyperParams, binning_numerics=False, create_patients_list_view=True, create_lab_events=True):
    """
    Build feature datasets for ALL admissions that were still hospitalized
    by the end of the observation window
    returns as a data frame, and also persisted as "df_final_dataset"
    """

    # create list of patients, max_observation_window
    if create_patients_list_view: 
        df_all_pts_within_observation_window, view_name_all_pts_within_observation_window = \
            cohort.query_all_pts_within_observation_window(params.observation_window_hours)
        write_dataframe(df_all_pts_within_observation_window, 'df_all_pts_within_observation_window')
    else:
        view_name_all_pts_within_observation_window = f'default.all_pts_{params.observation_window_hours}_hours'
        df_all_pts_within_observation_window = load_dataframe('df_all_pts_within_observation_window')

    # generate features for all patients (under observation window)

    ## Static features
    df_static_data = load_static_features(view_name_all_pts_within_observation_window)

    # Antibiotics prescriptions:
    onehotrx_df = load_antibiotics(view_name_all_pts_within_observation_window)

    # Previous admissions:
    admits_df = load_previous_admissions(view_name_all_pts_within_observation_window, params, binning_numerics)

    # Open Wounds Diagnosis:
    wounds_df = load_open_wounds(view_name_all_pts_within_observation_window)

    # Intubation procedures:
    df_intubation = load_intubation_procedures(view_name_all_pts_within_observation_window)

    # Note Events:
    notes = load_notes(view_name_all_pts_within_observation_window)

    df_antibiotics_history = load_antibiotics_history(notes)

    # lab events
    if create_lab_events:
        df_lab_events = load_lab_events(view_name_all_pts_within_observation_window)
    else:
        df_lab_events = load_dataframe('df_lab_events')

    # lab results
    df_lab_results = get_lab_results(df_lab_events)

    df_lab_flags = get_lab_flags(df_lab_events, binning_numerics)

    # join lab results
    df_lab = df_lab_results.merge(df_lab_flags, how='left', on=['hadm_id'])
    # sort columns by lab tests names
    df_lab = df_lab.set_index('hadm_id').reindex(sorted(df_lab.columns), axis=1).drop(columns=['hadm_id']).reset_index()

    df_dataset_unprocessed = join_static_and_lab_data(df_lab, df_static_data)

    
    if binning_numerics:
        # numeric values: bin
        df_dataset_unprocessed = clean_and_bin_numeric_values(df_dataset_unprocessed, params)
    else:
        # numeric values: clean and standardize
        df_dataset_unprocessed = clean_and_standardize_numeric_values(df_dataset_unprocessed)
    

    

    # join on antibiotics, previous admissions and wound
    df_dataset_processed = df_dataset_unprocessed
    df_dataset_processed = pd.merge(df_dataset_processed, onehotrx_df, on='hadm_id', how='left')
    df_dataset_processed = pd.merge(df_dataset_processed, admits_df, on='hadm_id', how='left')
    df_dataset_processed = pd.merge(df_dataset_processed, wounds_df, on='hadm_id', how='left')
    df_dataset_processed = pd.merge(df_dataset_processed, df_intubation, on='hadm_id', how='left')
    df_dataset_processed = pd.merge(df_dataset_processed, df_antibiotics_history, on='hadm_id', how='left')

    # categorical values: One Hot Encode
    df_dataset_processed = one_hot_encode_categorical(df_dataset_processed)
    
    df_dataset_processed.fillna(0, inplace=True)

    df_final_dataset = df_dataset_processed
    print(f"df_final_dataset: {df_final_dataset.shape}")
    write_dataframe(df_final_dataset, 'df_final_dataset')
    print(f"dataset data saved as 'df_final_dataset'")
    # df_final_dataset = load_dataframe('df_final_dataset')

    save_auto_encoder_training_data(df_final_dataset)

    return df_final_dataset



def save_auto_encoder_training_data(df_features: DataFrame):
    df_temp = df_features.copy()
    df_temp['y'] = np.zeros((df_features.shape[0],))
    if 'hadm_id' in df_temp.columns:
        df_temp = df_temp.drop(columns='hadm_id')
    autoencoder_fulldata = df_temp.to_numpy()
    # Save to a file
    target_datafile = 'data/autoencoder_fulldata.npy'
    np.save(target_datafile, autoencoder_fulldata)
    print(f"autoencoder training data (with y=0) was saved to {target_datafile}")



def load_static_features(view_name_all_pts_within_observation_window):
    df_static_data = create_dataset.static_data(hadm_ids_table=view_name_all_pts_within_observation_window)
    df_static_data = df_static_data.drop(columns=['admittime'])
    static_feature_names = df_static_data.columns.tolist()
    process_static_data(df_static_data)
    write_dataframe(df_static_data, 'df_static_data')
    # df_static_data = load_dataframe('df_static_data')
    # static_feature_names = df_static_data.columns.tolist()
    return df_static_data


def one_hot_encode_categorical(df_dataset_unprocessed):
    categorical_cols = df_dataset_unprocessed.select_dtypes('object').columns.tolist()
    df_dataset_processed = pd.get_dummies(df_dataset_unprocessed,
                                          columns=categorical_cols,
                                          dummy_na=True,
                                          drop_first=True)
    df_dataset_processed.fillna(0)
    print(f"df_dataset_processed: {df_dataset_processed.shape}")
    write_dataframe(df_dataset_processed, 'df_dataset_processed')
    # df_dataset_processed = load_dataframe('df_dataset_processed')
    return df_dataset_processed


def clean_and_standardize_numeric_values(df_dataset_unprocessed):
    
    columns_to_standardize = select_numeric_values_to_process(df_dataset_unprocessed)
    df_dataset_unprocessed = stanardize_numeric_values(df_dataset_unprocessed, columns_to_standardize)
    numeric_cols = df_dataset_unprocessed.select_dtypes('number')
    numeric_cols = [col for col in numeric_cols if not col.endswith('_flag')]
    df_new_numeric = replace_missing_val(df_dataset_unprocessed, numeric_cols)
    df_dataset_unprocessed = df_dataset_unprocessed.drop(columns=numeric_cols).join(df_new_numeric)
    return df_dataset_unprocessed

def clean_and_bin_numeric_values(df, params):
#     df = df.set_index('hadm_id')
    columns_to_process = select_numeric_values_to_process(df)
    
#     bin_labels=[str(num) for num in range(1,params.num_of_bins_for_numerics+1)]
#     print(bin_labels)
    df = bin_numerics(dataset=df, 
                        numeric_columns=columns_to_process, 
                        bins=params.num_of_bins_for_numerics
                        )
    
    return df

def select_numeric_values_to_process(df):
    df_tmp = df.select_dtypes(include='number')
    columns_to_process = [col for col in df_tmp.columns.tolist() if not col.endswith('_flag') and not col=='hadm_id']
    return columns_to_process
    



def join_static_and_lab_data(df_lab, df_static_data):
    df_lab = df_lab.set_index(['hadm_id'])
    df_static_data = df_static_data.set_index(['hadm_id'])
    df_dataset_unprocessed = df_lab.join(df_static_data, how='inner')  # join on index hadm_id
    print(f"join_static_and_lab_data: {df_dataset_unprocessed.shape}")
    write_dataframe(df_dataset_unprocessed, 'join_static_and_lab_data')
    # df_dataset_unprocessed = load_dataframe('join_static_and_lab_data')
    return df_dataset_unprocessed


def get_lab_flags(df_lab_events, binning_numerics):
    df_lab_flags = pivot_flags_to_columns(df_lab_events, binning_numerics)
    print(f"df_lab_flags: {df_lab_flags.shape}")
    write_dataframe(df_lab_flags, 'df_lab_flags')
    # df_lab_flags = load_dataframe('df_lab_flags')
    # lab_flags_feature_names = df_lab_flags.columns.tolist()
    return df_lab_flags


def get_lab_results(df_lab_events):
    df_lab_results = pivot_labtests_to_columns(df_lab_events)
    fix_lab_results_categories(df_lab_results)
    df_lab_results = df_lab_results.drop(columns=['50827', '50856', '51100', '51482', '50981'])
    print(f"shape before dropping sparses {df_lab_results.shape}")
    df_lab_results = drop_sparse_columns(
        df_lab_results,
        columns=df_lab_results.drop(columns=['hadm_id']).columns.tolist(),
        max_sparsity_to_keep=0.95
    )
    print(f"shape after dropping sparses {df_lab_results.shape}")
    numeric, categorical, weird = detect_data_types(df_lab_results.drop(columns=['hadm_id']))
    set_numeric_columns(df_lab_results, numeric)
    print(f"df_lab_results: {df_lab_results.shape}")
    write_dataframe(df_lab_results, 'df_lab_results')
    # df_lab_results = load_dataframe('df_lab_results')
    # lab_results_feature_names = df_lab_results.columns.tolist()
    return df_lab_results


def detect_data_types(df, columns=None):
    if columns is None:
        columns = df.columns.tolist()
    numeric = []
    categorical = []
    weird = []
    N = df.shape[0]
    for code in columns:
        n_missing = df[code].isna().sum()
        size = N - n_missing
        size_unique = df[code].nunique()
        sum_na = pd.to_numeric(df[code][df[code].notna()], errors='coerce').isna().sum()
        if sum_na / size < 0.05:
            numeric.append(code)
        elif sum_na / size > 0.05 and size_unique < 100:
            categorical.append(code)
        else:
            weird.append(code)
    return numeric, categorical, weird


def set_numeric_columns(df, numeric_columns: list):
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce', axis=1)
    return df


def fix_lab_results_categories(df_lab_results):
    df_lab_results['51501'] = np.where(df_lab_results['51501'].isin(['<1', '1', '2']), '0-2',
                                       np.where(df_lab_results['51501'].isin(['3', '4']), '3-5',
                                                df_lab_results['51501']))
    df_lab_results['51506'] = np.where(df_lab_results['51506'].isin(['CLEAR']), 'Clear',
                                       np.where(df_lab_results['51506'].isin(['SLHAZY']), 'SlHazy',
                                                np.where(df_lab_results['51506'].isin(['HAZY']), 'Hazy',
                                                         np.where(df_lab_results['51506'].isin(['SlCloudy']), 'SlCldy',
                                                                  np.where(df_lab_results['51506'].isin(['CLOUDY']),
                                                                           'Cloudy', df_lab_results['51506'])))))
    df_lab_results['51463'] = np.where(df_lab_results['51463'].isin(['0']), 'NEG',
                                       np.where(df_lab_results['51463'].isin(['NOTDONE']), 'NONE',
                                                np.where(df_lab_results['51463'].isin(['LRG']), 'MANY',
                                                         df_lab_results['51463'])))
    df_lab_results['51508'] = np.where(df_lab_results['51508'].isin(['YELLOW', 'YEL']), 'Yellow',
                                       np.where(df_lab_results['51508'].isin(['STRAW']), 'Straw',
                                                np.where(df_lab_results['51508'].isin(['AMBER', 'AMB']), 'Amber',
                                                         np.where(df_lab_results['51508'].isin(['RED', 'R']), 'Red',
                                                                  np.where(
                                                                      df_lab_results['51508'].isin(['ORANGE', 'O']),
                                                                      'Orange',
                                                                      np.where(df_lab_results['51508'].isin(
                                                                          ['DKAMB', 'DKAMBER', 'DKAMBE', 'DRKAMBER']),
                                                                          'DkAmb',
                                                                          np.where(
                                                                              df_lab_results['51508'].isin([' ']),
                                                                              np.nan,
                                                                              np.where(
                                                                                  df_lab_results['51508'].isin(
                                                                                      ['GREEN', 'Lt', 'NONE',
                                                                                       'DKBROWN', 'Black', 'HAZY',
                                                                                       'LTBROWN', 'BLUE']),
                                                                                  'Other',
                                                                                  df_lab_results['51508']))))))))
    df_lab_results['51508'] = df_lab_results['51508'].map(
        {
            'PINK': 'Pink',
            'None': 'NONE'
        }
    )
    # >80 is a category by iteslef, so keeping it.
    # df_lab_results['51484'] = np.where(df_lab_results['51484'].isin(['>80']), '80',df_lab_results['51484'])
    # >300 is a category by itself, so keeping it
    # df_lab_results['51492'] = np.where(df_lab_results['51492'].isin(['>300']), '300',
    #                     np.where(df_lab_results['51492'].isin([' ']), np.nan, df_lab_results['51492']))
    df_lab_results['51492'] = np.where(df_lab_results['51492'].isin([' ']), np.nan, df_lab_results['51492'])
    df_lab_results['51514'] = np.where(df_lab_results['51514'].isin(['.2']), '0.2',
                                       np.where(df_lab_results['51514'].isin(['>8']), '>8.0',
                                                np.where(df_lab_results['51514'].isin(['>12']), '>12.0',
                                                         np.where(df_lab_results['51514'].isin(['NotDone', ' ']),
                                                                  np.nan, df_lab_results['51514']))))
    df_lab_results['51003'] = np.where(df_lab_results['51003'].isin(
        {'<0.01', 'LESS THAN 0.01', '<0.010', 'LESS THAN 0.010', '<0.10', '<0.02', 'LESS THAN 0.1'}), '0.001',
        np.where(df_lab_results['51003'].isin(
            {'GREATER THAN 25.0', '>25.0', '>25', 'GREATER THAN 25', '>25.00'}), '30.0',
            df_lab_results['51003']))
    df_lab_results['51519'] = np.where(df_lab_results['51519'].isin(['0', 'N']), 'NONE',
                                       np.where(df_lab_results['51519'].isin(['NOTDONE']), np.nan,
                                                df_lab_results['51519']))
    df_lab_results['51266'] = np.where(df_lab_results['51266'].isin(['UNABLE TO ESTIMATE DUE TO PLATELET CLUMPS']),
                                       'NOTDETECTED', df_lab_results['51266'])
    df_lab_results['51478'] = df_lab_results['51478'].map(
        {
            'Neg': 'NEG',
            'N': 'NEG',
        }
    )
    df_lab_results['51484'] = df_lab_results['51484'].map(
        {
            'T': 'TR',
            'Tr': 'TR',
            'Neg': 'NEG',
            '>80': '150.0'
        }
    )
    df_lab_results['51492'] = df_lab_results['51492'].map(
        {
            'Neg': 'NEG',
            '>600': '600.0',
            '>300': '500.0',
        }
    )
    df_lab_results['51463'] = df_lab_results['51463'].map(
        {
            ' ': 'NONE',
            'F': 'FEW',
            'MOD-': 'MOD',
            '1.0': 'RARE',
            '7I': 'NONE',
            '2.0': 'FEW'
        }
    )
    df_lab_results['51003'] = np.where(
        df_lab_results['51003'].isin({'NotDone', 'NOT DONE', 'ERROR', 'NOT DONE , TOTAL CK LESS THAN 100'}), np.nan,
        np.where(df_lab_results['51003'].isin({'>500', 'GREATER THAN 500'}), '600.0',
                 np.where(df_lab_results['51003'].isin({'<1'}), '0.0', df_lab_results['51003'])))
    df_lab_results['50922'] = np.where(df_lab_results['50922'].isin({'NEG', 'NEGATIVE', 'ERROR'}), -1.0,
                                       df_lab_results['50922'])
    df_lab_results['51493'] = df_lab_results['51493'].map(
        {'0-2': '1.0',
         '3-5': '4.0',
         '11-20': '15.0',
         '>50': '80.0',
         '6-10': '8.0',
         '21-50': '35.0',
         '<1': '0.01',
         '>1000': '1100.0',
         'O-2': '1.0',
         ' 3-5': '4.0',
         ' ': np.nan,
         'LOADED': np.nan,
         ' 0-2': '1.0',
         'NOTDONE': np.nan,
         '0-20-2': np.nan,
         '0-2+': np.nan,
         'TNTC': np.nan,
         '3/5': np.nan,
         '21-200-2': np.nan})
    df_lab_results['51516'] = df_lab_results['51516'].map(
        {'0-2': '1.0',
         '3-5': '4.0',
         '11-20': '15.0',
         '>50': '80.0',
         '6-10': '8.0',
         '21-50': '35.0',
         '<1': '0.01',
         '>1000': '1100.0',
         'O-2': '1.0',
         ' 3-5': '4.0',
         ' ': np.nan,
         'LOADED': np.nan,
         ' 0-2': '1.0',
         'NOTDONE': np.nan,
         '0-20-2': np.nan,
         '0-2+': np.nan,
         'TNTC': np.nan,
         '3/5': np.nan,
         '21-200-2': np.nan})
    df_lab_results['51476'] = df_lab_results['51476'].map(
        {'0-2': '1.0',
         '3-5': '4.0',
         '11-20': '15.0',
         '>50': '80.0',
         '6-10': '8.0',
         '21-50': '35.0',
         '<1': '0.01',
         '>1000': '1100.0',
         'O-2': '1.0',
         ' 3-5': '4.0',
         ' ': np.nan,
         'LOADED': np.nan,
         ' 0-2': '1.0',
         'NOTDONE': np.nan,
         '0-20-2': np.nan,
         '0-2+': np.nan,
         'TNTC': np.nan,
         '3/5': np.nan,
         '21-200-2': np.nan,
         '0-2,TRANS': '1.0',
         '<1 /HPF': '0.5',
         '11-20-': '15.0',
         '0.-2': '1.0',
         ' 0-2': '1.0',
         })
    df_lab_results['50911'] = df_lab_results['50911'].map(
        {
            'NotDone': np.nan,
            '>500': '600.0',
            'GREATER THAN 500': '550.0',
            'NOT DONE': np.nan,
            '<1': '0.0',
            'ERROR': np.nan,
            'NOT DONE , TOTAL CK LESS THAN 100': np.nan}
    )
    df_lab_results['50924'] = df_lab_results['50924'].map(
        {
            'GREATER THAN 2000': '5900.0',  # median of > 2000
            'GREATER THAN 1000': '1500.0',  # median of > 1000
            '>2000': '5900.0',
            'GREATER THEN 2000': '5900.0',
            '> 2000': '5900.0',
        }
    )


def pivot_labtests_to_columns(df):
    df = df.reset_index(drop=True)
    df = df.pivot(index=['hadm_id'], columns=['itemid'], values=['value'])
    df.columns = df.columns.to_flat_index()
    df.columns = [str(colname[1]) for colname in df.columns]
    df = df.reset_index(['hadm_id'])

    return df


def pivot_flags_to_columns(df, binning_numerics):
    df = df.copy()
    if binning_numerics:
        value_map = {'abnormal': "Abnormal", 'delta': "Abnormal", 'False': "Normal"}
    else:
        value_map = {'abnormal': 1, 'delta': 1, 'False': -1}
    df['flag'] = df['flag'].fillna('False').map(value_map)
    df['flag_name'] = df['itemid'].astype(str) + pd.Series(["_flag"] * df.shape[0]).astype(str)
    df = df.pivot(index=['hadm_id'], columns=['flag_name'], values=['flag'])
    df.columns = df.columns.to_flat_index()
    df.columns = [str(colname[1]) for colname in df.columns]
    
    if binning_numerics:
#         print("flags before binning")
#         print(df.columns.tolist())
#         df = pd.get_dummies(df, dummy_na=True, drop_first=True)
#         print("flags after binning")
#         print(df.columns.tolist())
        pass
    else:
        df = df.fillna(0)
        df = df.astype('uint8')
    df = df.reset_index(['hadm_id'])
    return df


def load_lab_events(view_name_hadm_ids):
    df_lab_events = create_dataset.lab_events(view_name_hadm_ids)
    df_lab_events = df_lab_events.dropna(subset=['value'])
    df_lab_events['flag'].fillna('False').map({'abnormal': True, 'delta': True, 'False': False}).value_counts()
    print('lab events before selection: ', df_lab_events.shape)
    df_lab_events = keep_last_labtest_instance(df_lab_events)
    print('lab events after selection: ', df_lab_events.shape)
    write_dataframe(df_lab_events, 'df_lab_events')
    # df_lab_events = load_dataframe('df_lab_events')
    return df_lab_events


def keep_last_labtest_instance(df):
    """
    select last instance of every type of test for a patient
    """
    df = df.sort_values('charttime', axis=0)
    df = df.drop_duplicates(subset=['hadm_id', 'itemid'], keep='last', ignore_index=True)
    return df


def process_static_data(dataset):
    dataset['admission_location'] = \
        np.where(dataset['admission_location'].isin(['** INFO NOT AVAILABLE **']), 'EMERGENCY ROOM ADMIT',
                 np.where(dataset['admission_location'].isin(['TRANSFER FROM SKILLED NUR', 'TRANSFER FROM OTHER HEALT',
                                                              'TRANSFER FROM HOSP/EXTRAM']),
                          'TRANSFER FROM MED FACILITY', dataset['admission_location']))
    dataset['language'] = \
        np.where(~dataset['language'].isin(['ENGL', 'SPAN']), 'OTHER', dataset['language'])

    dataset['religion'] = \
        np.where(
            ~dataset['religion'].isin(['CATHOLIC', 'NOT SPECIFIED', 'UNOBTAINABLE', 'PROTESTANT QUAKER', 'JEWISH']),
            'OTHER',
            np.where(dataset['religion'].isin(['UNOBTAINABLE']), 'NOT SPECIFIED', dataset['religion']))

    dataset['ethnicity'] = \
        np.where(dataset['ethnicity'].isin(['ASIAN - CHINESE',
                                            'ASIAN - ASIAN INDIAN',
                                            'ASIAN - VIETNAMESE',
                                            'ASIAN - OTHER',
                                            'ASIAN - FILIPINO',
                                            'ASIAN - CAMBODIAN']), 'ASIAN',
                 np.where(dataset['ethnicity'].isin(['WHITE - RUSSIAN',
                                                     'WHITE - BRAZILIAN',
                                                     'WHITE - OTHER EUROPEAN']), 'WHITE',
                          np.where(dataset['ethnicity'].isin(['BLACK/CAPE VERDEAN',
                                                              'BLACK/HAITIAN',
                                                              'BLACK/AFRICAN']), 'BLACK/AFRICAN AMERICAN',
                                   np.where(dataset['ethnicity'].isin(['HISPANIC/LATINO - PUERTO RICAN',
                                                                       'HISPANIC/LATINO - DOMINICAN',
                                                                       'HISPANIC/LATINO - SALVADORAN',
                                                                       'HISPANIC/LATINO - CUBAN',
                                                                       'HISPANIC/LATINO - MEXICAN']),
                                            'HISPANIC OR LATINO',
                                            np.where(dataset['ethnicity'].isin(['MULTI RACE ETHNICITY',
                                                                                'MIDDLE EASTERN',
                                                                                'PORTUGUESE',
                                                                                'AMERICAN INDIAN/ALASKA NATIVE',
                                                                                'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
                                                                                'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE']),
                                                     'OTHER',
                                                     np.where(dataset['ethnicity'].isin(['UNABLE TO OBTAIN',
                                                                                         'PATIENT DECLINED TO ANSWER']),
                                                              'UNKNOWN/NOT SPECIFIED',
                                                              dataset['ethnicity']))))))
    dataset['marital_status'] = dataset['marital_status'].fillna(value='UNKNOWN')
    # clean the age column
    threshold = 105
    dataset['age'] = dataset['age'].apply(lambda x: x if x < threshold else np.nan)


def load_antibiotics(view_name_hadm_ids):
    rx = create_dataset.prescriptions(view_name_hadm_ids)
    rx['value'] = 1
    rx = rx[['hadm_id', 'drug', 'value']]
    # Drop duplicated records
    rx = rx.drop_duplicates(subset=['hadm_id', 'drug'], keep='first')
    # One-hot encoder for prescriptions
    onehotrx_df = rx.pivot_table(index='hadm_id', columns='drug', values='value', fill_value=0).reset_index()
    onehotrx_df.fillna(0)
    return onehotrx_df


def load_previous_admissions(view_name_hadm_ids, params, binning_numerics=False):
    admits = create_dataset.previous_admissions(view_name_hadm_ids)
    admits_df = admits.groupby('hadm_id').agg({'prev_hadm_id': 'nunique'}).reset_index()
    admits_df = admits_df.rename(columns={'prev_hadm_id': 'n_admits'})
    if binning_numerics:
        # numeric values: bin
        admits_df = bin_numerics(dataset=admits_df, numeric_columns=['n_admits'], bins=params.num_of_bins_for_numerics)
    else:
        # numeric values: clean and standardize
        admits_df = stanardize_numeric_values(admits_df, list_of_clms=['n_admits'])
        
    
    print('Previous admits: ', admits_df.shape)
    print('--------------------------------------------------------------')
    return admits_df


def load_open_wounds(view_name_hadm_ids):
    wounds = create_dataset.open_wounds_diags(view_name_hadm_ids)
    wounds['wounds'] = 1  # wounds indicator column
    # Group on hand_id & drop icd9 code column
    wounds_df = wounds.drop_duplicates(subset=['hadm_id'], keep='first')
    wounds_df = wounds_df.drop(columns='icd9_code')
    print('Open wounds: ', wounds_df.shape)
    print('--------------------------------------------------------------')
    return wounds_df


def load_antibiotics_history(notes):
    # List antibiotics
    antibitics_list = ['antibiotic', 'antibiotics', 'amikacin', 'ampicillin', 'sulbactam',
                       'cefazolin', 'cefepime', 'cefpodoxime', 'ceftazidime',
                       'ceftriaxone', 'cefuroxime', 'chloramphenicol', 'ciprofloxacin',
                       'clindamycin', 'daptomycin', 'erythromycin', 'gentamicin', 'imipenem',
                       'levofloxacin', 'linezolid', 'meropenem', 'nitrofurantoin', 'oxacillin',
                       'penicillin', 'penicillin G', 'piperacillin', 'tazobactam',
                       'rifampin', 'tetracycline', 'tobramycin', 'trimethoprim', 'vancomycin']
    notes_check = pd.DataFrame()
    for n, df in notes.groupby('row_id'):
        # using list comprehension to check if string contains list element
        res = [ele for ele in antibitics_list if (ele in df['clean_tx'].values[0])]
        if len(res) >= 1:
            # print(len(res))
            # print(df['clean_tx'].values[0])
            data = pd.DataFrame({'row_id': [n], 'hadm_id': [df['hadm_id'].values[0]], 'antibiotic_yn': [1]})
            notes_check = notes_check.append(data, ignore_index=True)
    # Group on hadm_id
    notes_check = notes_check.groupby(['hadm_id']).agg({'antibiotic_yn': 'max'}).reset_index()
    print('Anibiotic in notes: ', notes_check.shape)
    print('--------------------------------------------------------------')
    return notes_check


def load_notes(view_name_all_pts_within_observation_window):
    notes = create_dataset.noteevents(view_name_all_pts_within_observation_window)
    # Clean notes
    notes = clean_text(df=notes, text_field='text')
    return notes


def clean_text(df, text_field='text'):
    """
    Preparing patient notes for processing
    """
    df['clean_tx'] = df['text'].str.lower()
    df['clean_tx'] = df['clean_tx'].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    df['clean_tx'] = df['clean_tx'].apply(lambda elem: re.sub(r"\d+", "", elem))

    return df


def load_intubation_procedures(view_name_all_pts_within_observation_window):
    intubation = create_dataset.intubation_cpt(view_name_all_pts_within_observation_window)
    intubation['intubation'] = 1  # intubation indicator column
    # Group on hand_id & drop cpt code and date columns
    intubation = intubation.drop_duplicates(subset=['hadm_id'], keep='first')
    intubation = intubation.drop(columns=['chartdate', 'cpt_cd'])
    print('Intubation records: ', intubation.shape)
    print('--------------------------------------------------------------')
    return intubation



