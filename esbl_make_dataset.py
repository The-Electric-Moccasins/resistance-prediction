# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 13:55:59 2021

@author: tatia
"""

from dataproc.cohort import query_esbl_pts, remove_dups, observation_window
from dataproc.sampling import generate_samples
from dataproc.sampling import stratify_set
from dataproc.roc_auc_curves import plt_roc_auc_curve, plt_precision_recall_curve
from dataproc.create_dataset import dataset_creation
from dataproc.create_dataset import prescriptions
from dataproc.create_dataset import previous_admissions
from dataproc.create_dataset import open_wounds_diags
from dataproc.embeddings import loinc_values
from hyper_params import HyperParams
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# load hyperparams instance
params = HyperParams()



def cohort_creation(observation_window_hours):
    # Select esbl microbiology test
    esbl_admits = query_esbl_pts()
    # Remove dups
    esbl_admits = remove_dups(esbl_admits)
    # Create observation window
    esbl_admits_window = observation_window(esbl_admits, window_size=observation_window_hours)
    # Subset columns
    pts_labels = esbl_admits_window[['hadm_id', 'index_date','RESISTANT_YN']]    
    
    return pts_labels


def loinc_values_proc(loinc_codes):
    loinc_vals = loinc_values(loinc_codes)
    loinc_vals.dropna(subset=['value'], inplace=True)
    loinc_vals = loinc_vals.astype({'value': 'string', 'loinc_code': 'category'})
    loinc_vals['value'] = loinc_vals['value'].map(lambda x: x.lstrip('LESS THAN '))
    loinc_vals['value'] = loinc_vals['value'].map(lambda x: x.lstrip('GREATER THAN '))
    loinc_vals['value'] = loinc_vals['value'].map(lambda x: x.lstrip('>GREATER THAN '))
    loinc_vals['value'] = loinc_vals['value'].map(lambda x: x.lstrip('<LESS THAN '))
    loinc_vals['value'] = loinc_vals['value'].map(lambda x: x.rstrip(' NG/ML'))
    loinc_vals['value'] = loinc_vals['value'].map(lambda x: x.lstrip('<>'))
    loinc_vals['value'] = loinc_vals['value'].map(lambda x: x.replace(',', '.'))
    loinc_vals.drop(list(loinc_vals.loc[loinc_vals['value'] == 'UNABLE TO ANALYZE'].index),  inplace=True)
    loinc_vals.drop(list(loinc_vals.loc[loinc_vals['value'] == 'MOLYSIS FALSELY DECREASES THIS RESULT'].index),  inplace=True)
    loinc_vals.drop(list(loinc_vals.loc[loinc_vals['value'] == 'COMPUTER NETWORK FAILURE. TEST NOT RESULTED.'].index),  inplace=True)
    loinc_vals.drop(list(loinc_vals.loc[loinc_vals['value'] == 'UNABLE TO DETERMINE'].index),  inplace=True)
    loinc_vals.drop(list(loinc_vals.loc[loinc_vals['value'] == ':UNABLE TO DETERMINE'].index),  inplace=True)
    loinc_vals.drop(list(loinc_vals.loc[loinc_vals['value'] == 'UNABLE TO QUANTITATE'].index),  inplace=True)
    loinc_vals.drop(list(loinc_vals.loc[loinc_vals['value'] == 'UNABLE TO REPORT'].index),  inplace=True)
    
    return loinc_vals

def lab_records_categories(loinc_vals):
    numeric = []
    categorical = []
    weird = []
    for code in loinc_codes:
        size = len(loinc_vals.loc[loinc_vals['loinc_code'] == str(code), 'value'])
        size_unique = len(loinc_vals.loc[loinc_vals['loinc_code'] == str(code), 'value'].unique())
        sum_na = pd.to_numeric(loinc_vals.loc[loinc_vals['loinc_code'] == str(code), 'value'], errors='coerce').isna().sum()
        if sum_na / size < 0.05:
            numeric.append(code)
        elif sum_na / size > 0.05 and size_unique < 100:
            categorical.append(code)
        else:
            weird.append(code)
        
    # Remove columns that are not useful:
    # remove lab column that contains only 'inf' and 'Nan'
    numeric.remove('26498-6')
    # remove lab column that only contains phrase 'See comments'
    categorical.remove('33914-3')
    # remove lab column that contains phrase 'Random'
    categorical.remove('13362-9')
                    
    return numeric, categorical, weird

 
def sum_stats_numeric_labs(loinc_vals, numeric):
    numeric_stats = []
    for code in numeric:
        a = pd.to_numeric(loinc_vals.loc[loinc_vals['loinc_code'] == str(code), 'value'], errors='coerce').describe()
        numeric_stats.append(a)
    numeric_stats_df = pd.concat(numeric_stats, axis=1, keys=numeric)    
    return numeric_stats_df


def stanardize_numeric_values(df, list_of_clms, ref_df):
    """
    Use the median and interquartile range to 
    standardize the numeric variables
    value = (value – median) / (p75 – p25)
    """
    for code in list_of_clms:
        median = ref_df[code]['50%']
        p25 = ref_df[code]['25%']
        p75 = ref_df[code]['75%']
        df[code] = (df[code] - median) / (p75 - p25)
    # Subset relevant columns
    columns = ['hadm_id'] + list_of_clms
    df = df[columns].copy()
    return df

def replace_missing_val(df, list_of_clms, how='median'):
    """
    Imputation of missing values using median
    """
    imp = SimpleImputer(strategy=how)
    df_prc = imp.fit_transform(df[list_of_clms])
    df_prc = pd.DataFrame(df_prc, columns=list_of_clms)
    # Concat hadm_id and df_prc
    out = pd.concat([df['hadm_id'], df_prc], axis = 1)
    return out

def clean_categoric_lab_records(df, categorical):
    df['30089-7'] = np.where(df['30089-7'].isin(['<1','1','2']), '0-2',
                    np.where(df['30089-7'].isin(['3','4']),'3-5', df['30089-7']))

    df['5767-9'] = np.where(df['5767-9'].isin(['CLEAR']), 'Clear',
                   np.where(df['5767-9'].isin(['SLHAZY']), 'SlHazy',
                   np.where(df['5767-9'].isin(['HAZY']), 'Hazy',
                   np.where(df['5767-9'].isin(['SlCloudy']),'SlCldy',  
                   np.where(df['5767-9'].isin(['CLOUDY']),'Cloudy',df['5767-9'])))))
    
    df['5769-5'] = np.where(df['5769-5'].isin(['0']), 'NEG',
                   np.where(df['5769-5'].isin(['NOTDONE']), 'NONE',
                   np.where(df['5769-5'].isin(['LRG']), 'MANY', df['5769-5'])))
    
    df['5778-6'] = np.where(df['5778-6'].isin(['YELLOW','YEL']), 'Yellow',
                   np.where(df['5778-6'].isin(['STRAW']), 'Straw',
                   np.where(df['5778-6'].isin(['AMBER','AMB']), 'Amber', 
                   np.where(df['5778-6'].isin(['RED']), 'Red', 
                   np.where(df['5778-6'].isin(['ORANGE']), 'Orange', 
                   np.where(df['5778-6'].isin(['DKAMB','DKAMBER']), 'DkAmb', 
                   np.where(df['5778-6'].isin([' ']), np.nan, df['5778-6'])))))))
    
    df['5797-6'] = np.where(df['5797-6'].isin(['>80']), '80',df['5797-6'])
    
    df['5804-0'] = np.where(df['5804-0'].isin(['>300']), '300',
                   np.where(df['5804-0'].isin([' ']), np.nan, df['5804-0']))
    
    df['5818-0'] = np.where(df['5818-0'].isin(['.2']), '0.2',
                   np.where(df['5818-0'].isin(['>8','>8.0']), '8',
                   np.where(df['5818-0'].isin(['>12']), '12',
                   np.where(df['5818-0'].isin(['NotDone']), np.nan, df['5818-0']))))
    
    df['5822-2'] = np.where(df['5822-2'].isin(['0', 'N']), 'NONE',
                   np.where(df['5822-2'].isin(['NOTDONE']), np.nan, df['5822-2']))

    df['778-1'] = np.where(df['778-1'].isin(['UNABLE TO ESTIMATE DUE TO PLATELET CLUMPS']), 'NOTDETECTED', df['778-1'])
    # Subset columns
    columns = ['hadm_id'] + categorical
    df = df[columns].copy()
    
    return df

def clean_static_demog_vars(df, staticvars):
    df['admission_location'] = \
        np.where(df['admission_location'].isin(['** INFO NOT AVAILABLE **']), 'EMERGENCY ROOM ADMIT',
        np.where(df['admission_location'].isin(['TRANSFER FROM SKILLED NUR','TRANSFER FROM OTHER HEALT',
                                'TRANSFER FROM HOSP/EXTRAM']), 'TRANSFER FROM MED FACILITY',df['admission_location']))
    df['language'] = \
        np.where(~df['language'].isin(['ENGL','SPAN']),'OTHER',df['language'])
    
    df['religion'] = \
        np.where(~df['religion'].isin(['CATHOLIC','NOT SPECIFIED','UNOBTAINABLE','PROTESTANT QUAKER','JEWISH']),'OTHER',
        np.where(df['religion'].isin(['UNOBTAINABLE']),'NOT SPECIFIED', df['religion'] ))
    
    df['ethnicity'] = \
        np.where(df['ethnicity'].isin(['ASIAN - CHINESE',
                                            'ASIAN - ASIAN INDIAN',
                                            'ASIAN - VIETNAMESE',
                                            'ASIAN - OTHER',
                                            'ASIAN - FILIPINO',
                                            'ASIAN - CAMBODIAN']), 'ASIAN',
        np.where(df['ethnicity'].isin(['WHITE - RUSSIAN',
                                            'WHITE - BRAZILIAN',
                                            'WHITE - OTHER EUROPEAN']),'WHITE',
        np.where(df['ethnicity'].isin(['BLACK/CAPE VERDEAN',
                                            'BLACK/HAITIAN',
                                            'BLACK/AFRICAN']), 'BLACK/AFRICAN AMERICAN',
        np.where(df['ethnicity'].isin(['HISPANIC/LATINO - PUERTO RICAN',
                                           'HISPANIC/LATINO - DOMINICAN',
                                           'HISPANIC/LATINO - SALVADORAN',
                                           'HISPANIC/LATINO - CUBAN',
                                           'HISPANIC/LATINO - MEXICAN']), 'HISPANIC OR LATINO',   
        np.where(df['ethnicity'].isin(['MULTI RACE ETHNICITY',
                                            'MIDDLE EASTERN',
                                            'PORTUGUESE',
                                            'AMERICAN INDIAN/ALASKA NATIVE',
                                            'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
                                            'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE']), 'OTHER',
        np.where(df['ethnicity'].isin(['UNABLE TO OBTAIN',
                                            'PATIENT DECLINED TO ANSWER']), 'UNKNOWN/NOT SPECIFIED',
    df['ethnicity']))))))
    # Subset columns
    columns = ['hadm_id'] + staticvars
    df = df[columns].copy()
        
    return df
   


if __name__ == '__main__':
        # ESBL Cohort:
        pts_labels = cohort_creation(params.observation_window_hours)
        print('ESBL COHORT:')
        print(pts_labels['hadm_id'].nunique())
        print(pts_labels['RESISTANT_YN'].value_counts())
        print('--------------------------------------------------------------')
        
        # Antibiotics prescriptions:
        rx = prescriptions(pts_labels['hadm_id'], params.observation_window_hours)
        rx['value'] = 1
        rx = rx[['hadm_id','drug','value']]        
        # Drop duplicated records
        rx = rx.drop_duplicates(subset=['hadm_id','drug'], keep='first')
        # One-hot encoder for prescriptions
        onehotrx_df = rx.pivot_table(index='hadm_id', columns='drug', values='value', fill_value=0).reset_index()
        print('RX records: ', onehotrx_df.shape)
        print('--------------------------------------------------------------')
        
        # Previous admissions:
        admits =  previous_admissions(pts_labels['hadm_id'])
        admits_df = admits.groupby('hadm_id').agg({'prev_hadm_id':'nunique'}).reset_index()
        admits_df = admits_df.rename(columns={'prev_hadm_id':'n_admits'})
        print('Previous admits: ', admits_df.shape)
        print('--------------------------------------------------------------')
        
        # Open Wounds Diagnosis:
        wounds = open_wounds_diags(pts_labels['hadm_id'])
        wounds['wounds'] = 1 # wounds indicator column
        # Group on hand_id & drop icd9 code column
        wounds_df = wounds.drop_duplicates(subset=['hadm_id'], keep = 'first')
        wounds_df = wounds_df.drop(columns='icd9_code')
        print('Open wounds: ', wounds_df.shape)
        print('--------------------------------------------------------------')    
        
        # Lab and Static data features:
        features = dataset_creation(pts_labels['hadm_id'], params.observation_window_hours)
        features = features.merge(pts_labels[['hadm_id','RESISTANT_YN']], on='hadm_id')
        features['hadm_id'] = features['hadm_id'].astype(int)
        print('Features shape: ', features.shape)
        # Loinc codes
        loinc_codes = list(features.drop(columns=['hadm_id', 'subject_id', 'admittime','admission_type']).columns)[:-8]
        loinc_vals = loinc_values_proc(loinc_codes)
        # Split lab values into numeric and categorical 
        numeric, categorical, weird = lab_records_categories(loinc_vals)
        print('Numeric vars: ', len(numeric))
        print('Categorical vars: ', len(categorical))
        features = features.drop(columns=weird, errors='ignore')
        # Summary statistics for numeric lab codes:
        numeric_stats_df = sum_stats_numeric_labs(loinc_vals, numeric)  
        print('--------------------------------------------------------------')  
        
        # Stanardize numeric lab values:
        features[numeric] = features[numeric].apply(pd.to_numeric, errors='coerce', axis=1)
        numlabvars_df = stanardize_numeric_values(features, numeric, numeric_stats_df)
        numlabvars_df = replace_missing_val(numlabvars_df, numeric, how='median')
        print('Stanardize numeric values: ', numlabvars_df.shape)
        
        # Clean categorical lab values:
        catlabvaes_df = clean_categoric_lab_records(features, categorical)
        # replace 'Nan' values in categorical variables by 'UNKNOWN'
        catlabvaes_df.update(catlabvaes_df.fillna('UNKNOWN'))
        # One-hot encoder for categorical vars
        onehotlabvars_df = pd.get_dummies(catlabvaes_df)
        # To reduce the correlation among variables, remove one feature column from the one-hot encoded array:
        col_list = list(onehotlabvars_df.filter(regex='_UNKNOWN'))
        onehotlabvars_df = onehotlabvars_df[onehotlabvars_df.columns.drop(col_list)]
        print('One-Hot categorical values: ', onehotlabvars_df.shape)
        
        # Clean demographic static values:
        staticvars = ['admission_type', 'admission_location', 'insurance', 'language', 
               'religion', 'marital_status', 'ethnicity', 'gender']
        staticvars_df = clean_static_demog_vars(features, staticvars)
        # One-hot encoder for categorical vars
        onehotstaticvars_df = pd.get_dummies(staticvars_df)
        # To reduce the correlation among variables, remove one feature column from the one-hot encoded array:
        col_list = ['admission_type_URGENT', 'admission_location_TRANSFER FROM MED FACILITY', 
            'insurance_Self Pay', 'language_OTHER', 'religion_NOT SPECIFIED', 'marital_status_UNKNOWN (DEFAULT)',
            'ethnicity_UNKNOWN/NOT SPECIFIED', 'gender_M']
        onehotstaticvars_df = onehotstaticvars_df[onehotstaticvars_df.columns.drop(col_list)]
        print('One-Hot static values: ', onehotstaticvars_df.shape)
        print('--------------------------------------------------------------')  
        
        # Combine all variables into one dataset:
        fulldata = pd.merge(numlabvars_df, onehotlabvars_df, on='hadm_id', how='left')
        fulldata = pd.merge(fulldata, onehotstaticvars_df, on='hadm_id', how='left')
        fulldata = pd.merge(fulldata, onehotrx_df, on='hadm_id', how='left')
        fulldata = pd.merge(fulldata, admits_df, on='hadm_id', how='left')
        fulldata = pd.merge(fulldata, wounds_df, on='hadm_id', how='left')
        fulldata = pd.merge(fulldata, pts_labels[['RESISTANT_YN', 'hadm_id']], on='hadm_id', how='left')
        fulldata.fillna(0, inplace=True)
        fulldata.drop(columns=['hadm_id'], inplace=True)
        print('Full Datset Size: ', fulldata.shape)
        
        # Save to a file  
        fulldata_np = fulldata.to_numpy()
        np.save('data/fulldata_extra.npy', fulldata)
        fulldata.to_csv('data/fulldata_extra.csv', sep=',', index=False)
    