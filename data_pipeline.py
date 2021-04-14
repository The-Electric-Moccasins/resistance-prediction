from dataproc.cohort import query_esbl_pts, remove_dups, observation_window, query_all_pts
from dataproc.sampling import generate_samples
from dataproc.create_dataset import dataset_creation
from hyper_params import HyperParams

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

params = HyperParams()
# cohort_query =  query_esbl_pts
cohort_query =  query_all_pts
RECALC_LOINC_VALS = False
DATA_DIR = 'data_ae_train'  # 'data'
data_file_name = f'{DATA_DIR}/fulldata.npy'


# Patients cohort:



def main():
    esbl_admits = cohort_query()
    # Remove dups
    esbl_admits = remove_dups(esbl_admits)
    # Create observation window
    esbl_admits_window = observation_window(esbl_admits, window_size=params.observation_window_hours)
    # Subset columns
    pts_labels = esbl_admits_window[['hadm_id', 'index_date','RESISTANT_YN']]
    pts_labels.to_pickle(f'{DATA_DIR}/patient_labels.pkl')

    # Import cohort/labels data from the .pkl file:

    # pts_labels = pd.read_pickle(f'{DATA_DIR}/patient_labels.pkl')
    print(pts_labels.shape)

    # Patient's features data:

    # Loading the features
    features = dataset_creation(pts_labels['hadm_id'], params.observation_window_hours)
    features = features.merge(pts_labels[['hadm_id','RESISTANT_YN']], on='hadm_id')
    features.to_pickle(f'{DATA_DIR}/features.pkl')

    # features = pd.read_pickle(f'{DATA_DIR}/features.pkl')


    loinc_codes = list(features.drop(columns=['hadm_id', 'subject_id', 'admittime','admission_type']).columns)[:-8]
    # print(list(loinc_codes))

    features_summary = features[loinc_codes].describe()

    if RECALC_LOINC_VALS:
        from dataproc.embeddings import loinc_values

        loinc_vals = loinc_values(loinc_codes)
        loinc_vals.to_pickle(f'{DATA_DIR}/loinc_vals_raw.pic')
        loinc_vals.dropna(subset=['value'], inplace=True)
        loinc_vals = loinc_vals.astype({'value': 'string', 'loinc_code': 'category'})
        loinc_vals['value'] = loinc_vals['value'].map(lambda x: x.lstrip('LESS THAN '))
        loinc_vals['value'] = loinc_vals['value'].map(lambda x: x.lstrip('GREATER THAN '))
        loinc_vals['value'] = loinc_vals['value'].map(lambda x: x.lstrip('>GREATER THAN '))
        loinc_vals['value'] = loinc_vals['value'].map(lambda x: x.lstrip('<LESS THAN '))
        loinc_vals['value'] = loinc_vals['value'].map(lambda x: x.rstrip(' NG/ML'))
        loinc_vals['value'] = loinc_vals['value'].map(lambda x: x.lstrip('<>'))
        loinc_vals['value'] = loinc_vals['value'].map(lambda x: x.replace(',', '.'))
        loinc_vals.to_pickle(f'{DATA_DIR}/loinc_vals_str_clean.pic')

        loinc_vals.drop(list(loinc_vals.loc[loinc_vals['value'] == 'UNABLE TO ANALYZE'].index),  inplace=True)
        loinc_vals.drop(list(loinc_vals.loc[loinc_vals['value'] == 'MOLYSIS FALSELY DECREASES THIS RESULT'].index),  inplace=True)
        loinc_vals.drop(list(loinc_vals.loc[loinc_vals['value'] == 'COMPUTER NETWORK FAILURE. TEST NOT RESULTED.'].index),  inplace=True)
        loinc_vals.drop(list(loinc_vals.loc[loinc_vals['value'] == 'UNABLE TO DETERMINE'].index),  inplace=True)
        loinc_vals.drop(list(loinc_vals.loc[loinc_vals['value'] == ':UNABLE TO DETERMINE'].index),  inplace=True)
        loinc_vals.drop(list(loinc_vals.loc[loinc_vals['value'] == 'UNABLE TO QUANTITATE'].index),  inplace=True)
        loinc_vals.drop(list(loinc_vals.loc[loinc_vals['value'] == 'UNABLE TO REPORT'].index),  inplace=True)
        loinc_vals.to_pickle(f'{DATA_DIR}/loinc_vals.pic')
    else:
        loinc_vals = pd.read_pickle(f'{DATA_DIR}/loinc_vals.pic')

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


    # remove lab column that contains only 'inf' and 'Nan'
    numeric.remove('26498-6')
    # remove lab column that contains phrase 'See comments'
    categorical.remove('33914-3')
    # remove lab column that contains phrase 'Random'
    categorical.remove('13362-9')


    print('All:', len(loinc_codes))
    print('Numeric: ', len(numeric))
    print('Categorical: ', len(categorical))
    print('Weird:', len(weird))


    numeric_stats = []
    for code in numeric:
        a = pd.to_numeric(loinc_vals.loc[loinc_vals['loinc_code'] == str(code), 'value'], errors='coerce').describe()
        numeric_stats.append(a)
    numeric_stats_df = pd.concat(numeric_stats, axis=1, keys=numeric)


    dataset = features.drop(columns=weird, errors='ignore')


    # Convert to numeric selected columns
    dataset[numeric] = dataset[numeric].apply(pd.to_numeric, errors='coerce', axis=1)

    # Since many lab data have outliers the median and interquartile range can be used to standardizing the numeric variables:
    # - value = (value – median) / (p75 – p25)

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
        return df

    dataset = stanardize_numeric_values(dataset, numeric, numeric_stats_df)


    # Imputation of missing values using scikit-learn https://scikit-learn.org/stable/modules/impute.html#impute

    from sklearn.impute import SimpleImputer

    def replace_missing_val(df, list_of_clms, how='median'):
        """
        Imputation of missing values using median
        """
        imp = SimpleImputer(strategy=how)
        df_prc = imp.fit_transform(df[list_of_clms])
        #df[list_of_clms] = pd.DataFrame(df_prc, columns=list_of_clms)
        return df_prc


    numlabvars = replace_missing_val(dataset, numeric, how='median')

    # Clean lab categorical variables:

    dataset['30089-7'] = np.where(dataset['30089-7'].isin(['<1','1','2']), '0-2',
                         np.where(dataset['30089-7'].isin(['3','4']),'3-5', dataset['30089-7']))

    dataset['5767-9'] = np.where(dataset['5767-9'].isin(['CLEAR']), 'Clear',
                        np.where(dataset['5767-9'].isin(['SLHAZY']), 'SlHazy',
                        np.where(dataset['5767-9'].isin(['HAZY']), 'Hazy',
                        np.where(dataset['5767-9'].isin(['SlCloudy']),'SlCldy',
                        np.where(dataset['5767-9'].isin(['CLOUDY']),'Cloudy',dataset['5767-9'])))))

    dataset['5769-5'] = np.where(dataset['5769-5'].isin(['0']), 'NEG',
                        np.where(dataset['5769-5'].isin(['NOTDONE']), 'NONE',
                        np.where(dataset['5769-5'].isin(['LRG']), 'MANY', dataset['5769-5'])))

    dataset['5778-6'] = np.where(dataset['5778-6'].isin(['YELLOW','YEL']), 'Yellow',
                        np.where(dataset['5778-6'].isin(['STRAW']), 'Straw',
                        np.where(dataset['5778-6'].isin(['AMBER','AMB']), 'Amber',
                        np.where(dataset['5778-6'].isin(['RED']), 'Red',
                        np.where(dataset['5778-6'].isin(['ORANGE']), 'Orange',
                        np.where(dataset['5778-6'].isin(['DKAMB','DKAMBER']), 'DkAmb',
                        np.where(dataset['5778-6'].isin([' ']), np.nan, dataset['5778-6'])))))))

    dataset['5797-6'] = np.where(dataset['5797-6'].isin(['>80']), '80',dataset['5797-6'])

    dataset['5804-0'] = np.where(dataset['5804-0'].isin(['>300']), '300',
                        np.where(dataset['5804-0'].isin([' ']), np.nan, dataset['5804-0']))

    dataset['5818-0'] = np.where(dataset['5818-0'].isin(['.2']), '0.2',
                        np.where(dataset['5818-0'].isin(['>8','>8.0']), '8',
                        np.where(dataset['5818-0'].isin(['>12']), '12',
                        np.where(dataset['5818-0'].isin(['NotDone']), np.nan, dataset['5818-0']))))

    dataset['5822-2'] = np.where(dataset['5822-2'].isin(['0', 'N']), 'NONE',
                        np.where(dataset['5822-2'].isin(['NOTDONE']), np.nan, dataset['5822-2']))

    dataset['778-1'] = np.where(dataset['778-1'].isin(['UNABLE TO ESTIMATE DUE TO PLATELET CLUMPS']), 'NOTDETECTED', dataset['778-1'])


    dataset.update(dataset[categorical].fillna('UNKNOWN'))

    # Use one hot encoder for categoric lab features:

    enc = OneHotEncoder()
    enc.fit(dataset[categorical])
    enc.categories_[0:4]

    onehotlabvars = enc.transform(dataset[categorical]).toarray()

    # Clean demographic static variables:

    staticvars = ['admission_type', 'admission_location', 'insurance', 'language',
                   'religion', 'marital_status', 'ethnicity', 'gender']

    dataset['admission_location'] = \
    np.where(dataset['admission_location'].isin(['** INFO NOT AVAILABLE **']), 'EMERGENCY ROOM ADMIT',
    np.where(dataset['admission_location'].isin(['TRANSFER FROM SKILLED NUR','TRANSFER FROM OTHER HEALT',
                            'TRANSFER FROM HOSP/EXTRAM']), 'TRANSFER FROM MED FACILITY',dataset['admission_location']))
    dataset['language'] = \
    np.where(~dataset['language'].isin(['ENGL','SPAN']),'OTHER',dataset['language'])

    dataset['religion'] = \
    np.where(~dataset['religion'].isin(['CATHOLIC','NOT SPECIFIED','UNOBTAINABLE','PROTESTANT QUAKER','JEWISH']),'OTHER',
    np.where(dataset['religion'].isin(['UNOBTAINABLE']),'NOT SPECIFIED', dataset['religion'] ))

    dataset['ethnicity'] = \
    np.where(dataset['ethnicity'].isin(['ASIAN - CHINESE',
                                        'ASIAN - ASIAN INDIAN',
                                        'ASIAN - VIETNAMESE',
                                        'ASIAN - OTHER',
                                        'ASIAN - FILIPINO',
                                        'ASIAN - CAMBODIAN']), 'ASIAN',
    np.where(dataset['ethnicity'].isin(['WHITE - RUSSIAN',
                                        'WHITE - BRAZILIAN',
                                        'WHITE - OTHER EUROPEAN']),'WHITE',
    np.where(dataset['ethnicity'].isin(['BLACK/CAPE VERDEAN',
                                        'BLACK/HAITIAN',
                                        'BLACK/AFRICAN']), 'BLACK/AFRICAN AMERICAN',
    np.where(dataset['ethnicity'].isin(['HISPANIC/LATINO - PUERTO RICAN',
                                       'HISPANIC/LATINO - DOMINICAN',
                                       'HISPANIC/LATINO - SALVADORAN',
                                       'HISPANIC/LATINO - CUBAN',
                                       'HISPANIC/LATINO - MEXICAN']), 'HISPANIC OR LATINO',
    np.where(dataset['ethnicity'].isin(['MULTI RACE ETHNICITY',
                                        'MIDDLE EASTERN',
                                        'PORTUGUESE',
                                        'AMERICAN INDIAN/ALASKA NATIVE',
                                        'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
                                        'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE']), 'OTHER',
    np.where(dataset['ethnicity'].isin(['UNABLE TO OBTAIN',
                                        'PATIENT DECLINED TO ANSWER']), 'UNKNOWN/NOT SPECIFIED',
    dataset['ethnicity']))))))

    dataset['marital_status'] = dataset['marital_status'].fillna(value='UNKNOWN')


    # Use one hot encoder for demographic features:

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(dataset[staticvars])
    enc.categories_

    onehotstaticvars = enc.transform(dataset[staticvars]).toarray()

    # Combine all features and constract full dataset

    response = np.array([dataset['RESISTANT_YN']])
    response = response.T
    response.shape

    fulldata = np.concatenate((numlabvars, onehotlabvars, onehotstaticvars, response), axis=1)
    fulldata.shape


    np.save(data_file_name, fulldata)