import pandas as pd
from pandas import DataFrame
from sklearn.impute import SimpleImputer


def drop_sparse_columns(df, columns:list, max_sparsity_to_keep=0.9):
    # count the missing values in each column
    missing_values_counts = df[columns].isna().sum()
    # percent of columns with rare values
    max_missing_vals = max_sparsity_to_keep * df.shape[0]
    sparse_columns = missing_values_counts[missing_values_counts > max_missing_vals].index.tolist()
    # drop the rare labtests from the dataset
    df = df.drop(columns=sparse_columns)
    return df


def stanardize_numeric_values(df, list_of_clms=None):
    """
    Use the median and interquartile range to
    standardize the numeric variables
    value = (value – median) / (p1 – p99)
    """
    if list_of_clms is None:
        list_of_clms = df.columns.tolist()
    df_stats = df[list_of_clms].describe(percentiles=[.01, .25, .5, .75, .95, .99])
    list_of_clms = df_stats.columns.tolist()

    for code in list_of_clms:
        median = df_stats[code]['50%']
        p_low = df_stats[code]['1%']
        p_high = df_stats[code]['99%']
        df[code] = (df[code] - median) / (p_low - p_high)
    return df


def replace_missing_val(df, list_of_clms, how='median'):
    """
    Imputation of missing values using median
    """
    temp_df = df[list_of_clms]
    imp = SimpleImputer(strategy=how)
    df_prc = imp.fit_transform(temp_df)
    temp_df = pd.DataFrame(df_prc, columns=list_of_clms, index=df.index)
    return temp_df


def bin_numerics(dataset: DataFrame, numeric_columns:list, bins=6):
        df = dataset.copy(deep=True)
        numeric_columns = [col for col in numeric_columns if df[col].nunique() > 0]
        for col in numeric_columns:
            df[col] = pd.cut(df[col], bins=min(df[col].nunique(), 6), labels=False).fillna(0)
            df[col] = df[col].apply(lambda x: str(x))
            df[col] = df[col].astype('O')
#         df[numeric_columns].fillna(value=-1, inplace=True)
#         for col in numeric_columns:
#                 df[col] = df[col].astype('str')

        return df