import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def generate_samples(df_dataset: DataFrame, negative_to_positive_ratio: float, test_set_fraction : float, validation_set_fraction: float, random_state: int, by_column:str = 'RESISTANT_YN'):
    """
    Convert a dataset to train/validation/test samples. The train dataset can have a specified negative/positive ratio.

    1. random sample test and validation set (we do this first so their distribution is not hurt by our manipulations in the training set sampling)
    2. construct train set by pos/neg ration sepcification


    :param df_dataset:
    :param negative_to_positive_ratio:
    :param test_set_fraction:
    :param validation_set_fraction:
    :return:
    """

    """
    """
    # compute test/val size in advance
    df_train, df_validation, df_test = split_train_val_test(df_dataset, test_set_fraction, validation_set_fraction, random_state)

    # stratify the train set
    train_set = stratify_set(df_train, negative_to_positive_ratio, by_column=by_column)

    return train_set, df_validation, df_test


def stratify_set(df: DataFrame, negative_to_positive_ratio: float, by_column:str = 'RESISTANT_YN'):
    n = df.shape[0]
    num_positives = np.sum(df[by_column])
    num_negatives = n - num_positives
    target_num_positives = int(np.floor(n / (negative_to_positive_ratio + 1)))
    # if we do not have enough positives for the max train set size, use as much as available
    target_num_positives = int(min(target_num_positives, num_positives))
    target_num_negatives = int(np.floor(negative_to_positive_ratio * target_num_positives))
    df_train_positives = df[df[by_column] == 1.0].sample(n=target_num_positives)
    upsample_negatives = target_num_negatives > num_negatives

    df_train_negatives = df[df[by_column] == 0.0].sample(n=target_num_negatives, replace=upsample_negatives)
    df = pd.concat([df_train_positives, df_train_negatives])
    return df


def split_train_val_test(df_dataset, test_set_fraction, validation_set_fraction, random_state):
    n = df_dataset.shape[0]
    test_set_size = int(test_set_fraction * n)
    validation_set_size = int(validation_set_fraction * n)
    df_train_validation, df_test = train_test_split(df_dataset, test_size=test_set_size, random_state=random_state)
    df_train, df_validation = train_test_split(df_train_validation, test_size=validation_set_size,
                                               random_state=random_state)
    return df_train, df_validation, df_test

# select that much negatives
