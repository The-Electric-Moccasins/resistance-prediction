import torch
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_validate, learning_curve
from torch.utils.data import Dataset
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from hyper_params import HyperParams

params = HyperParams()


from embeddings.dataloader import TheDataSet
from embeddings.autoencoder import Autoencoder # so we can load this model

USE_AUTOENCODER = False


def get_autoencoder():
    autoencoder = None
    with open('data/autoencoder.pic', 'rb') as f:
        autoencoder = torch.load(f)
    return autoencoder


def load_data():
#     dataset = TheDataSet(datafile='data/fulldata.npy', pad_to_360=False)
    dataset = TheDataSet(datafile='data/labdata.npy', pad_to_360=False)
#     dataset = TheDataSet(datafile='data/fulldata_initial.npy', pad_to_360=False)
    data_loader = torch.utils.data.DataLoader(dataset)
    if USE_AUTOENCODER:
        print("Using AutoEncoder")
        autoencoder = get_autoencoder()
        get_x = lambda x: autoencoder.encoder(x.float()).detach().numpy()
    else:
        print("Using Plain Embeddings")
        get_x = lambda x: x.detach().numpy()
    alldata = [(get_x(r1), r2.detach().numpy()) for r1, r2 in data_loader]
    X, y = list(zip(*alldata))
    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y


def print_accuracy(cv_scores):
    # print("Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
    for score_name, scores in cv_scores.items():
        print("%s: %0.2f (+/- %0.2f)" % (score_name, scores.mean(), scores.std() * 2))




X, y = load_data()

print(y.dtype)

import pandas as pd
df_X = pd.DataFrame(X)
df_y = pd.DataFrame(y, columns=['y'])
df_data = pd.concat([df_X, df_y], axis=1)

from dataproc import sampling
params.validation_set_fraction=0.29
params.test_set_fraction=0.01
params.negative_to_positive_ratio=2
print(params.__dict__)
# train_set, df_validation, df_test = sampling.generate_samples(df_dataset=df_data,
#                           negative_to_positive_ratio=params.negative_to_positive_ratio,
#                           test_set_fraction=params.test_set_fraction,
#                           validation_set_fraction=params.validation_set_fraction,
#                           random_state=params.random_state,
#                           by_column='y')
X_train, X_validate, y_train, y_validate = train_test_split(df_X, df_y, stratify=df_y, test_size=0.29, random_state=params.random_state)
print(f"Train count: {Counter(y_train['y'])}")
print(f"Validate count: {Counter(y_validate['y'])}")
# X_train = train_set.drop(columns=['y'])
# y_train = train_set['y'].astype('int')
# X_validate = df_validation.drop(columns=['y'])
# y_validate = df_validation['y'].astype('int')
print(np.mean(y_validate['y']))
print(f"Train shape: {X_train.shape}")
print(f"Validate shape: {X_validate.shape}")


def train_random_forest():
    if USE_AUTOENCODER:
        rf_params = dict(                          class_weight='balanced_subsample',
                                                   n_estimators=133,
                                                   max_depth=4,
                                                   max_leaf_nodes=70,
                                                   max_features=0.8,
                                                   #max_samples=0.9,
                                                   #min_samples_leaf=10,
                                                   #min_samples_split=15,
                                                   n_jobs=2
                        )
    else:
        rf_params = dict(n_estimators=100,
                               max_depth=10,
                               max_leaf_nodes=90,
                               max_features=20,
                               max_samples=0.9,
                               min_samples_leaf=5,
                               min_samples_split=10,
                                 random_state=7,
                                n_jobs=2)

    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)
    return rf


def train_xgboost():
    import xgboost as xgb

    param_dist = dict(objective='binary:logistic',
                      n_estimators=100, # 170,
                      eval_metric='rmsle', # 'logloss',
                      max_depth=4,
                      eta=0.3,
                      booster='gbtree',
                      n_jobs=4,
                      # subsample=0.8,
                      # colsample_bynode=0.5
                    )

    xgboost_cls = xgb.XGBClassifier(**param_dist)
    xgboost_cls.fit(X_train, y_train)
    return xgboost_cls


model = train_random_forest()
# model = train_xgboost()
y_validate_hat = model.predict(X_validate)
print(f"predictions mean: {np.mean(y_validate_hat)}")
simple_score = model.score(X_validate, y_validate)
print(f"simple_score: {simple_score}")
# scoring = {'AUC': 'roc_auc', 'Accuracy': 'accuracy', 'Precision': 'precision', 'Recall': 'recall'}

scoring = ['roc_auc','accuracy','precision', 'recall', 'f1']
cv_scores = cross_validate(model, X_train, y_train['y'], scoring=scoring)
print_accuracy(cv_scores)
print(classification_report(y_validate, y_validate_hat))
