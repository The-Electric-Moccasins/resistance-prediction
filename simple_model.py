import torch
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, cross_validate, learning_curve
from torch.utils.data import Dataset
from sklearn.ensemble import RandomForestClassifier

from hyper_params import HyperParams

params = HyperParams()


from embeddings.dataloader import TheDataSet
from embeddings.autoencoder import Autoencoder # so we can load this model

USE_AUTOENCODER = True


def get_autoencoder():
    autoencoder = None
    with open('data/autoencoder.pic', 'rb') as f:
        autoencoder = torch.load(f)
    return autoencoder


def load_data():
    dataset = TheDataSet(datafile='data/fulldata.npy')
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


import pandas as pd
df_X = pd.DataFrame(X)
df_y = pd.DataFrame(y, columns=['y'])
df_data = pd.concat([df_X, df_y], axis=1)

from dataproc import sampling
params.validation_set_fraction=0.18
params.test_set_fraction=0.02
params.negative_to_positive_ratio=1
print(params.__dict__)
train_set, df_validation, df_test = sampling.generate_samples(df_dataset=df_data,
                          negative_to_positive_ratio=params.negative_to_positive_ratio,
                          test_set_fraction=params.test_set_fraction,
                          validation_set_fraction=params.validation_set_fraction,
                          random_state=params.random_state,
                          by_column='y')

X_train = train_set.drop(columns=['y'])
y_train = train_set['y']
X_validate = df_validation.drop(columns=['y'])
y_validate = df_validation['y']
print(f"Train shape: {X_train.shape}")
print(f"Validate shape: {X_validate.shape}")
rf = RandomForestClassifier(max_features=40, max_depth=8, n_estimators=90, min_samples_leaf=5, random_state=7)
rf.fit(X_train, y_train)
y_validate_hat = rf.predict(X_validate)
simple_score = rf.score(X_validate, y_validate)
print(f"simple_score: {simple_score}")
# scoring = {'AUC': 'roc_auc', 'Accuracy': 'accuracy', 'Precision': 'precision', 'Recall': 'recall'}
scoring = ['roc_auc','accuracy','precision', 'recall']
cv_scores = cross_validate(rf, X_train, y_train, scoring=scoring)
print_accuracy(cv_scores)
