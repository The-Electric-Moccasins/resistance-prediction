# -*- coding: utf-8 -*-
"""
Created on Tue May  4 19:24:20 2021

@author: tatia
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter



def split_dataset(data, test_size, random_state=42):
    y = data['RESISTANT_YN']
    X = data.drop(columns=['RESISTANT_YN'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)
    print('Training Set: ', Counter(y_train))
    return X_train, X_test, y_train, y_test


def oversample_minority_class(X_train, y_train):
    # Oversample minority class
    oversample = RandomOverSampler(sampling_strategy = 'minority')
    # fit and apply the transform
    X_over, y_over = oversample.fit_resample(X_train, y_train)
    print('Oversampling: ', Counter(y_over))
    return X_over, y_over


def undersample_majority_class(X_train, y_train, sampling_strategy = 0.55):
    # Undersample majority class
    undersample = RandomUnderSampler(sampling_strategy = sampling_strategy)
    # fit and apply the transform
    X_under, y_under = undersample.fit_resample(X_train, y_train)
    print('Undersampling: ', Counter(y_under))
    return X_under, y_under


def random_forest_model(X, y, random_state=42):
    forest = RandomForestClassifier(random_state=random_state, 
                               class_weight='balanced_subsample',
                               n_estimators=150,
                               max_depth=13,
                               max_leaf_nodes=50,
                               max_features=40,
                               max_samples=0.9,
                               min_samples_leaf=2,
                               min_samples_split=10)
    # Train model
    forest.fit(X, y)
    # Cross-Validation
    scores = cross_val_score(forest, X, y, scoring='f1_micro', cv=5)
    print('CV F1-score: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    return forest

def rand_forest_feature_importance(model, X, y):
    # Feature Importance:
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1] # highest to lowest
    # Table
    featureScores = pd.DataFrame(data = {'Rank': range(1, X.shape[1] + 1),
                                         'Label': X.columns[indices],
                                         'Importance Value': importance[indices]})
    return featureScores


def model_performance(model, y_true, y_pred):
    # Classification report (recall, preccision, f-score, accuracy):
    print(classification_report(y_true, y_pred))
    print()
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    print('TN:',tn, 'FP:',fp, 'FN:',fn, 'TP:',tp )
    print()
    
    