# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 19:45:09 2021

@author: tatia
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import matplotlib.pyplot as plt
from dataproc.roc_auc_curves import plt_roc_auc_curve, plt_precision_recall_curve

RANDOM_STATE = 42
score_f1 = make_scorer(f1_score, average='binary', pos_label=1)
score_pr = make_scorer(precision_score, average='binary', pos_label=1)
# Load stored data set
#fulldata = pd.read_csv('data/fulldata_extra.csv')
fulldata = pd.read_csv('data/fulldata_cephalosporin.csv')

def plot_learning_curve(X, y, model, scoring, scoring_name):
    train_sizes, train_scores, test_scores = learning_curve(estimator=model,
                                  X=X,
                                  y=y,
                                  train_sizes = np.linspace(0.1, 1.0, 5),
                                  scoring=scoring,   
                                  cv=3)
    train_mean= np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean= np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training')
    plt.fill_between(train_sizes, train_mean+train_std, train_mean-train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='orange', linestyle='--', marker='s', markersize=5, label='test')
    plt.fill_between(train_sizes, test_mean+test_std, test_mean-test_std, alpha=0.15, color='orange')
    plt.grid()
    plt.xlabel('Sample size')
    plt.ylabel(scoring_name)
    plt.title('Learning Curve')
    plt.legend(loc='upper right')
    plt.show()
    
def plot_validation_curve(X, y, model, scoring, scoring_name, param_name, param_range):
    # Set parameter range
    train_scores, test_scores = validation_curve(estimator=model,
                                  X=X,
                                  y=y,
                                  param_name='randomforestclassifier__'+param_name,
                                  param_range =param_range,
                                  scoring=scoring,   
                                  cv=3)
    train_mean= np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean= np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training')
    plt.fill_between(param_range, train_mean+train_std, train_mean-train_std, alpha=0.15, color='blue')
    plt.plot(param_range, test_mean, color='orange', linestyle='--', marker='s', markersize=5, label='test')
    plt.fill_between(param_range, test_mean+test_std, test_mean-test_std, alpha=0.15, color='orange')
    plt.grid()
    plt.xlabel(param_name)
    plt.ylabel(scoring_name)
    plt.title('Validation Curve: ' + param_name)
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    
    # Split data:
    #----------------------------------------
    y = fulldata['RESISTANT_YN']
    X = fulldata.drop(columns=['RESISTANT_YN'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=RANDOM_STATE)
    # summarize class distribution
    print(Counter(y))
    print(Counter(y_train))
    
    # Perform random sampling:
    #----------------------------------------
    # Oversample minority class
    oversample = RandomOverSampler(sampling_strategy = 'minority')
    # fit and apply the transform
    X_over, y_over = oversample.fit_resample(X_train, y_train)
    print('Oversampling: ', Counter(y_over))
    
    # Undersample majority class
    undersample = RandomUnderSampler(sampling_strategy = 0.55)
    # fit and apply the transform
    X_under, y_under = undersample.fit_resample(X_train, y_train)
    print('Undersampling: ', Counter(y_under))
    
    # Plot Learning Curve: 
    #----------------------------------------
    # Model
    forest = make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE, 
                               class_weight='balanced_subsample',
                               n_estimators=150,
                               max_depth=20,
                               max_leaf_nodes=70,
                               max_features=40,
                               max_samples=0.9,
                               min_samples_leaf=2,
                               min_samples_split=10))
    # Learning curve as function of sample siz
    plot_learning_curve(X_train, y_train, forest, score_f1, "F1")
    
    # Plot Validation Curve: 
    #----------------------------------------
    # Model
    forest = make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE, 
                               class_weight='balanced_subsample',
                               n_estimators=150,
                               max_depth=13,
                               max_leaf_nodes=50,
                               max_features=60,
                               max_samples=0.9,
                               min_samples_leaf=2,
                               min_samples_split=10))
    # Validation curve as function of hyperparameter
    param_name = 'max_features'
    param_range = [ 10,20,30,40,50,60,70]
    plot_validation_curve(X_over, y_over, forest, score_f1, "F1", param_name, param_range)
    
    # Random Forest Model:
    #-----------------------------------------
    forest = RandomForestClassifier(random_state=RANDOM_STATE, 
                               class_weight='balanced_subsample',
                               n_estimators=150,
                               max_depth=13,
                               max_leaf_nodes=50,
                               max_features=60,
                               max_samples=0.9,
                               min_samples_leaf=2,
                               min_samples_split=10)
    # Train model
    forest.fit(X_over, y_over)
    # Prediction
    y_true, y_pred = y_test, forest.predict(X_test)
    # Classification report (recall, preccision, f-score, accuracy):
    print(classification_report(y_true, y_pred))
    print()
    tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()
    print('TN:',tn, 'FP:',fp, 'FN:',fn, 'TP:',tp )
    print()
    scores = cross_val_score(forest, X_train, y_train, scoring=score_f1, cv=5)
    print('CV F1-score: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    plt_roc_auc_curve(forest, X_test, y_test, model_name='Rand Forest') 
    plt_precision_recall_curve(forest, X_test, y_test, model_name='Rand Forest')
    
    # Feature Importance:
    importance = forest.feature_importances_
    indices = np.argsort(importance)[::-1] # highest to lowest
    # Table
    featureScores = pd.DataFrame(data = {'Rank': range(1, X_over.shape[1] + 1),
                                         'Label': X_over.columns[indices],
                                         'Importance Value': importance[indices]})
    
    # Use Top 50 most important features
    #-----------------------------------------
    columns = list(featureScores['Label'][0:50].values)
    y = fulldata['RESISTANT_YN']
    X_best = fulldata[columns]
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_best, y, stratify=y, test_size=0.3, random_state=RANDOM_STATE)
    X_over_b, y_over_b = oversample.fit_resample(X_train_b, y_train_b)
    print('Oversampling: ', Counter(y_over_b))
    # Model
    forest = make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE, 
                               class_weight='balanced_subsample',
                               n_estimators=150,
                               max_depth=13,
                               max_leaf_nodes=50,
                               max_features=40,
                               max_samples=0.9,
                               min_samples_leaf=2,
                               min_samples_split=10))
     # Validation curve as function of hyperparameter
    param_name = 'max_leaf_nodes'
    param_range = [ 20,30,40,50,60]
    plot_validation_curve(X_over_b, y_over_b, forest, score_f1, "F1", param_name, param_range)
    
    
    forest = RandomForestClassifier(random_state=RANDOM_STATE, 
                               class_weight='balanced_subsample',
                               n_estimators=150,
                               max_depth=13,
                               max_leaf_nodes=50,
                               max_features=40,
                               max_samples=0.9,
                               min_samples_leaf=2,
                               min_samples_split=10)
    # Train model
    forest.fit(X_over_b, y_over_b)
    # Prediction
    y_true, y_pred = y_test_b, forest.predict(X_test_b)
    # Classification report (recall, preccision, f-score, accuracy):
    print(classification_report(y_true, y_pred))
    print()
    tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()
    print('TN:',tn, 'FP:',fp, 'FN:',fn, 'TP:',tp )
    print()
    scores = cross_val_score(forest, X_over_b, y_over_b, scoring=score_f1, cv=5)
    print('CV F1-score: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    plt_roc_auc_curve(forest, X_test_b, y_test_b, model_name='Rand Forest') 
    plt_precision_recall_curve(forest, X_test_b, y_test_b, model_name='Rand Forest')
    
    
