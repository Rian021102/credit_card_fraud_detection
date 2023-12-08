import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import logistic regression classifier
from sklearn.linear_model import LogisticRegression
#import Oversampling and undersampling libraries
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
#import random over sampler
from imblearn.over_sampling import RandomOverSampler
#import evaluation libraries
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
#import precision recall and f1 score
from sklearn.metrics import precision_score, recall_score, f1_score

# create function to handle RandomOverSampler
def oversample(X_train_scaled, y_train):
    # create object for RandomOverSampler
    ros = RandomOverSampler()
    # fit training data only
    X_ros, y_ros = ros.fit_resample(X_train_scaled, y_train)
    # return oversampled data
    return X_ros, y_ros

# create function to handle SMOTE
def smote(X_train_scaled, y_train):
    # create object for SMOTE
    sm = SMOTE()
    # fit training data only
    X_sm, y_sm = sm.fit_resample(X_train_scaled, y_train)
    # return oversampled data
    return X_sm, y_sm

# create function to handle NearMiss
def nearmiss(X_train_scaled, y_train):
    # create object for NearMiss
    nm = NearMiss()
    # fit training data only
    X_nm, y_nm = nm.fit_resample(X_train_scaled, y_train)
    # return oversampled data
    return X_nm, y_nm

#train logistic regression model using oversampled data
def oversample_lr(X_ros, y_ros, X_test_scaled, y_test):
    # create object for LogisticRegression
    lr = LogisticRegression()
    # fit training data only
    lr.fit(X_ros, y_ros)
    # predict using oversampled data
    y_pred = lr.predict(X_test_scaled)
    # plot confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
    plt.xlabel('Predict')
    plt.ylabel('Actual}')
    plt.show()
    # print confusion matrix
    print(confusion_matrix(y_test, y_pred))
    # print accuracy score
    print(accuracy_score(y_test, y_pred))
    # print classification report
    print(classification_report(y_test, y_pred))
    # print precision score
    print(precision_score(y_test, y_pred))
    # print recall score
    print(recall_score(y_test, y_pred))
    # print f1 score
    print(f1_score(y_test, y_pred))

#train logistic regression model using SMOTE data
def smote_lr(X_sm, y_sm, X_test_scaled, y_test):
    # create object for LogisticRegression
    lr = LogisticRegression()
    # fit training data only
    lr.fit(X_sm, y_sm)
    # predict using SMOTE data
    y_pred = lr.predict(X_test_scaled)
    # plot confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
    plt.xlabel('Predict')
    plt.ylabel('Actual}')
    plt.show()
    # print confusion matrix
    print(confusion_matrix(y_test, y_pred))
    # print accuracy score
    print(accuracy_score(y_test, y_pred))
    # print classification report
    print(classification_report(y_test, y_pred))
    # print precision score
    print(precision_score(y_test, y_pred))
    # print recall score
    print(recall_score(y_test, y_pred))
    # print f1 score
    print(f1_score(y_test, y_pred))

#train logistic regression model using NearMiss data
def nearmiss_lr(X_nm, y_nm, X_test_scaled, y_test):
    # create object for LogisticRegression
    lr = LogisticRegression()
    # fit training data only
    lr.fit(X_nm, y_nm)
    # predict using NearMiss data
    y_pred = lr.predict(X_test_scaled)
    # plot confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
    plt.xlabel('Predict')
    plt.ylabel('Actual}')
    plt.show()
    # print confusion matrix
    print(confusion_matrix(y_test, y_pred))
    # print accuracy score
    print(accuracy_score(y_test, y_pred))
    # print classification report
    print(classification_report(y_test, y_pred))
    # print precision score
    print(precision_score(y_test, y_pred))
    # print recall score
    print(recall_score(y_test, y_pred))
    # print f1 score
    print(f1_score(y_test, y_pred))
    