import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
#import randomforestclassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve

def train_rfs_bal(X_train, y_train, X_test, y_test):
    rfs=RandomForestClassifier(class_weight='balanced')
    rfs.fit(X_train,y_train)
    y_pred=rfs.predict(X_test)
    # Print confusion matrix
    print(confusion_matrix(y_test, y_pred))
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
    plt.xlabel('Predict')
    plt.ylabel('Actual')
    plt.show()

    # Print classification report
    print(classification_report(y_test, y_pred))

    # Print accuracy score
    print('Accuracy: ', accuracy_score(y_test, y_pred))

    # Print precision score
    print('Precision: ', precision_score(y_test, y_pred))

    # Print recall score
    print('Recall: ', recall_score(y_test, y_pred))

    # Print f1 score
    print('F1 Score: ', f1_score(y_test, y_pred))

    return rfs

def train_rfs_bag(X_train,y_train,X_test,y_test):
    rfs_bag=RandomForestClassifier(class_weight='balanced_subsample')
    rfs_bag.fit(X_train,y_train)
    y_pred=rfs_bag.predict(X_test)
    # Print confusion matrix
    print(confusion_matrix(y_test, y_pred))
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
    plt.xlabel('Predict')
    plt.ylabel('Actual')
    plt.show()

    # Print classification report
    print(classification_report(y_test, y_pred))

    # Print accuracy score
    print('Accuracy: ', accuracy_score(y_test, y_pred))

    # Print precision score
    print('Precision: ', precision_score(y_test, y_pred))

    # Print recall score
    print('Recall: ', recall_score(y_test, y_pred))

    # Print f1 score
    print('F1 Score: ', f1_score(y_test, y_pred))

    return rfs_bag
