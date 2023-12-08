import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score,f1_score

def train_xgb(X_train,y_train,X_test,y_test):
    clf=XGBClassifier()
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
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
    return clf

def param_xgb(X_train,y_train,X_test,y_test):
    # Calculate class weights
    class_counts = y_train.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]

    # Set up XGBoost classifier with scale_pos_weight
    clf = XGBClassifier(scale_pos_weight=scale_pos_weight)

    # Train model
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

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

    return clf