import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
from sklearn.metrics import roc_auc_score, average_precision_score

def train_logreg(X_train_scaled, y_train, X_test_scaled, y_test):
    # create object logistic regression
    logreg = LogisticRegression()
    # train model
    logreg.fit(X_train_scaled, y_train)
    # predict
    y_pred = logreg.predict(X_test_scaled)
    # print confusion matrix
    print(confusion_matrix(y_test, y_pred))
    # plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
    plt.xlabel('Predict')
    plt.ylabel('Actual')
    plt.show()
    # print confusion matrix
    print(confusion_matrix(y_test, y_pred))
    # print classification report
    print(classification_report(y_test, y_pred))
    # print accuracy score
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    # print precision score
    print('Precision: ', precision_score(y_test, y_pred))
    # print recall score
    print('Recall: ', recall_score(y_test, y_pred))
    # print f1 score
    print('F-1 Score: ', f1_score(y_test, y_pred))
    
    # calculate ROC curve values
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test_scaled)[:, 1])
    
    # plot roc curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')  # Random classifier line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
    # calculate precision-recall values
    precision, recall, _ = precision_recall_curve(y_test, logreg.predict_proba(X_test_scaled)[:, 1])
    
    # plot precision-recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label='Precision-Recall curve (AP = %0.2f)' % average_precision_score(y_test, y_pred))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()
    
    # print auc score
    print('AUC: ', roc_auc_score(y_test, y_pred))
    # print average precision score
    print('Average Precision Score: ', average_precision_score(y_test, y_pred))
    
    return logreg






def log_reg_balance(X_train_scaled, y_train, X_test_scaled, y_test):
    #hyperparameter tuning threshold
    logreg_balance = LogisticRegression(class_weight='balanced',max_iter=1000,solver='lbfgs')
    logreg_balance.fit(X_train_scaled,y_train)
    y_pred=logreg_balance.predict(X_test_scaled)
    print(confusion_matrix(y_test,y_pred))
    #plot confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
    plt.xlabel('Predict')
    plt.ylabel('Actual')
    plt.show()
    #print classification report
    print(classification_report(y_test,y_pred))
    #print accuracy score
    print('Accuracy: ',accuracy_score(y_test,y_pred))
    #print precision score
    print('Precision: ',precision_score(y_test,y_pred))
    #print recall score
    print('Recall: ',recall_score(y_test,y_pred))
    #print f1 score
    print('F1 Score: ',f1_score(y_test,y_pred))

    # calculate ROC curve values
    fpr, tpr, thresholds = roc_curve(y_test, logreg_balance.predict_proba(X_test_scaled)[:, 1])
    
    # plot roc curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')  # Random classifier line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
    # calculate precision-recall values
    precision, recall, _ = precision_recall_curve(y_test, logreg_balance.predict_proba(X_test_scaled)[:, 1])
    
    # plot precision-recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label='Precision-Recall curve (AP = %0.2f)' % average_precision_score(y_test, y_pred))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()
    
    # print auc score
    print('AUC: ', roc_auc_score(y_test, y_pred))
    # print average precision score
    print('Average Precision Score: ', average_precision_score(y_test, y_pred))

    return logreg_balance

def train_class_weight(X_train_scaled,y_train,X_test_scaled,y_test):
    class_weight={0:0.5,1:289}
    logreg_class_weight=LogisticRegression(class_weight=class_weight,solver='newton-cg',max_iter=1000)
    #train model
    logreg_class_weight.fit(X_train_scaled,y_train)
    #predict
    y_pred=logreg_class_weight.predict(X_test_scaled)
    #print confusion matrix
    print(confusion_matrix(y_test,y_pred))
    #plot confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
    plt.xlabel('Predict')
    plt.ylabel('Actual')
    plt.show()
    #print classification report
    print(classification_report(y_test,y_pred))
    #print accuracy score
    print(accuracy_score(y_test,y_pred))
    #print precision score
    print(precision_score(y_test,y_pred))
    #print recall score
    print(recall_score(y_test,y_pred))
    #print f1 score
    print(f1_score(y_test,y_pred))
    return logreg_class_weight

#create function to train logistic regression with hyperparameter tuning class weight and solver using GridSearchCV and StratifiedGroupKFold
def param_log_reg(X_train_scaled, y_train, X_test_scaled, y_test):
    logreg = LogisticRegression(solver='newton-cg', max_iter=1000)
    # create class weight
    weights = {'class_weight': [{0: x, 1: 1.0 - x} for x in np.linspace(0.0, 0.99, 200)]}
    grid_search = GridSearchCV(logreg, weights, cv=5, scoring='f1', n_jobs=-1)
    # Perform the grid search
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model from the search
    best_model = grid_search.best_estimator_

    #print best hyperparameter

    print('Best hyperparameter: ', grid_search.best_params_)

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test_scaled)

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
    print('Accuracy: ',accuracy_score(y_test, y_pred))
    # Print precision score
    print('Precision: ',precision_score(y_test, y_pred))
    # Print recall score
    print('Recall: ',recall_score(y_test, y_pred))
    # Print f1 score
    print('F1 Score: ',f1_score(y_test, y_pred))
    return grid_search

def cons_learn_logreg(X_train_scaled, y_train, X_test_scaled, y_test):
    class_weight={0:1,1:10}
    logreg = LogisticRegression(class_weight=class_weight, solver='newton-cg', max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    y_pred = logreg.predict(X_test_scaled)
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

    return logreg








    