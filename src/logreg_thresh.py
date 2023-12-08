from sklearn.metrics import roc_curve, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import numpy as np

def find_best_threshold(y_true, y_proba, metric='f1'):
    thresholds = np.arange(0.1, 1, 0.1)
    best_threshold = 0
    best_score = 0

    for threshold in thresholds:
        y_pred = (y_proba[:, 1] > threshold).astype(int)
        score = 0

        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold

def log_reg_balance_thresh(X_train_scaled, y_train, X_test_scaled, y_test):
    # Hyperparameter tuning threshold
    logreg_balance = LogisticRegression(class_weight='balanced', max_iter=1000)
    logreg_balance.fit(X_train_scaled, y_train)

    # Get predicted probabilities for positive class
    y_proba = logreg_balance.predict_proba(X_test_scaled)

    # Find the best threshold for F1-score (you can choose a different metric)
    best_threshold = find_best_threshold(y_test, y_proba, metric='f1')

    # Apply the best threshold to make predictions
    y_pred = (y_proba[:, 1] > best_threshold).astype(int)

    # Print best threshold
    print("Best Threshold:", best_threshold)

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
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Print precision score
    print("Precision:", precision_score(y_test, y_pred))

    # Print recall score
    print("Recall:", recall_score(y_test, y_pred))

    # Print F1 score
    print("F1 Score:", f1_score(y_test, y_pred))

    return logreg_balance
