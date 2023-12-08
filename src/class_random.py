import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import random forest
from sklearn.ensemble import RandomForestClassifier
# import confusion matrix
from sklearn.metrics import confusion_matrix
# import classification report
from sklearn.metrics import classification_report
# import accuracy score
from sklearn.metrics import accuracy_score
# import precision score
from sklearn.metrics import precision_score
# import recall score
from sklearn.metrics import recall_score
# import f1 score
from sklearn.metrics import f1_score
# import roc auc score
from sklearn.metrics import roc_auc_score
# import roc curve
from sklearn.metrics import roc_curve
# import precision recall curve
from sklearn.metrics import precision_recall_curve
# import average precision score

def param_rf(X_train,y_train,X_test,y_test):
    model_rf = RandomForestClassifier(class_weight='balanced')
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_test)
    # Print confusion matrix
    print(confusion_matrix(y_test, y_pred))
    #plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax,fmt='g',cmap='Blues') #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Not Fraud', 'Fraud']); ax.yaxis.set_ticklabels(['Not Fraud', 'Fraud'])
    plt.show()
    # Print classification report
    print(classification_report(y_test, y_pred))
    # Print accuracy score
    print(accuracy_score(y_test, y_pred))
    # Print precision score
    print(precision_score(y_test, y_pred))
    # Print recall score
    print(recall_score(y_test, y_pred))

    return model_rf
