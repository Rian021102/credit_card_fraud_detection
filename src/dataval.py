import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import standar scaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def loaddata(path):
    df=pd.read_csv(path)
    print(df.head())
    #count class
    print(df.Class.value_counts())
    #count ratio
    print(df.Class.value_counts()/len(df))
    #select X and y
    X=df.drop('Class',axis=1)
    y=df['Class']
    #split data
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    #print shape
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    return X_train,X_test,y_train,y_test

def eda_train(X_train,y_train):
    #concat X_train and y_train
    eda_train=pd.concat([X_train,y_train],axis=1)
    #count class
    print(eda_train.Class.value_counts())
    #print missing value
    print(eda_train.isnull().sum())
    #print number of sample
    print(len(eda_train))
    #filter where class is 0
    eda_train_0=eda_train[eda_train.Class==0]
    #filter where class is 1
    eda_train_1=eda_train[eda_train.Class==1]
    #print sample of train
    total_sample=len(y_train)
    print("Total sample: ",total_sample)
    total_sample_0=len(eda_train_0)
    print("Total sample class 0: ",total_sample_0)
    total_sample_1=len(eda_train_1)
    print("Total sample class 1: ",total_sample_1)
    #calculate class weight
    class_weight_0=total_sample/(2*total_sample_0)
    class_weight_1=total_sample/(2*total_sample_1)
    print("Class weight 0: ",class_weight_0)
    print("Class weight 1: ",class_weight_1)
   #plot count plot of class and pie chart beside it. Count plot scale needs to show the class imbalance
    plt.figure(figsize=(10,8))
    plt.subplot(1,2,1)
    plt.title('Count Class')
    sns.countplot(x='Class',data=eda_train)
    plt.subplot(1,2,2)
    plt.title('Pie Chart Class')
    plt.pie(eda_train.Class.value_counts(),labels=['Not Fraud','Fraud'],autopct='%1.1f%%')
    plt.show()
   
    

    #plot histogram time vs class
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.title('Histogram Time vs Class')
    plt.xlabel('Time')
    plt.ylabel('Class 0')
    plt.hist(eda_train_0.Time,bins=50)
    plt.subplot(2,1,2)
    plt.title('Histogram Time vs Class')
    plt.xlabel('Time')
    plt.ylabel('Class 1')
    plt.hist(eda_train_1.Time,bins=50)
   #print features with correlation > 0.8
    corr=X_train.corr()
    #turn corr to dataframe
    corr_df=pd.DataFrame(corr)
    print(corr_df)

#preprocessing data
def preprocessing(X_train,X_test,y_train,y_test):
    #standar scaler
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    print("X_train_scaled: ", X_train_scaled.shape)
    print("X_test_scaled: ",X_test_scaled.shape)
    print("y_train: ", y_train.shape)
    print("y_test: ", y_test.shape)
    return X_train_scaled,X_test_scaled

def calculate_class_weight(y_train):
    class_weights = compute_class_weight(class_weight='balanced',classes=[0, 1], y=y_train)
    print("Class weight: ",class_weights)
    return class_weights


