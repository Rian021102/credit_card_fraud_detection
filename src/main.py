from dataval import loaddata,eda_train, preprocessing,calculate_class_weight
from model_log_reg import train_logreg,log_reg_balance,train_class_weight,param_log_reg,cons_learn_logreg
from logreg_thresh import log_reg_balance_thresh
from model_xgb import param_xgb,train_xgb
from rfsmodel import train_rfs_bal,train_rfs_bag
def main():
    file="/Users/rianrachmanto/pypro/data/creditcard.csv"
    X_train,X_test,y_train,y_test=loaddata(file)
    #eda_train(X_train,y_train)
    X_train_scaled,X_test_scaled=preprocessing(X_train,X_test,y_train,y_test)
    #train_logreg(X_train_scaled,y_train,X_test_scaled,y_test)
    #log_reg_balance(X_train_scaled,y_train,X_test_scaled,y_test)
    #log_reg_balance_thresh(X_train_scaled,y_train,X_test_scaled,y_test)
    #train_class_weight(X_train_scaled,y_train,X_test_scaled,y_test)
    #param_log_reg(X_train_scaled,y_train,X_test_scaled,y_test)
    #train_xgb(X_train,y_train,X_test,y_test)
    param_xgb(X_train,y_train,X_test,y_test)
    #cons_learn_logreg(X_train_scaled,y_train,X_test_scaled,y_test)
    #train_rfs_bal(X_train,y_train,X_test,y_test)
    #train_rfs_bag(X_train,y_train,X_test,y_test)
if __name__ == "__main__":
    main()
