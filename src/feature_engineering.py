#1. Load the training and testing data
#2. Scale the training data
#3. save scaled data into processed folder

import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_preprocessing import load_and_Split_data

X_train,X_test,y_train,y_test = load_and_Split_data()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # fit_transform() is a method used to transform the data into scaled form. range is -1 to 1
X_test_scaled = scaler.transform(X_test) 

# 3. save scaled data into processed folder
pd.DataFrame(X_train_scaled).to_csv("../data/processed/X_train.csv",index=False)
pd.DataFrame(X_test_scaled).to_csv("../data/processed/X_test.csv",index=False)
pd.DataFrame(y_train).to_csv("../data/processed/y_train.csv",index=False)
pd.DataFrame(y_test).to_csv("../data/processed/y_test.csv",index=False)
 
#save the scaler

with open("../artifacts/scaler.pkl","wb") as f:
    pickle.dump(scaler,f)

print("Successfully saved the scaler.pkl")