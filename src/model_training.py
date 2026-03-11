#1. Load the Processed data from processed folder
#2. Create model and train data
#3. Save the model into artifacts folder

import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

X_train = pd.read_csv("../data/processed/X_train.csv")
X_test = pd.read_csv("../data/processed/X_test.csv")  
y_train = pd.read_csv("../data/processed/y_train.csv")
y_test = pd.read_csv("../data/processed/y_test.csv")

print(X_train)

model = LinearRegression()
model.fit(X_train,y_train)

with open("../artifacts/model.pkl","wb") as f:
    pickle.dump(model,f)