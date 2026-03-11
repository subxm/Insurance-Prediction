#1. Load scaler.pkl and model.pkl from artifacts folder
#2. Create a function to predict

import pickle
import numpy as np
import os

class Insurance_Prediction:
    def __init__(self):
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        artifacts_dir = os.path.join(script_dir, "..", "artifacts")
        
        with open(os.path.join(artifacts_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

        with open(os.path.join(artifacts_dir, "model.pkl"), "rb") as f:
            self.model = pickle.load(f)

    def prediction(self,Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs):
        Input = np.array([[Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs]])
        Scaled_Input = self.scaler.transform(Input)
        result = self.model.predict(Scaled_Input)
        return result[0]