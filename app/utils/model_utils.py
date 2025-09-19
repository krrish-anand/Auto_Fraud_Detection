# Utility functions for the Streamlit app
import pandas as pd
import joblib
import os

def load_data(data_path):
    return pd.read_csv(data_path)

def load_model(model_path):
    return joblib.load(model_path)

def preprocess_input(input_dict, columns):
    # Ensure the input is in the correct order and format for the model
    import numpy as np
    return np.array([input_dict.get(col, 0) for col in columns]).reshape(1, -1)
