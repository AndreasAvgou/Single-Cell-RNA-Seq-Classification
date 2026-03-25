import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X = train_df.drop('tag', axis=1).values
    y = train_df['tag'].values
    X_test_final = test_df.values
    return X, y, X_test_final

def preprocess_pipeline(X, y, X_test_final):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.15, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test_final = scaler.transform(X_test_final)
    
    return X_train, X_val, y_train, y_val, X_test_final, le