# prediction_frequence_assurance.ipynb

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from category_encoders import CountEncoder
import os

def load_data(train_input_path, test_input_path, train_output_path):
    """Charge les fichiers CSV d'entraînement et de test."""
    train_df = pd.read_csv(train_input_path)
    test_df = pd.read_csv(test_input_path)
    y_train = pd.read_csv(train_output_path)
    return train_df, test_df, y_train

def identify_column_types(df):
    """Identifie les colonnes numériques et catégorielles."""
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    return num_cols, cat_cols

def fill_missing_values(train_df, test_df, num_cols, cat_cols):
    """Remplace les valeurs manquantes dans les colonnes numériques par 0 et les colonnes catégorielles par -999."""
    train_df[num_cols] = train_df[num_cols].fillna(0)
    test_df[num_cols] = test_df[num_cols].fillna(0)
    train_df[cat_cols] = train_df[cat_cols].fillna(-999)
    test_df[cat_cols] = test_df[cat_cols].fillna(-999)
    return train_df, test_df

def save_cleaned_data(train_df, test_df, train_path='train_cleaned.csv', test_path='test_cleaned.csv'):
    """Sauvegarde les fichiers nettoyés."""
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

def feature_engineering(train_df, test_df, binary_cols):
    """Encode les variables catégorielles avec CountEncoder et standardise les données numériques."""
    encoder = CountEncoder()
    train_df[binary_cols] = encoder.fit_transform(train_df[binary_cols])
    test_df[binary_cols] = encoder.transform(test_df[binary_cols])

    scaler = StandardScaler()
    train_df[binary_cols] = scaler.fit_transform(train_df[binary_cols])
    test_df[binary_cols] = scaler.transform(test_df[binary_cols])
    return train_df, test_df

