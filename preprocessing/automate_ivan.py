import pandas as pd
import numpy as np
import os


def preprocess_data(df):
    df = df.copy()

    # =========================
    # DATA CLEANING
    # =========================
    df = df.drop(columns=['customerID'], errors='ignore')

    # Fix TotalCharges dan tipe
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')

    replace_map = {
        'No internet service': 'No',
        'No phone service': 'No'
    }

    obj_cols = df.select_dtypes(include='object').columns

    for col in obj_cols:
        df[col] = df[col].apply(lambda x: replace_map.get(x, x))

    df = df.drop_duplicates()

    # =========================
    # HANDLE MISSING
    # =========================
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # =========================
    # FEATURE ENGINEERING
    # =========================
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[-1, 12, 36, 72],
        labels=['short', 'medium', 'long']
    )

    df['avg_charge'] = df['TotalCharges'] / (df['tenure'] + 1)

    # =========================
    # FEATURE SELECTION
    # =========================
    df = df.drop(columns=['TotalCharges'])

    # =========================
    # ENCODING
    # =========================
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # =========================
    # FINAL CLEAN
    # =========================
    bool_cols = df.select_dtypes('bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    df.columns = df.columns.str.replace(" ", "_", regex=False)
    df.columns = df.columns.str.replace("(", "", regex=False)
    df.columns = df.columns.str.replace(")", "", regex=False)

    return df


if __name__ == "__main__":
    url = "https://docs.google.com/spreadsheets/d/15D7O9LWyW_KpULzVR0aQJC3vY1DovTf4/export?format=xlsx"

    df = pd.read_excel(url)
    df_processed = preprocess_data(df)

    output_path = os.path.join(os.path.dirname(__file__), "data_preprocessing.csv")
    df_processed.to_csv(output_path, index=False)

    print("Preprocessing selesai ✅")
    print("Shape:", df_processed.shape)