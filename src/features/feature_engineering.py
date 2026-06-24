# src/features/feature_engineering.py
# PURPOSE: Create new useful features for ML model

import pandas as pd
import numpy as np
import os

def engineer_features(df):
    """
    Create new features from existing columns.
    More features = better ML model accuracy.
    """
    print("\n Engineering new features...")

    # Feature 1 — Total Infrastructure Risk Score
    # Combines all infrastructure related columns
    df['InfrastructureRisk'] = (
        df['DamsQuality'] +
        df['DrainageSystems'] +
        df['DeterioratingInfrastructure']
    ) / 3
    print("Created: InfrastructureRisk")

    # Feature 2 — Environmental Risk Score
    # Combines environmental factors
    df['EnvironmentalRisk'] = (
        df['Deforestation'] +
        df['WetlandLoss'] +
        df['Siltation']
    ) / 3
    print("Created: EnvironmentalRisk")

    # Feature 3 — Human Activity Risk Score
    # Combines human caused factors
    df['HumanActivityRisk'] = (
        df['Urbanization'] +
        df['Encroachments'] +
        df['AgriculturalPractices'] +
        df['InadequatePlanning']
    ) / 4
    print("Created: HumanActivityRisk")

    # Feature 4 — Climate Risk Score
    df['ClimateRisk'] = (
        df['MonsoonIntensity'] +
        df['ClimateChange']
    ) / 2
    print("Created: ClimateRisk")

    # Feature 5 — Overall Risk Score
    # Combined score of all risk factors
    df['OverallRiskScore'] = (
        df['InfrastructureRisk'] +
        df['EnvironmentalRisk'] +
        df['HumanActivityRisk'] +
        df['ClimateRisk']
    ) / 4
    print("Created: OverallRiskScore")

    # Feature 6 — Flood Risk Category
    # Convert probability to category labels
    df['RiskCategory'] = pd.cut(
        df['FloodProbability'],
        bins=[0, 0.45, 0.50, 0.55, 1.0],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    print("Created: RiskCategory")

    # Feature 7 — High Monsoon Flag
    # 1 if monsoon intensity is above average
    df['HighMonsoon'] = (df['MonsoonIntensity'] > 5).astype(int)
    print("Created: HighMonsoon")

    print(f"\n Total features: {df.shape[1]} columns")
    print(f"Total samples: {df.shape[0]} rows")

    return df

def save_engineered_data(df):
    """Save featured dataset."""
    os.makedirs('data/processed', exist_ok=True)
    path = 'data/processed/flood_features.csv'
    df.to_csv(path, index=False)
    print(f"\n Featured dataset saved: {path}")

if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv('data/processed/flood_data_clean.csv')
    print(f"Loaded clean data: {df.shape}")

    # Engineer features
    df = engineer_features(df)

    # Save
    save_engineered_data(df)

    print("\n New Features Sample:")
    new_cols = ['InfrastructureRisk', 'EnvironmentalRisk',
                'HumanActivityRisk', 'ClimateRisk',
                'OverallRiskScore', 'RiskCategory', 'HighMonsoon']
    print(df[new_cols].head())

    print("\n Risk Category Distribution:")
    print(df['RiskCategory'].value_counts())