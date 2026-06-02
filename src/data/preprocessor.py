# src/data/preprocessor.py
# PURPOSE: Clean and preprocess all datasets

import pandas as pd
import numpy as np
import os

def clean_flood_data(df):
    """
    Clean the flood prediction dataset.
    Steps: remove duplicates, handle missing values, fix data types.
    """
    print("\n🧹 Cleaning Flood Dataset...")
    print(f"   Original shape: {df.shape}")
    
    # Step 1 — Remove duplicate rows
    df = df.drop_duplicates()
    print(f"   After removing duplicates: {df.shape}")
    
    # Step 2 — Remove the 'id' column (not useful for ML)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        print("   Dropped 'id' column")
    
    # Step 3 — Check for missing values
    missing = df.isnull().sum().sum()
    print(f"   Missing values found: {missing}")
    
    # Step 4 — Fill missing values with column mean
    df = df.fillna(df.mean(numeric_only=True))
    print("   Missing values filled with column mean")
    
    # Step 5 — Remove outliers using IQR method
    # IQR = Interquartile Range (Q3 - Q1)
    # Values beyond 1.5*IQR are considered outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    before = len(df)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)  # 25th percentile
        Q3 = df[col].quantile(0.75)  # 75th percentile
        IQR = Q3 - Q1                 # Interquartile range
        
        lower = Q1 - 1.5 * IQR       # Lower boundary
        upper = Q3 + 1.5 * IQR       # Upper boundary
        
        # Keep only rows within boundaries
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    after = len(df)
    print(f"   Outliers removed: {before - after} rows")
    print(f"   Final shape: {df.shape}")
    
    return df

def clean_rainfall_data(df):
    """
    Clean the rainfall dataset.
    """
    print("\n🧹 Cleaning Rainfall Dataset...")
    print(f"   Original shape: {df.shape}")
    
    # Step 1 — Remove duplicates
    df = df.drop_duplicates()
    
    # Step 2 — Rename columns for clarity
    df = df.rename(columns={
        'subdivision': 'State',
        'YEAR': 'Year',
        'JUN-SEP': 'TotalMonsoonRainfall'
    })
    
    # Step 3 — Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    
    # Step 4 — Remove rows where rainfall is 0 or negative
    df = df[df['TotalMonsoonRainfall'] > 0]
    
    print(f"   Final shape: {df.shape}")
    print("   ✅ Rainfall data cleaned!")
    
    return df

def save_cleaned_data(flood_df, rainfall_df):
    """
    Save cleaned datasets to the processed folder.
    """
    # Create processed folder if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save cleaned files
    flood_df.to_csv('data/processed/flood_data_clean.csv', index=False)
    rainfall_df.to_csv('data/processed/rainfall_data_clean.csv', index=False)
    
    print("\n💾 Cleaned datasets saved to data/processed/")
    print("   ✅ flood_data_clean.csv")
    print("   ✅ rainfall_data_clean.csv")

# Test when run directly
if __name__ == "__main__":
    from data_loader import load_all_data
    
    # Load data
    flood_df, rainfall_df = load_all_data()
    
    # Clean data
    flood_clean = clean_flood_data(flood_df)
    rainfall_clean = clean_rainfall_data(rainfall_df)
    
    # Save cleaned data
    save_cleaned_data(flood_clean, rainfall_clean)
    
    print("\n📊 Flood Data Info:")
    print(flood_clean.describe().round(2))