# src/data/data_loader.py
# PURPOSE: Load all datasets into the project

import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_flood_data():
    """
    Load the main flood prediction dataset.
    Returns a pandas DataFrame.
    """
    # Build the file path
    path = os.path.join('data', 'raw', 'flood_records', 'flood_data.csv')
    
    # Read the CSV file
    df = pd.read_csv(path)
    
    print(f"✅ Flood data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def load_rainfall_data():
    """
    Load the India rainfall dataset.
    Returns a pandas DataFrame.
    """
    path = os.path.join('data', 'raw', 'rainfall', 'rainfall_data.csv')
    
    df = pd.read_csv(path)
    
    print(f"✅ Rainfall data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def load_all_data():
    """
    Load all datasets at once.
    Returns both DataFrames.
    """
    print("\n📂 Loading all datasets...")
    print("-" * 40)
    
    flood_df = load_flood_data()
    rainfall_df = load_rainfall_data()
    
    print("-" * 40)
    print("✅ All datasets loaded successfully!\n")
    
    return flood_df, rainfall_df

# Test when run directly
if __name__ == "__main__":
    flood_df, rainfall_df = load_all_data()
    print("\nFlood Data Sample:")
    print(flood_df.head(3))
    print("\nRainfall Data Sample:")
    print(rainfall_df.head(3))