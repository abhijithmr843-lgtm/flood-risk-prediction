# src/models/train_model.py
# PURPOSE: Train Random Forest and XGBoost models

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_features():
    """Load the engineered feature dataset."""
    df = pd.read_csv('data/processed/flood_features.csv')
    print(f"✅ Features loaded: {df.shape}")
    return df

def prepare_data(df):
    """
    Prepare data for ML training.
    Split into features (X) and target (y).
    """
    print("\n⚙️ Preparing data for training...")
    
    # Drop non-numeric and target columns from features
    drop_cols = ['FloodProbability', 'RiskCategory']
    
    # X = input features (what model learns from)
    X = df.drop(columns=drop_cols)
    
    # y = target (what model predicts)
    # Encode RiskCategory to numbers
    # Low=0, Medium=1, High=2, Very High=3
    le = LabelEncoder()
    y = le.fit_transform(df['RiskCategory'])
    
    print(f"   Features (X): {X.shape}")
    print(f"   Target (y): {y.shape}")
    print(f"   Classes: {le.classes_}")
    
    # Split into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,      # 20% for testing
        random_state=42,    # For reproducibility
        stratify=y          # Keep class balance
    )
    
    print(f"\n   Training samples: {X_train.shape[0]:,}")
    print(f"   Testing samples: {X_test.shape[0]:,}")
    
    return X_train, X_test, y_train, y_test, le

def train_random_forest(X_train, y_train):
    """
    Train Random Forest model.
    Random Forest = many decision trees voting together.
    """
    print("\n🌲 Training Random Forest...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,   # 100 decision trees
        max_depth=10,       # Max depth of each tree
        random_state=42,    # Reproducibility
        n_jobs=-1,          # Use all CPU cores
        verbose=1           # Show progress
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    print("✅ Random Forest training complete!")
    
    return rf_model

def train_xgboost(X_train, y_train):
    """
    Train XGBoost model.
    XGBoost = Extreme Gradient Boosting.
    Builds trees sequentially, each fixing previous errors.
    """
    print("\n⚡ Training XGBoost...")
    
    xgb_model = XGBClassifier(
        n_estimators=100,       # 100 boosting rounds
        max_depth=6,            # Max tree depth
        learning_rate=0.1,      # How fast model learns
        random_state=42,        # Reproducibility
        eval_metric='mlogloss', # Evaluation metric
        verbosity=1             # Show progress
    )
    
    # Train the model
    xgb_model.fit(X_train, y_train)
    print("✅ XGBoost training complete!")
    
    return xgb_model

def evaluate_model(model, X_test, y_test, model_name, le):
    """
    Evaluate model performance.
    Shows accuracy, precision, recall, F1 score.
    """
    print(f"\n📊 Evaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed report
    print(f"\n   Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_
    ))
    
    return accuracy, y_pred

def plot_confusion_matrix(y_test, y_pred, model_name, le):
    """Plot confusion matrix for model evaluation."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    # Save plot
    filename = model_name.lower().replace(' ', '_')
    plt.savefig(f'reports/figures/{filename}_confusion_matrix.png', dpi=150)
    plt.show()
    print(f"✅ Confusion matrix saved!")

def plot_feature_importance(model, X_train, model_name):
    """Plot top 15 most important features."""
    plt.figure(figsize=(10, 8))
    
    # Get feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=True).tail(15)
    
    importance.plot(kind='barh', color='steelblue')
    plt.title(f'{model_name} - Top 15 Feature Importance', fontsize=14)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    filename = model_name.lower().replace(' ', '_')
    plt.savefig(f'reports/figures/{filename}_feature_importance.png', dpi=150)
    plt.show()
    print(f"✅ Feature importance plot saved!")

def save_models(rf_model, xgb_model, le):
    """Save trained models to disk."""
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    joblib.dump(xgb_model, 'models/xgboost_model.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
    print("\n💾 Models saved:")
    print("   ✅ models/random_forest_model.pkl")
    print("   ✅ models/xgboost_model.pkl")
    print("   ✅ models/label_encoder.pkl")

def compare_models(rf_acc, xgb_acc):
    """Compare both models visually."""
    plt.figure(figsize=(8, 5))
    
    models = ['Random Forest', 'XGBoost']
    accuracies = [rf_acc * 100, xgb_acc * 100]
    colors = ['steelblue', 'orange']
    
    bars = plt.bar(models, accuracies, color=colors, width=0.4)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() - 2,
            f'{acc:.2f}%',
            ha='center',
            va='top',
            color='white',
            fontsize=14,
            fontweight='bold'
        )
    
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('reports/figures/model_comparison.png', dpi=150)
    plt.show()
    print("✅ Model comparison plot saved!")

if __name__ == "__main__":
    print("="*50)
    print("  FLOOD RISK — ML MODEL TRAINING")
    print("="*50)
    
    # Load data
    df = load_features()
    
    # Prepare data
    X_train, X_test, y_train, y_test, le = prepare_data(df)
    
    # Train models
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    # Evaluate models
    rf_acc, rf_pred = evaluate_model(
        rf_model, X_test, y_test, "Random Forest", le)
    xgb_acc, xgb_pred = evaluate_model(
        xgb_model, X_test, y_test, "XGBoost", le)
    
    # Plot confusion matrices
    plot_confusion_matrix(y_test, rf_pred, "Random Forest", le)
    plot_confusion_matrix(y_test, xgb_pred, "XGBoost", le)
    
    # Plot feature importance
    plot_feature_importance(rf_model, X_train, "Random Forest")
    plot_feature_importance(xgb_model, X_train, "XGBoost")
    
    # Compare models
    compare_models(rf_acc, xgb_acc)
    
    # Save models
    save_models(rf_model, xgb_model, le)
    
    print("\n" + "="*50)
    print("🎉 PHASE 4 COMPLETE!")
    print(f"   Random Forest Accuracy: {rf_acc*100:.2f}%")
    print(f"   XGBoost Accuracy:       {xgb_acc*100:.2f}%")
    print("="*50)