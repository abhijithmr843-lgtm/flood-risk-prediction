# app/pages/analysis_page.py
# PURPOSE: Data analysis and model performance page

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(
    page_title="Data Analysis",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load processed dataset."""
    try:
        df = pd.read_csv('data/processed/flood_features.csv')
        return df
    except:
        return None

def main():
    st.title("📊 Flood Data Analysis")
    st.markdown("Explore flood patterns, feature importance and model performance.")
    st.divider()
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Data not found! Please run Phase 3 first.")
        return
    
    st.success(f"✅ Dataset loaded: {df.shape[0]:,} samples, {df.shape[1]} features")
    
    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs([
        "📈 Flood Distribution",
        "🔥 Feature Analysis",
        "🤖 Model Performance"
    ])
    
    with tab1:
        st.subheader("Flood Risk Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk category pie chart
            risk_counts = df['RiskCategory'].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Category Distribution",
                color_discrete_map={
                    'Low': 'green',
                    'Medium': 'orange',
                    'High': 'red',
                    'Very High': 'darkred'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Flood probability histogram
            fig = px.histogram(
                df,
                x='FloodProbability',
                nbins=50,
                title="Flood Probability Distribution",
                color_discrete_sequence=['steelblue']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Probability", f"{df['FloodProbability'].mean():.3f}")
        col2.metric("Max Probability", f"{df['FloodProbability'].max():.3f}")
        col3.metric("Min Probability", f"{df['FloodProbability'].min():.3f}")
        col4.metric("Std Deviation", f"{df['FloodProbability'].std():.3f}")
    
    with tab2:
        st.subheader("Feature Importance Analysis")
        
        # Correlation with flood probability
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()['FloodProbability'].sort_values(ascending=True)
        
        fig = px.bar(
            x=corr.values,
            y=corr.index,
            orientation='h',
            title="Feature Correlation with Flood Probability",
            color=corr.values,
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature scatter plot
        st.subheader("Feature vs Flood Probability")
        feature = st.selectbox(
            "Select feature to analyze:",
            ['MonsoonIntensity', 'ClimateRisk', 'OverallRiskScore',
             'EnvironmentalRisk', 'HumanActivityRisk', 'InfrastructureRisk']
        )
        
        fig = px.scatter(
            df.sample(5000),
            x=feature,
            y='FloodProbability',
            color='RiskCategory',
            title=f"{feature} vs Flood Probability",
            opacity=0.5
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Model Performance Comparison")
        
        # Model comparison
        models_data = {
            'Model': ['Random Forest', 'XGBoost', 'CNN'],
            'Accuracy': [60.54, 69.62, 100.00],
            'Type': ['Classical ML', 'Gradient Boosting', 'Deep Learning']
        }
        
        models_df = pd.DataFrame(models_data)
        
        fig = px.bar(
            models_df,
            x='Model',
            y='Accuracy',
            color='Type',
            title="Model Accuracy Comparison",
            text='Accuracy'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(yaxis_range=[0, 110])
        st.plotly_chart(fig, use_container_width=True)
        
        # Saved plots
        st.subheader("📸 Saved Analysis Plots")
        
        plots = {
            "Flood Distribution": "reports/figures/flood_distribution.png",
            "Correlation Heatmap": "reports/figures/correlation_heatmap.png",
            "Feature Relationships": "reports/figures/feature_relationships.png",
            "CNN Training History": "reports/figures/cnn_training_history.png"
        }
        
        for plot_name, plot_path in plots.items():
            if os.path.exists(plot_path):
                st.image(plot_path, caption=plot_name, use_column_width=True)

if __name__ == "__main__":
    main()