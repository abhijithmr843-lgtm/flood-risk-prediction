# app/pages/prediction_page.py
# PURPOSE: Flood risk prediction page

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(
    page_title="Flood Prediction",
    page_icon="",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load trained ML models."""
    try:
        rf_model = joblib.load('models/random_forest_model.pkl')
        xgb_model = joblib.load('models/xgboost_model.pkl')
        le = joblib.load('models/label_encoder.pkl')
        return rf_model, xgb_model, le
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def get_risk_color(risk_category):
    """Return color based on risk level."""
    colors = {
        'Low': '',
        'Medium': '',
        'High': '',
        'Very High': ''
    }
    return colors.get(risk_category, '')

def main():
    st.title("Flood Risk Prediction")
    st.markdown("Enter the environmental parameters to get flood risk assessment.")
    st.divider()

    # Load models
    rf_model, xgb_model, le = load_models()

    if rf_model is None:
        st.error("Models not found! Please run Phase 4 first.")
        return

    st.success("ML Models loaded successfully!")

    # Input form
    st.subheader("Environmental Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("** Climate Factors**")
        monsoon = st.slider("Monsoon Intensity", 0, 10, 5)
        climate_change = st.slider("Climate Change Impact", 0, 10, 5)
        topography = st.slider("Topography & Drainage", 0, 10, 5)
        river_mgmt = st.slider("River Management", 0, 10, 5)
        coastal = st.slider("Coastal Vulnerability", 0, 10, 5)
        watersheds = st.slider("Watersheds", 0, 10, 5)
        landslides = st.slider("Landslides Risk", 0, 10, 5)

    with col2:
        st.markdown("** Environmental Factors**")
        deforestation = st.slider("Deforestation Level", 0, 10, 5)
        wetland_loss = st.slider("Wetland Loss", 0, 10, 5)
        siltation = st.slider("Siltation Level", 0, 10, 5)
        agri = st.slider("Agricultural Practices", 0, 10, 5)
        encroach = st.slider("Encroachments", 0, 10, 5)
        popuation = st.slider("Population Score", 0, 10, 5)
        political = st.slider("Political Factors", 0, 10, 5)

    with col3:
        st.markdown("** Infrastructure Factors**")
        dams = st.slider("Dams Quality", 0, 10, 5)
        drainage = st.slider("Drainage Systems", 0, 10, 5)
        infra = st.slider("Deteriorating Infrastructure", 0, 10, 5)
        disaster_prep = st.slider("Disaster Preparedness", 0, 10, 5)
        urbanization = st.slider("Urbanization Level", 0, 10, 5)
        planning = st.slider("Inadequate Planning", 0, 10, 5)

    st.divider()

    # Predict button
    if st.button("Predict Flood Risk", type="primary", use_container_width=True):

        # Calculate engineered features
        infra_risk = (dams + drainage + infra) / 3
        env_risk = (deforestation + wetland_loss + siltation) / 3
        human_risk = (urbanization + encroach + agri + planning) / 4
        climate_risk = (monsoon + climate_change) / 2
        overall_risk = (infra_risk + env_risk + human_risk + climate_risk) / 4
        high_monsoon = 1 if monsoon > 5 else 0

        # Create input dataframe
        input_data = pd.DataFrame([{
            'MonsoonIntensity': monsoon,
            'TopographyDrainage': topography,
            'RiverManagement': river_mgmt,
            'Deforestation': deforestation,
            'Urbanization': urbanization,
            'ClimateChange': climate_change,
            'DamsQuality': dams,
            'Siltation': siltation,
            'AgriculturalPractices': agri,
            'Encroachments': encroach,
            'IneffectiveDisasterPreparedness': disaster_prep,
            'DrainageSystems': drainage,
            'CoastalVulnerability': coastal,
            'Landslides': landslides,
            'Watersheds': watersheds,
            'DeterioratingInfrastructure': infra,
            'PopulationScore': popuation,
            'WetlandLoss': wetland_loss,
            'InadequatePlanning': planning,
            'PoliticalFactors': political,
            'InfrastructureRisk': infra_risk,
            'EnvironmentalRisk': env_risk,
            'HumanActivityRisk': human_risk,
            'ClimateRisk': climate_risk,
            'OverallRiskScore': overall_risk,
            'HighMonsoon': high_monsoon
        }])

        # Get predictions
        rf_pred = le.inverse_transform(rf_model.predict(input_data))[0]
        xgb_pred = le.inverse_transform(xgb_model.predict(input_data))[0]

        # Get probabilities
        rf_proba = rf_model.predict_proba(input_data)[0]
        xgb_proba = xgb_model.predict_proba(input_data)[0]

        # Display results
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Random Forest")
            emoji = get_risk_color(rf_pred)
            st.markdown(f"## {emoji} {rf_pred} Risk")

            # Show probabilities
            prob_df = pd.DataFrame({
                'Risk Level': le.classes_,
                'Probability': rf_proba
            })
            st.bar_chart(prob_df.set_index('Risk Level'))

        with col2:
            st.markdown("### XGBoost")
            emoji = get_risk_color(xgb_pred)
            st.markdown(f"## {emoji} {xgb_pred} Risk")

            # Show probabilities
            prob_df = pd.DataFrame({
                'Risk Level': le.classes_,
                'Probability': xgb_proba
            })
            st.bar_chart(prob_df.set_index('Risk Level'))

        st.divider()

        # Overall risk score
        st.subheader("Overall Risk Score")
        risk_pct = overall_risk * 10
        st.progress(int(risk_pct))
        st.markdown(f"### Overall Risk Score: **{risk_pct:.1f}/100**")

        # Recommendations
        st.subheader("Recommendations")
        if overall_risk > 7:
            st.error("""
             **EXTREME RISK — Immediate Action Required!**
            - Evacuate low-lying areas immediately
            - Alert local disaster management
            - Monitor water levels continuously
            """)
        elif overall_risk > 5:
            st.warning("""
             **HIGH RISK — Take Precautions!**
            - Prepare emergency supplies
            - Monitor weather forecasts
            - Clear drainage systems
            """)
        else:
            st.success("""
             **LOWER RISK — Stay Prepared!**
            - Maintain drainage systems
            - Monitor monsoon forecasts
            - Keep emergency contacts ready
            """)

if __name__ == "__main__":
    main()