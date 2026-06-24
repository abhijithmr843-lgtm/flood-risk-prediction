# app/pages/map_page.py
# PURPOSE: Advanced GIS flood risk map page

import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from src.visualization.map_generator import (
    get_india_flood_risk_data,
    create_full_flood_risk_map
)
from streamlit_folium import st_folium

st.set_page_config(
    page_title="GIS Flood Risk Map",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #1a2f5e 50%, #0d1f3c 100%);
    color: white;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f3c 0%, #1a2f5e 100%);
}
.section-title {
    font-size: 1.1rem;
    font-weight: bold;
    color: #4fc3f7;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 15px;
}
.district-card {
    background: linear-gradient(135deg, #1a2f5e, #2a4a8a);
    border: 1px solid #3a6abc;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_map():
    """Load the GIS map (cached for performance)."""
    return create_full_flood_risk_map()


def main():
    st.markdown(
        "<div class='section-title' style='font-size:1.8rem'>"
        "GIS FLOOD RISK MAP</div>",
        unsafe_allow_html=True
    )
    st.caption("Interactive heatmap of flood-prone districts across India")

    st.divider()

    df = get_india_flood_risk_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Districts Monitored", len(df))
    col2.metric("Avg Risk Score", f"{df['risk_score'].mean():.1f}")
    col3.metric("Highest Risk", df.loc[df['risk_score'].idxmax(), 'district'])
    col4.metric("Total Population at Risk", f"{df['population_affected'].sum():,}")

    st.divider()

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            "<div class='section-title'>Interactive Heatmap</div>",
            unsafe_allow_html=True
        )
        flood_map = load_map()
        st_folium(
            flood_map,
            width=None,
            height=600,
            returned_objects=[],
            key="flood_risk_map"
        )

    with col2:
        st.markdown(
            "<div class='section-title'>Top 5 Risk Districts</div>",
            unsafe_allow_html=True
        )

        top5 = df.nlargest(5, 'risk_score')

        for _, row in top5.iterrows():
            if row['risk_score'] >= 85:
                color = '#f44336'
            elif row['risk_score'] >= 75:
                color = '#ff9800'
            else:
                color = '#4fc3f7'

            st.markdown(f"""
            <div class='district-card'>
                <div style='color:{color}; font-weight:bold; font-size:1rem'>
                    {row['district']}
                </div>
                <div style='color:#90caf9; font-size:0.8rem'>
                    {row['state']}
                </div>
                <div style='color:white; font-size:1.3rem; font-weight:bold; margin-top:5px'>
                    {row['risk_score']}/100
                </div>
                <div style='color:#90caf9; font-size:0.75rem'>
                    {row['population_affected']:,} people at risk
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    with st.expander("View All District Data"):
        display_df = df.sort_values('risk_score', ascending=False)
        display_df.columns = ['District', 'State', 'Latitude', 'Longitude',
                                'Risk Score', 'Population Affected']
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


if __name__ == "__main__":
    main()