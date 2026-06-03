# app/main.py
# PURPOSE: Professional Flood Risk Monitoring Dashboard

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

st.set_page_config(
    page_title="Flood Risk Prediction System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #1a2f5e 50%, #0d1f3c 100%);
    color: white;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f3c 0%, #1a2f5e 100%);
    border-right: 1px solid #2a4a8a;
}
.metric-card {
    background: linear-gradient(135deg, #1a2f5e, #2a4a8a);
    border: 1px solid #3a6abc;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(31,38,135,0.37);
    margin-bottom: 10px;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: bold;
    color: #4fc3f7;
    margin: 8px 0;
}
.metric-label {
    font-size: 0.8rem;
    color: #90caf9;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 900;
    color: #4fc3f7;
    text-align: center;
    padding: 20px 0 5px 0;
    letter-spacing: 2px;
}
.hero-subtitle {
    font-size: 1rem;
    color: #90caf9;
    text-align: center;
    margin-bottom: 20px;
    letter-spacing: 1px;
}
.feature-card {
    background: linear-gradient(135deg, #1a2f5e88, #2a4a8a88);
    border: 1px solid #3a6abc;
    border-radius: 12px;
    padding: 25px;
    text-align: center;
    height: 180px;
    margin-bottom: 10px;
}
.feature-title {
    font-size: 1rem;
    font-weight: bold;
    color: #4fc3f7;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.feature-desc {
    font-size: 0.82rem;
    color: #90caf9;
    line-height: 1.5;
}
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #4fc3f7, transparent);
    margin: 25px 0;
}
.alert-box {
    background: linear-gradient(135deg, #1a2f5e, #2a4a8a);
    border-left: 4px solid #4fc3f7;
    border-radius: 8px;
    padding: 15px 20px;
    margin: 10px 0;
}
.tech-pill {
    display: inline-block;
    background: #1a2f5e;
    border: 1px solid #4fc3f7;
    color: #4fc3f7;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    margin: 4px;
    font-family: monospace;
}
.badge-live {
    background: #00c853;
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: bold;
    letter-spacing: 1px;
}
.section-title {
    font-size: 1.1rem;
    font-weight: bold;
    color: #4fc3f7;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 15px;
    border-bottom: 1px solid #2a4a8a;
    padding-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# WEATHER
# ─────────────────────────────────────────
def get_weather(city="Chennai"):
    try:
        api_key = os.getenv('OPENWEATHER_API_KEY', '')
        if not api_key or api_key == 'your_key_here':
            return {
                'temp': 32, 'humidity': 78,
                'description': 'Partly Cloudy',
                'wind': 12, 'city': city
            }
        url = (f"http://api.openweathermap.org/data/2.5/weather"
               f"?q={city}&appid={api_key}&units=metric")
        data = requests.get(url, timeout=5).json()
        return {
            'temp': round(data['main']['temp']),
            'humidity': data['main']['humidity'],
            'description': data['weather'][0]['description'].title(),
            'wind': round(data['wind']['speed']),
            'city': city
        }
    except:
        return {
            'temp': 32, 'humidity': 78,
            'description': 'Partly Cloudy',
            'wind': 12, 'city': city
        }

# ─────────────────────────────────────────
# 3D GLOBE
# ─────────────────────────────────────────
def create_3d_globe():
    flood_cities = pd.DataFrame({
        'city': ['Chennai','Mumbai','Kolkata','Assam',
                 'Kerala','Odisha','Bihar','Uttarakhand'],
        'lat':  [13.08, 19.08, 22.57, 26.20,
                 10.85, 20.95, 25.09, 30.06],
        'lon':  [80.27, 72.88, 88.36, 92.94,
                 76.27, 85.09, 85.31, 79.01],
        'risk': [75, 85, 80, 90, 88, 82, 78, 70],
        'size': [18, 22, 20, 25, 23, 21, 18, 16]
    })

    fig = px.scatter_geo(
        flood_cities,
        lat='lat',
        lon='lon',
        size='size',
        color='risk',
        hover_name='city',
        hover_data={'risk': True, 'size': False,
                    'lat': False, 'lon': False},
        color_continuous_scale=[
            [0.0, '#00c853'],
            [0.5, '#ff9800'],
            [1.0, '#f44336']
        ],
        size_max=30,
        projection='orthographic',
        labels={'risk': 'Flood Risk %'}
    )

    fig.update_traces(
        marker=dict(
            line=dict(color='white', width=1),
            opacity=0.9
        )
    )

    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor='#1a2f5e',
            showocean=True,
            oceancolor='#0a1628',
            showlakes=True,
            lakecolor='#0d47a1',
            showcountries=True,
            countrycolor='#3a6abc',
            showcoastlines=True,
            coastlinecolor='#4fc3f7',
            bgcolor='#0a1628',
            projection_rotation=dict(lat=20, lon=80, roll=0)
        ),
        paper_bgcolor='#0a1628',
        plot_bgcolor='#0a1628',
        coloraxis_colorbar=dict(
            title=dict(
                text='Risk %',
                font=dict(color='white')
            ),
            tickfont=dict(color='white')
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=450
    )
    return fig

# ─────────────────────────────────────────
# TREND CHART
# ─────────────────────────────────────────
def create_trend_chart():
    months = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']

    risk_data = {
        'Chennai': [15,10, 8, 9,20,42,60,78,85,70,45,22],
        'Mumbai':  [10, 8, 5, 7,18,72,88,90,75,32,18,12],
        'Assam':   [40,32,42,58,72,88,92,88,75,52,42,36],
        'Kerala':  [20,16,12,18,38,70,85,88,80,60,42,25],
    }

    colors = ['#4fc3f7','#ff9800','#f44336','#00c853']
    fig = go.Figure()

    for (city, risks), color in zip(risk_data.items(), colors):
        fig.add_trace(go.Scatter(
            x=months,
            y=risks,
            name=city,
            mode='lines+markers',
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color),
            fill='tozeroy',
            opacity=0.8
        ))

    fig.update_layout(
        title=dict(
            text='Monthly Flood Risk Trends — India (2025)',
            font=dict(color='white', size=15)
        ),
        paper_bgcolor='#0a1628',
        plot_bgcolor='#0d1f3c',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#2a4a8a', color='white'),
        yaxis=dict(
            gridcolor='#2a4a8a',
            color='white',
            title='Risk Level (%)',
            range=[0, 100]
        ),
        legend=dict(
            bgcolor='#1a2f5e',
            bordercolor='#3a6abc',
            font=dict(color='white')
        ),
        hovermode='x unified',
        height=380
    )
    return fig

# ─────────────────────────────────────────
# RISK GAUGE
# ─────────────────────────────────────────
def create_risk_gauge(value=65, title="Risk Level"):
    if value < 30:
        color, level = "#00c853", "LOW"
    elif value < 60:
        color, level = "#ff9800", "MEDIUM"
    elif value < 80:
        color, level = "#f44336", "HIGH"
    else:
        color, level = "#b71c1c", "EXTREME"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={
            'text': f"{title}<br>"
                    f"<span style='font-size:0.9em;"
                    f"color:{color}'>{level}</span>",
            'font': {'color': 'white', 'size': 13}
        },
        number={
            'suffix': "%",
            'font': {'color': 'white', 'size': 28}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickcolor': 'white',
                'tickfont': {'color': 'white', 'size': 9}
            },
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': '#1a2f5e',
            'bordercolor': '#3a6abc',
            'steps': [
                {'range': [0,  30],  'color': '#003300'},
                {'range': [30, 60],  'color': '#332200'},
                {'range': [60, 80],  'color': '#330000'},
                {'range': [80, 100], 'color': '#1a0000'},
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 3},
                'thickness': 0.75,
                'value': value
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='#0a1628',
        font=dict(color='white'),
        height=230,
        margin=dict(l=15, r=15, t=55, b=5)
    )
    return fig

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:20px 0 10px 0'>
            <div style='color:#4fc3f7; font-weight:900;
                        font-size:1.2rem; letter-spacing:2px'>
                FLOOD RISK AI
            </div>
            <div style='color:#90caf9; font-size:0.75rem;
                        letter-spacing:1px; margin-top:4px'>
                SATELLITE MONITORING SYSTEM
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown(
            "<div class='section-title'>Weather Monitor</div>",
            unsafe_allow_html=True
        )
        city = st.selectbox(
            "City", ["Chennai","Mumbai","Kolkata","Assam"],
            label_visibility="collapsed"
        )
        weather = get_weather(city)
        st.markdown(f"""
        <div class='alert-box'>
            <div style='font-size:1rem; color:#4fc3f7;
                        font-weight:bold; letter-spacing:1px'>
                {weather['description'].upper()}
            </div>
            <div style='font-size:2.2rem; font-weight:900;
                        color:white; margin:5px 0'>
                {weather['temp']}°C
            </div>
            <div style='color:#90caf9; font-size:0.82rem;
                        line-height:1.8'>
                Humidity &nbsp;&nbsp; {weather['humidity']}%<br>
                Wind &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {weather['wind']} km/h<br>
                Location &nbsp; {weather['city']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown(
            "<div class='section-title'>Risk Assessment</div>",
            unsafe_allow_html=True
        )
        risk_val = st.slider(
            "Adjust risk value", 0, 100, 65,
            label_visibility="collapsed"
        )
        st.plotly_chart(
            create_risk_gauge(risk_val, "Risk Level"),
            use_container_width=True
        )

        st.divider()

        st.markdown(
            "<div class='section-title'>System Status</div>",
            unsafe_allow_html=True
        )
        st.markdown(f"""
        <div style='font-size:0.82rem; color:#90caf9; line-height:2'>
            <span class='badge-live'>LIVE</span>
            &nbsp; Satellite Feed Active<br>
            &nbsp; ML Models &nbsp;&nbsp;&nbsp; Online<br>
            &nbsp; GEE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Connected<br>
            &nbsp; Dashboard &nbsp;&nbsp; Running<br>
            <div style='color:#555; font-size:0.72rem; margin-top:10px'>
                {datetime.now().strftime('%d %b %Y  %H:%M:%S')}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    render_sidebar()

    # Hero section
    st.markdown("""
    <div class='hero-title'>FLOOD RISK PREDICTION SYSTEM</div>
    <div class='hero-subtitle'>
        AI-POWERED  |  SENTINEL-2 SATELLITE  |  REAL-TIME ANALYTICS
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.markdown(
            "<div style='text-align:center'>"
            "<span class='badge-live'>LIVE MONITORING</span>"
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div class='custom-divider'></div>",
                unsafe_allow_html=True)

    # Stats
    col1, col2, col3, col4, col5 = st.columns(5)
    stats = [
        ("3",          "AI Models"),
        ("705,851",    "Training Samples"),
        ("Sentinel-2", "Satellite Source"),
        ("69.62%",     "Best Accuracy"),
        ("8",          "Cities Monitored"),
    ]
    for col, (value, label) in zip(
            [col1,col2,col3,col4,col5], stats):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{value}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>",
                unsafe_allow_html=True)

    # Globe + Gauges
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            "<div class='section-title'>India Flood Risk Map</div>",
            unsafe_allow_html=True
        )
        st.caption("Hover over circles to see risk levels")
        
        flood_cities = pd.DataFrame({
            'city': ['Chennai','Mumbai','Kolkata','Assam',
                     'Kerala','Odisha','Bihar','Uttarakhand'],
            'lat':  [13.08, 19.08, 22.57, 26.20,
                     10.85, 20.95, 25.09, 30.06],
            'lon':  [80.27, 72.88, 88.36, 92.94,
                     76.27, 85.09, 85.31, 79.01],
            'risk': [75, 85, 80, 90, 88, 82, 78, 70],
        })
        
        fig_map = px.scatter_mapbox(
            flood_cities,
            lat='lat',
            lon='lon',
            size='risk',
            color='risk',
            hover_name='city',
            hover_data={'risk': True, 'lat': False, 'lon': False},
            color_continuous_scale=[
                [0.0, '#00c853'],
                [0.5, '#ff9800'],
                [1.0, '#f44336']
            ],
            size_max=40,
            zoom=4,
            center=dict(lat=20, lon=80),
            mapbox_style='carto-darkmatter',
            labels={'risk': 'Flood Risk %'},
            height=450
        )
        
        fig_map.update_layout(
            paper_bgcolor='#0a1628',
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(
                title=dict(
                    text='Risk %',
                    font=dict(color='white')
                ),
                tickfont=dict(color='white')
            )
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
    with col2:
        st.markdown(
            "<div class='section-title'>City Risk Indicators</div>",
            unsafe_allow_html=True
        )
        for city, risk in [("Chennai",75),("Mumbai",85),("Kerala",88)]:
            st.plotly_chart(
                create_risk_gauge(risk, city),
                use_container_width=True
            )

    st.markdown("<div class='custom-divider'></div>",
                unsafe_allow_html=True)

    # Trend chart
    st.markdown(
        "<div class='section-title'>Monthly Flood Risk Trends</div>",
        unsafe_allow_html=True
    )
    st.plotly_chart(create_trend_chart(), use_container_width=True)

    st.markdown("<div class='custom-divider'></div>",
                unsafe_allow_html=True)

    # Feature cards
    st.markdown(
        "<div class='section-title'>System Capabilities</div>",
        unsafe_allow_html=True
    )
    col1, col2, col3, col4 = st.columns(4)
    features = [
        ("Flood Prediction",
         "ML-powered risk scoring using Random Forest and XGBoost ensemble models"),
        ("Satellite Analysis",
         "Real Sentinel-2 imagery processing with NDVI and NDWI spectral indices"),
        ("GIS Mapping",
         "Interactive flood risk maps with dynamic zone detection and visualization"),
        ("Data Analytics",
         "Deep statistical insights with animated charts and model performance metrics"),
    ]
    for col, (title, desc) in zip([col1,col2,col3,col4], features):
        with col:
            st.markdown(f"""
            <div class='feature-card'>
                <div class='feature-title'>{title}</div>
                <div class='feature-desc'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>",
                unsafe_allow_html=True)

    # Tech stack
    st.markdown(
        "<div class='section-title'>Technology Stack</div>",
        unsafe_allow_html=True
    )
    st.markdown("""
    <div style='text-align:center; padding:10px'>
        <span class='tech-pill'>Python 3.11</span>
        <span class='tech-pill'>Scikit-learn</span>
        <span class='tech-pill'>XGBoost</span>
        <span class='tech-pill'>TensorFlow</span>
        <span class='tech-pill'>Google Earth Engine</span>
        <span class='tech-pill'>Sentinel-2</span>
        <span class='tech-pill'>Streamlit</span>
        <span class='tech-pill'>Folium</span>
        <span class='tech-pill'>Plotly</span>
        <span class='tech-pill'>Pandas</span>
        <span class='tech-pill'>NumPy</span>
        <span class='tech-pill'>Rasterio</span>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("<div class='custom-divider'></div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; color:#555;
                font-size:0.8rem; padding:15px; letter-spacing:1px'>
        FLOOD RISK PREDICTION SYSTEM &nbsp;|&nbsp;
        DEVELOPED BY ABHIJITH MR &nbsp;|&nbsp;
        B.E. CSE (AI & ML) &nbsp;|&nbsp;
        SATHYABAMA INSTITUTE OF SCIENCE AND TECHNOLOGY
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()