# app/pages/map_page.py
# PURPOSE: Satellite map visualization page

import streamlit as st
import folium
from streamlit_folium import st_folium
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(
    page_title="Satellite Map",
    page_icon="🗺️",
    layout="wide"
)

def create_flood_map(latitude, longitude, zoom=8):
    """Create interactive flood risk map."""
    
    # Create base map
    m = folium.Map(
        location=[latitude, longitude],
        zoom_start=zoom,
        tiles='CartoDB positron'
    )
    
    # Add satellite layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite View',
        overlay=False
    ).add_to(m)
    
    # Add marker for selected location
    folium.Marker(
        location=[latitude, longitude],
        popup=folium.Popup(
            f"<b>Selected Location</b><br>Lat: {latitude}<br>Lon: {longitude}",
            max_width=200
        ),
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Add flood risk zones (sample circles)
    flood_zones = [
        (latitude + 0.1, longitude + 0.1, 'High Risk Zone', 'red'),
        (latitude - 0.1, longitude + 0.2, 'Medium Risk Zone', 'orange'),
        (latitude + 0.2, longitude - 0.1, 'Low Risk Zone', 'green'),
    ]
    
    for lat, lon, label, color in flood_zones:
        folium.Circle(
            location=[lat, lon],
            radius=5000,
            color=color,
            fill=True,
            fill_opacity=0.3,
            popup=label
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def main():
    st.title("🗺️ Satellite Flood Risk Map")
    st.markdown("Interactive map showing flood risk zones and satellite imagery.")
    st.divider()
    
    # Location selector
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📍 Select Location")
        
        # Preset locations
        location = st.selectbox(
            "Choose a city:",
            [
                "Chennai, Tamil Nadu",
                "Mumbai, Maharashtra",
                "Kolkata, West Bengal",
                "Kerala Coast",
                "Assam Valley",
                "Custom Location"
            ]
        )
        
        # Location coordinates
        locations = {
            "Chennai, Tamil Nadu": (13.0827, 80.2707),
            "Mumbai, Maharashtra": (19.0760, 72.8777),
            "Kolkata, West Bengal": (22.5726, 88.3639),
            "Kerala Coast": (10.8505, 76.2711),
            "Assam Valley": (26.2006, 92.9376),
        }
        
        if location == "Custom Location":
            lat = st.number_input("Latitude", value=13.0827, format="%.4f")
            lon = st.number_input("Longitude", value=80.2707, format="%.4f")
        else:
            lat, lon = locations[location]
            st.info(f"📍 Lat: {lat}, Lon: {lon}")
        
        zoom = st.slider("Zoom Level", 5, 15, 8)
        
        st.divider()
        
        # Risk legend
        st.subheader("🎨 Risk Legend")
        st.markdown("🔴 High Risk Zone")
        st.markdown("🟠 Medium Risk Zone")
        st.markdown("🟢 Low Risk Zone")
        st.markdown("📍 Selected Location")
        
        st.divider()
        
        # Satellite info
        st.subheader("🛰️ Satellite Info")
        st.info("""
        **Data Source:** ESA Sentinel-2
        **Resolution:** 10 meters
        **Update:** Every 5 days
        **Bands:** 13 spectral bands
        """)
    
    with col2:
        st.subheader("🗺️ Interactive Map")
        
        # Create and display map
        flood_map = create_flood_map(lat, lon, zoom)
        st_folium(flood_map, width=800, height=600)
        
        st.caption("🌊 Red circles = High flood risk | Orange = Medium | Green = Low risk")

if __name__ == "__main__":
    main()