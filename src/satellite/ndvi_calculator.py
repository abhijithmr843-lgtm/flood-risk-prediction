# src/satellite/ndvi_calculator.py
# PURPOSE: Calculate NDVI and NDWI from Sentinel-2 images

import ee
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from gee_connector import initialize_gee, get_sentinel2_image

def calculate_ndvi(image):
    """
    Calculate NDVI - Normalized Difference Vegetation Index
    
    Formula: NDVI = (NIR - Red) / (NIR + Red)
    
    Sentinel-2 Bands:
    - B4 = Red band
    - B8 = Near Infrared (NIR) band
    
    NDVI Values:
    - (-1 to 0)  = Water, bare soil, built-up
    - (0 to 0.2) = Sparse vegetation
    - (0.2 to 0.5) = Moderate vegetation
    - (0.5 to 1)   = Dense vegetation
    """
    print("\n🌿 Calculating NDVI...")
    
    # Get NIR (B8) and Red (B4) bands
    nir = image.select('B8')   # Near Infrared
    red = image.select('B4')   # Red
    
    # Calculate NDVI using normalizedDifference function
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    print("   ✅ NDVI calculated!")
    return ndvi

def calculate_ndwi(image):
    """
    Calculate NDWI - Normalized Difference Water Index
    
    Formula: NDWI = (Green - NIR) / (Green + NIR)
    
    Sentinel-2 Bands:
    - B3 = Green band
    - B8 = Near Infrared (NIR) band
    
    NDWI Values:
    - NDWI > 0  = Water bodies (rivers, lakes, floods!)
    - NDWI < 0  = Land, vegetation
    """
    print("\n💧 Calculating NDWI...")
    
    # Get Green (B3) and NIR (B8) bands
    green = image.select('B3')  # Green
    nir = image.select('B8')    # Near Infrared
    
    # Calculate NDWI
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    
    print("   ✅ NDWI calculated!")
    return ndwi

def extract_water_bodies(ndwi, region):
    """
    Extract water bodies from NDWI image.
    Areas where NDWI > 0 are water.
    """
    print("\n🌊 Extracting water bodies...")
    
    # Threshold: NDWI > 0 means water
    water_mask = ndwi.gt(0).rename('WaterMask')
    
    # Calculate water area statistics
    stats = water_mask.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=100,          # 100 meter resolution
        maxPixels=1e9
    )
    
    water_pixels = stats.get('WaterMask').getInfo()
    
    if water_pixels:
        # Each pixel = 100m x 100m = 0.01 km²
        water_area_km2 = water_pixels * 0.01
        print(f"   Water pixels detected: {water_pixels:,}")
        print(f"   Estimated water area: {water_area_km2:.2f} km²")
    
    print("   ✅ Water bodies extracted!")
    return water_mask

def get_ndvi_statistics(ndvi, region):
    """Get NDVI statistics for the region."""
    print("\n📊 Getting NDVI Statistics...")
    
    stats = ndvi.reduceRegion(
        reducer=ee.Reducer.mean()
            .combine(ee.Reducer.min(), sharedInputs=True)
            .combine(ee.Reducer.max(), sharedInputs=True),
        geometry=region,
        scale=100,
        maxPixels=1e9
    )
    
    result = stats.getInfo()
    
    mean_ndvi = result.get('NDVI_mean', 0)
    min_ndvi  = result.get('NDVI_min', 0)
    max_ndvi  = result.get('NDVI_max', 0)
    
    print(f"   Mean NDVI: {mean_ndvi:.4f}")
    print(f"   Min NDVI:  {min_ndvi:.4f}")
    print(f"   Max NDVI:  {max_ndvi:.4f}")
    
    # Interpret NDVI
    if mean_ndvi < 0.2:
        health = "⚠️ Low vegetation — High flood risk area"
    elif mean_ndvi < 0.4:
        health = "🟡 Moderate vegetation"
    else:
        health = "✅ Healthy vegetation — Lower flood risk"
    
    print(f"   Interpretation: {health}")
    
    return mean_ndvi, min_ndvi, max_ndvi

def visualize_indices(ndvi, ndwi, region):
    """
    Get visualization URLs for NDVI and NDWI maps.
    These can be displayed in the Streamlit dashboard.
    """
    print("\n🗺️ Generating visualization URLs...")
    
    # NDVI visualization parameters
    ndvi_params = {
        'min': -1,
        'max': 1,
        'palette': ['red', 'yellow', 'green'],
        'dimensions': 512,
        'region': region
    }
    
    # NDWI visualization parameters  
    ndwi_params = {
        'min': -1,
        'max': 1,
        'palette': ['brown', 'white', 'blue'],
        'dimensions': 512,
        'region': region
    }
    
    # Get thumbnail URLs
    ndvi_url = ndvi.getThumbURL(ndvi_params)
    ndwi_url = ndwi.getThumbURL(ndwi_params)
    
    print(f"   ✅ NDVI map URL generated!")
    print(f"   ✅ NDWI map URL generated!")
    
    return ndvi_url, ndwi_url

# Test when run directly
if __name__ == "__main__":
    print("="*50)
    print("  SATELLITE INDEX ANALYSIS")
    print("="*50)
    
    # Initialize GEE
    initialize_gee()
    
    # Get satellite image for Chennai
    image, region = get_sentinel2_image(
        longitude=80.2707,
        latitude=13.0827,
        start_date='2023-10-01',
        end_date='2023-12-31'
    )
    
    if image:
        # Calculate indices
        ndvi = calculate_ndvi(image)
        ndwi = calculate_ndwi(image)
        
        # Extract water bodies
        water_mask = extract_water_bodies(ndwi, region)
        
        # Get statistics
        mean_ndvi, min_ndvi, max_ndvi = get_ndvi_statistics(ndvi, region)
        
        # Get visualization URLs
        ndvi_url, ndwi_url = visualize_indices(ndvi, ndwi, region)
        
        print("\n" + "="*50)
        print("🎉 SATELLITE ANALYSIS COMPLETE!")
        print(f"   NDVI Mean: {mean_ndvi:.4f}")
        print(f"\n📌 NDVI Map URL:")
        print(f"   {ndvi_url}")
        print(f"\n📌 NDWI Map URL:")
        print(f"   {ndwi_url}")
        print("="*50)