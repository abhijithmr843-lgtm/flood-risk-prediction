# src/satellite/water_extractor.py
# PURPOSE: Detect flood regions from satellite data

import ee
from gee_connector import initialize_gee, get_sentinel2_image
from ndvi_calculator import calculate_ndwi

def detect_flood_regions(before_image, after_image, region):
    """
    Detect flooded areas by comparing
    before and after flood satellite images.
    
    Logic:
    - Calculate NDWI for both images
    - Where water increased = flooded area
    """
    print("\n🌊 Detecting flood regions...")
    
    # Calculate NDWI for both images
    ndwi_before = before_image.normalizedDifference(['B3', 'B8'])
    ndwi_after  = after_image.normalizedDifference(['B3', 'B8'])
    
    # Water mask for both
    water_before = ndwi_before.gt(0)
    water_after  = ndwi_after.gt(0)
    
    # Flooded = water after BUT NOT water before
    flood_mask = water_after.And(water_before.Not()).rename('FloodMask')
    
    print("   ✅ Flood regions detected!")
    return flood_mask

def calculate_flood_area(flood_mask, region):
    """Calculate total flooded area in km²."""
    print("\n📏 Calculating flood area...")
    
    stats = flood_mask.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=100,
        maxPixels=1e9
    )
    
    flood_pixels = stats.get('FloodMask').getInfo()
    
    if flood_pixels:
        flood_area = flood_pixels * 0.01
        print(f"   Flooded pixels: {flood_pixels:,}")
        print(f"   Flooded area: {flood_area:.2f} km²")
        
        # Risk assessment
        if flood_area < 10:
            risk = "🟢 Low flood impact"
        elif flood_area < 50:
            risk = "🟡 Moderate flood impact"
        elif flood_area < 100:
            risk = "🔴 High flood impact"
        else:
            risk = "🚨 Extreme flood impact"
        
        print(f"   Assessment: {risk}")
        return flood_area, risk
    
    return 0, "No flood detected"

def get_flood_visualization(flood_mask, region):
    """Get flood visualization URL."""
    params = {
        'min': 0,
        'max': 1,
        'palette': ['black', 'red'],
        'dimensions': 512,
        'region': region
    }
    
    url = flood_mask.getThumbURL(params)
    print(f"\n✅ Flood map URL generated!")
    return url

if __name__ == "__main__":
    print("="*50)
    print("  FLOOD REGION DETECTION")
    print("="*50)
    
    initialize_gee()
    
    # Before flood image (dry season)
    print("\n📅 Getting PRE-FLOOD image...")
    before_result = get_sentinel2_image(
        longitude=80.2707,
        latitude=13.0827,
        start_date='2023-06-01',
        end_date='2023-08-31'
    )
    
    # After flood image (flood season)
    print("\n📅 Getting POST-FLOOD image...")
    after_result = get_sentinel2_image(
        longitude=80.2707,
        latitude=13.0827,
        start_date='2023-10-01',
        end_date='2023-12-31'
    )
    
    if before_result and after_result:
        before_image, region = before_result
        after_image, _ = after_result
        
        # Detect floods
        flood_mask = detect_flood_regions(
            before_image, after_image, region)
        
        # Calculate area
        flood_area, risk = calculate_flood_area(
            flood_mask, region)
        
        # Get visualization
        flood_url = get_flood_visualization(flood_mask, region)
        
        print("\n" + "="*50)
        print("🎉 FLOOD DETECTION COMPLETE!")
        print(f"   Flooded Area: {flood_area:.2f} km²")
        print(f"   Risk Level: {risk}")
        print(f"\n📌 Flood Map URL:")
        print(f"   {flood_url}")
        print("="*50)