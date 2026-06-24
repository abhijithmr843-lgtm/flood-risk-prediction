# src/satellite/gee_connector.py
# PURPOSE: Connect to Google Earth Engine and fetch Sentinel-2 data

import ee
import os
from dotenv import load_dotenv

load_dotenv()

def initialize_gee():
    """
    Initialize Google Earth Engine connection.
    Must be called before any GEE operations.
    """
    try:
        project_id = os.getenv('GEE_PROJECT_ID', 'flood-risk-prediction-498117')
        ee.Initialize(project=project_id)
        print("Google Earth Engine initialized!")
        return True
    except Exception as e:
        print(f"GEE initialization failed: {e}")
        return False

def get_sentinel2_image(longitude, latitude, start_date, end_date):
    """
    Fetch Sentinel-2 satellite image for a location.

    Args:
        longitude: Location longitude (e.g., 80.27 for Chennai)
        latitude: Location latitude (e.g., 13.08 for Chennai)
        start_date: Start date string (e.g., '2023-01-01')
        end_date: End date string (e.g., '2023-12-31')

    Returns:
        Sentinel-2 image collection
    """
    print(f"\n Fetching Sentinel-2 data...")
    print(f"Location: ({latitude}, {longitude})")
    print(f"Period: {start_date} to {end_date}")

    # Create a point geometry for the location
    point = ee.Geometry.Point([longitude, latitude])

    # Create a 50km buffer around the point
    region = point.buffer(50000)

    # Filter Sentinel-2 image collection
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(region) # Filter by location
        .filterDate(start_date, end_date) # Filter by date
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) # Less than 20% clouds
        .sort('CLOUDY_PIXEL_PERCENTAGE') # Sort by cloud cover
    )

    # Get number of images found
    count = collection.size().getInfo()
    print(f"Images found: {count}")

    if count == 0:
        print("No images found! Try different dates.")
        return None

    # Get the clearest image (least clouds)
    image = collection.first()
    print("Best image selected!")

    return image, region

def get_image_info(image):
    """Get basic information about a satellite image."""
    if image is None:
        return

    print("\n Image Information:")
    info = image.getInfo()

    # Get date
    date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    print(f"Date: {date}")

    # Get cloud cover
    cloud = image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
    print(f"Cloud Cover: {cloud:.1f}%")

    return date, cloud

# Test when run directly
if __name__ == "__main__":
    print("="*50)
    print("GEE CONNECTOR TEST")
    print("="*50)

    # Initialize GEE
    if initialize_gee():
        # Test with Chennai, India coordinates
        # Chennai is flood-prone — perfect for our project!
        image, region = get_sentinel2_image(
            longitude=80.2707,
            latitude=13.0827,
            start_date='2023-10-01',
            end_date='2023-12-31'
        )

        if image:
            get_image_info(image)
            print("\n GEE Connector working perfectly!")