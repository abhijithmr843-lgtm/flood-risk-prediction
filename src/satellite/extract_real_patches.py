# src/satellite/extract_real_patches.py
# PURPOSE: Download real Sentinel-2 patches for known flood events,
# auto-label them using NDWI, and save as a training dataset.

import ee
import os
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from gee_connector import initialize_gee

# ─────────────────────────────────────────
# FLOOD EVENTS — real, documented locations
# Date windows widened to account for monsoon cloud cover,
# which frequently blocks optical satellites like Sentinel-2.
# ─────────────────────────────────────────
FLOOD_EVENTS = [
 {
 "name": "Chennai_2023",
 "lon": 80.2707, "lat": 13.0827,
 "pre_start": "2023-06-01", "pre_end": "2023-08-31",
 "post_start": "2023-10-01", "post_end": "2023-12-31",
 },
 {
 "name": "Assam_2024",
 "lon": 91.7362, "lat": 26.1445,
 "pre_start": "2024-01-01", "pre_end": "2024-03-31",
 "post_start": "2024-06-01", "post_end": "2024-10-31",
 },
 {
 "name": "Kerala_Wayanad_2024",
 "lon": 76.1320, "lat": 11.6854,
 "pre_start": "2024-01-01", "pre_end": "2024-04-30",
 "post_start": "2024-07-01", "post_end": "2024-09-30",
 },
 {
 "name": "Mumbai_2024",
 "lon": 72.8777, "lat": 19.0760,
 "pre_start": "2024-01-01", "pre_end": "2024-03-31",
 "post_start": "2024-06-01", "post_end": "2024-09-30",
 },
 {
 "name": "Mumbai_2023",
 "lon": 72.8777, "lat": 19.0760,
 "pre_start": "2024-01-01", "pre_end": "2024-03-31",
 "post_start": "2023-06-01", "post_end": "2023-09-30",
 },
]

PATCH_SIZE = 32 # pixels per patch side (we upscale to 64 later for CNN)
FINAL_PATCH_SIZE = 64 # size fed into the CNN
REGION_HALF_DEG = 0.25 # roughly 25-28km in each direction from center
THUMB_DIMENSIONS = 1024 # larger thumbnail = more patches per image
CLOUD_THRESHOLD = 45 # max acceptable cloud cover %


def get_best_image(region, start_date, end_date):
 """Get the clearest available Sentinel-2 image for a date range."""
 collection = (
 ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
 .filterBounds(region)
 .filterDate(start_date, end_date)
 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_THRESHOLD))
 .sort('CLOUDY_PIXEL_PERCENTAGE')
 )
 count = collection.size().getInfo()
 if count == 0:
 return None, 0
 return collection.first(), count


def compute_ndwi(image):
 """NDWI = (Green - NIR) / (Green + NIR). High NDWI = water."""
 return image.normalizedDifference(['B3', 'B8']).rename('NDWI')


def get_rgb_thumbnail(image, region, dimensions=THUMB_DIMENSIONS):
 """Fetch an RGB visual thumbnail as a numpy array."""
 vis_params = {
 'bands': ['B4', 'B3', 'B2'], # true color
 'min': 0,
 'max': 3000,
 'region': region,
 'dimensions': dimensions,
 'format': 'png'
 }
 url = image.getThumbURL(vis_params)
 response = requests.get(url, timeout=30)
 img = Image.open(BytesIO(response.content)).convert('RGB')
 return np.array(img)


def get_ndwi_array(ndwi_image, region, dimensions=THUMB_DIMENSIONS):
 """Fetch NDWI as a grayscale array (scaled to 0-255, white = water)."""
 vis_params = {
 'min': -1,
 'max': 1,
 'region': region,
 'dimensions': dimensions,
 'format': 'png',
 'palette': ['000000', 'ffffff'] # black = no water, white = water
 }
 url = ndwi_image.getThumbURL(vis_params)
 response = requests.get(url, timeout=30)
 img = Image.open(BytesIO(response.content)).convert('L') # grayscale
 return np.array(img)


def slice_into_patches(rgb_array, ndwi_array, patch_size=PATCH_SIZE):
 """
 Slice a full image + its NDWI mask into non-overlapping patches.
 Returns list of (rgb_patch, mean_ndwi_value_0_to_1).
 """
 h, w, _ = rgb_array.shape
 patches = []

 for y in range(0, h - patch_size + 1, patch_size):
 for x in range(0, w - patch_size + 1, patch_size):
 rgb_patch = rgb_array[y:y+patch_size, x:x+patch_size, :]
 ndwi_patch = ndwi_array[y:y+patch_size, x:x+patch_size]

 # Skip patches that are mostly black (no data / edge of image)
 if rgb_patch.mean() < 5:
 continue

 mean_water = ndwi_patch.mean() / 255.0 # normalize 0-1
 patches.append((rgb_patch, mean_water))

 return patches


def process_event(event):
 """
 Download pre/post images for one flood event, slice into patches,
 and auto-label based on water increase.
 """
 print(f"\n{'='*60}")
 print(f"PROCESSING: {event['name']}")
 print(f"{'='*60}")

 lon, lat = event['lon'], event['lat']
 region = ee.Geometry.Rectangle([
 lon - REGION_HALF_DEG, lat - REGION_HALF_DEG,
 lon + REGION_HALF_DEG, lat + REGION_HALF_DEG
 ])

 print(f"Fetching PRE-flood image ({event['pre_start']} to {event['pre_end']})...")
 pre_image, pre_count = get_best_image(region, event['pre_start'], event['pre_end'])
 print(f"Available pre-flood images in range: {pre_count}")
 if pre_image is None:
 print("No pre-flood image found. Skipping event.")
 return [], []

 print(f"Fetching POST-flood image ({event['post_start']} to {event['post_end']})...")
 post_image, post_count = get_best_image(region, event['post_start'], event['post_end'])
 print(f"Available post-flood images in range: {post_count}")
 if post_image is None:
 print("No post-flood image found. Skipping event.")
 return [], []

 print("Computing NDWI for both images...")
 pre_ndwi = compute_ndwi(pre_image)
 post_ndwi = compute_ndwi(post_image)

 print("Downloading RGB + NDWI arrays (this takes ~30-60s)...")
 pre_rgb_arr = get_rgb_thumbnail(pre_image, region)
 post_rgb_arr = get_rgb_thumbnail(post_image, region)
 pre_ndwi_arr = get_ndwi_array(pre_ndwi, region)
 post_ndwi_arr = get_ndwi_array(post_ndwi, region)

 print("Slicing into patches...")
 pre_patches = slice_into_patches(pre_rgb_arr, pre_ndwi_arr, PATCH_SIZE)
 post_patches = slice_into_patches(post_rgb_arr, post_ndwi_arr, PATCH_SIZE)

 print(f"Pre-flood patches: {len(pre_patches)}")
 print(f"Post-flood patches: {len(post_patches)}")

 flood_images = []
 no_flood_images = []
 water_increases = []

 n = min(len(pre_patches), len(post_patches))
 for i in range(n):
 pre_rgb, pre_water = pre_patches[i]
 post_rgb, post_water = post_patches[i]

 water_increase = post_water - pre_water
 water_increases.append(water_increase)

 # Loosened thresholds — real data is noisier than synthetic data
 if post_water > 0.2 and water_increase > 0.08:
 flood_images.append(post_rgb)
 elif pre_water < 0.2 and post_water < 0.2:
 no_flood_images.append(pre_rgb)
 no_flood_images.append(post_rgb)

 if water_increases:
 print(f"Debug — water increase stats: "
 f"min={min(water_increases):.3f}, "
 f"max={max(water_increases):.3f}, "
 f"mean={np.mean(water_increases):.3f}")

 print(f"Labeled FLOOD patches: {len(flood_images)}")
 print(f"Labeled NO-FLOOD patches: {len(no_flood_images)}")

 return flood_images, no_flood_images


def build_real_dataset():
 """Process all flood events and combine into one labeled dataset."""
 initialize_gee()

 all_flood = []
 all_no_flood = []

 for event in FLOOD_EVENTS:
 try:
 flood_imgs, no_flood_imgs = process_event(event)
 all_flood.extend(flood_imgs)
 all_no_flood.extend(no_flood_imgs)
 except Exception as e:
 print(f"Error processing {event['name']}: {e}")
 continue

 print(f"\n{'='*60}")
 print("DATASET SUMMARY")
 print(f"{'='*60}")
 print(f"Total FLOOD patches: {len(all_flood)}")
 print(f"Total NO-FLOOD patches: {len(all_no_flood)}")

 min_count = min(len(all_flood), len(all_no_flood))
 if min_count == 0:
 print("\n WARNING: One class has zero samples!")
 print("Try widening the date ranges or loosening thresholds further.")
 return None, None

 np.random.seed(42)
 flood_idx = np.random.choice(len(all_flood), min_count, replace=False)
 no_flood_idx = np.random.choice(len(all_no_flood), min_count, replace=False)

 X_flood = np.array([all_flood[i] for i in flood_idx])
 X_no_flood = np.array([all_no_flood[i] for i in no_flood_idx])

 X = np.concatenate([X_no_flood, X_flood], axis=0)
 y = np.concatenate([
 np.zeros(len(X_no_flood)),
 np.ones(len(X_flood))
 ])

 print(f"\n Balanced dataset: {len(X)} total patches")
 print(f"({min_count} flood, {min_count} no-flood)")

 return X, y


def save_dataset(X, y):
 """Resize patches to CNN input size, normalize, and save to disk."""
 os.makedirs('data/processed', exist_ok=True)

 print(f"\n Resizing patches from {PATCH_SIZE}x{PATCH_SIZE} "
 f"to {FINAL_PATCH_SIZE}x{FINAL_PATCH_SIZE}...")

 X_resized = np.array([
 np.array(Image.fromarray(img).resize(
 (FINAL_PATCH_SIZE, FINAL_PATCH_SIZE)))
 for img in X
 ])
 X_normalized = X_resized.astype(np.float32) / 255.0

 np.save('data/processed/real_flood_patches_X.npy', X_normalized)
 np.save('data/processed/real_flood_patches_y.npy', y)

 print(f"\n Saved dataset:")
 print(f"data/processed/real_flood_patches_X.npy shape={X_normalized.shape}")
 print(f"data/processed/real_flood_patches_y.npy shape={y.shape}")


if __name__ == "__main__":
 X, y = build_real_dataset()

 if X is not None:
 save_dataset(X, y)
 print("\n Real satellite patch dataset ready!")
 print("Next: retrain the CNN using this dataset.")
 else:
 print("\n Dataset build failed — see warnings above.")