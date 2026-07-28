"""
Phase 1: Dataset Generation
Extracts clear (target) and cloudy optical imagery via Google Earth Engine API.
Exports directly to Google Drive as matched pairs.
"""
import ee

def export_tile_to_drive(roi, tile_name):
    # Target (Ground Truth)
    target = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(roi) \
        .filterDate("2026-01-01", "2026-03-30") \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 1)) \
        .median().select(["B4", "B3", "B2", "B8"])

    # Cloudy Input
    cloudy = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(roi) \
        .filterDate("2026-06-01", "2026-08-30") \
        .filter(ee.Filter.gt("CLOUDY_PIXEL_PERCENTAGE", 40)) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)) \
        .first().select(["B4", "B3", "B2", "B8"])

    # Asynchronous background export jobs
    ee.batch.Export.image.toDrive(
        image=target,
        description=f"{tile_name}_target",
        folder="ISRO_SEN12MS_DATASET",
        scale=10,
        region=roi,
        maxPixels=1e10
    ).start()

    ee.batch.Export.image.toDrive(
        image=cloudy,
        description=f"{tile_name}_cloudy",
        folder="ISRO_SEN12MS_DATASET",
        scale=10,
        region=roi,
        maxPixels=1e10
    ).start()

def main():
    ee.Initialize()
    
    regions = [
        ("Pune", ee.Geometry.Rectangle([73.80, 18.45, 73.95, 18.60])),
        ("Mumbai", ee.Geometry.Rectangle([72.80, 19.00, 72.95, 19.15])),
        ("Delhi", ee.Geometry.Rectangle([77.10, 28.50, 77.25, 28.65])),
        ("Bangalore", ee.Geometry.Rectangle([77.50, 12.90, 77.65, 13.05]))
    ]

    for name, roi in regions:
        export_tile_to_drive(roi, name)

    print("Batch jobs submitted to Google Earth Engine. Monitoring required via Earth Engine Code Editor.")

if __name__ == "__main__":
    main()
