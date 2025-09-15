# -*- coding: utf-8 -*-
"""# GEE Landsat Visualization — Least Cloudy, AOI, Composites & Indices
---------------------------------------------------------------------------------------------
Author: Gulam Mohiuddin (2025)

End-to-end **Google Earth Engine (Python)** workflow for Landsat 8 **Collection 2, Level-2**:
- Select **least-cloudy** image from a preprocessed collection (`imgCol_merge_L8_band_coh`)
- Use **OSM** AOI (Phnom Penh) or your own geometry
- Visualize **true color / CIR / SWIR / Agriculture** band combos
- Compute & visualize **NDVI, NDWI, NDBI** (teaching-friendly palettes)
> Built for MSc *Forest Information Technology* practicals (Eberswalde Univ.).  
> Works great in **Colab** or local Jupyter.

---
"""

# installing libraries
!pip install geemap --upgrade
!pip install geopandas
!pip install osmnx

# loading necessary libraries
import os
import sys
import ee
import geemap
import osmnx as ox
import geopandas as gpd

# authentication
ee.Authenticate()

# initialize the GEE.
ee.Initialize(project='put your project name here')

"""**2. Introducing area of interest (AOI)**"""

# Phnom Penh, Cambodia
print("Fetching Phnom Penh...")
phnom_gdf = ox.geocode_to_gdf('Phnom Penh, Cambodia')
phnom_geojson = phnom_gdf.iloc[0].geometry.__geo_interface__
phnom_aoi = ee.Geometry(phnom_geojson)
roi = phnom_aoi

# visualising the AOI
Map = geemap.Map()
Map.addLayer(roi, {}, 'AOI')
Map.centerObject(roi, zoom= 10)
Map

"""**3. Calling an image collection**"""

# Temporal Parameters
year_start = 2014
year_end = 2015
month_start = 1
month_end = 12

# Cloud cover limit
max_cloud_cover = 60  # in %

# Function 1: Cloud masking using CFMask
def function_mask_clouds_l89(img):
    cloud_mask = img.select('QA_PIXEL').bitwiseAnd(1 << 3).eq(0)
    return img.updateMask(cloud_mask)

"""***Understanding Function 1: cloud masking***

***QA_PIXEL:***

Every Landsat 8/9 image comes with a special quality band called QA_PIXEL. This band doesn’t show colors or reflectance. Instead, each bit (a tiny 0/1 flag) inside the band tells us about the pixel: such as, Is it cloud? Is it cloud shadow? Is it cirrus cloud? etc. Think of it as a box of switches — each switch says “yes/no” for a certain condition.

***1 << 3:***

1 << 3 means check bit number 3 in that quality band.
Bit 3 = “Is this pixel a cloud (high confidence)?”

If bit 3 is 1, it’s cloud.

If bit 3 is 0, it’s not cloud.

***.bitwiseAnd(1 << 3).eq(0):***

.bitwiseAnd(1 << 3) looks only at bit 3.

.eq(0) keeps the pixels where that bit is 0 (not cloud).

So now we have a “mask” that says:

1 = keep (clear pixel)

0 = throw away (cloudy pixel)
"""

# Landsat 8
# Surface Reflectance (SR) Collection (Level 2, Collection 2, Tier 1)
imgCol_L8_SR = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')\
    .filterBounds(roi)\
    .filter(ee.Filter.calendarRange(year_start,year_end,'year'))\
    .filter(ee.Filter.calendarRange(month_start,month_end,'month'))\
    .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover))\
    .map(function_mask_clouds_l89)

size_L8_SR = imgCol_L8_SR.size()
print("Total LS 8 collection SR: ",size_L8_SR.getInfo())

"""**4. Renaming the bands and applying scale factors/readiometric correction**

**Renaming the bands**

Band names can vary accross the sensors of the same sattelite (i.e. NIR is band 4 in Landsat 7 ETM+ but band 5 in Landsat 8 OLI/TIRS sensors). If we intend to use multiple sensors in the same task, the band names needs to be harmonised.


***Scale factor and offset***

***Purpose:***
To perform the conversion from the sensor's raw digital number (DN) to a physical unit like radiance or reflectance.

***How they work:***
The scale factor (also called gain) multiplies the raw DN value, scaling it to the appropriate units.

The offset (also called bias or dark current) is added to the scaled value to account for any background signal or baseline error from the sensor.

***Formula:***
The user then applies the operation: Radiance or Reflectance = (DN * Scale Factor) + Offset.
"""

# Function 2: Renaming bands for coherence and and applying scale factors and offset
def function_bands_l89(img):
       bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
       thermal_band = ['ST_B10']
       new_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2']
       new_thermal_bands = ['LST']
       vnirswir = img.select(bands).multiply(0.0000275).add(-0.2).rename(new_bands)
       tir = img.select(thermal_band).multiply(0.00341802).add(149).subtract(273.15).rename(new_thermal_bands) # converting it from K to celcius by subtracting 273.15
       return vnirswir.addBands(tir).copyProperties(img, img.propertyNames())

# Applying the band coherence function and scale factors
imgCol_merge_L8_band_coh = imgCol_L8_SR.map(function_bands_l89)

imgCol_merge_L8_band_coh

"""**5. Identifying the least cloudy image from the collection**"""

# Sort the collection by CLOUD_COVER (ascending) and pick the first one
least_cloudy = imgCol_merge_L8_band_coh.sort('CLOUD_COVER').first()

# Print image info
print(f"Least cloudy image ID: {least_cloudy.get('system:index').getInfo()}")
print(f"Cloud cover: {least_cloudy.get('CLOUD_COVER').getInfo()}%")

"""**6. visualisation of the selected image**"""

# Create map
Map = geemap.Map()
Map.centerObject(least_cloudy, 8)

"""***Natural Color (red, green, blue):***

The natural color composite uses a band combination of red, green, and blue. It replicates close to what our human eyes can see (GIS Geography, 2025).
"""

# Natural color (R, G, B)
vis_nat = {'bands': ['R','G','B'], 'min': 0.0, 'max': 0.3}
Map.addLayer(least_cloudy, vis_nat, 'Natural Color (RGB)')
Map

"""***Color Infrared (NIR, red, green):***
This band combination is also called the near-infrared (NIR) composite. It uses near-infrared, red, and green. Because chlorophyll reflects near-infrared light, this band composition is useful for analyzing vegetation. In particular, areas in red have better vegetation health (GIS Geography, 2025).
"""

# Color infrared (NIR, R, G) — vegetation in red
vis_cir = {'bands': ['NIR','R','G'], 'min': 0.0, 'max': 0.3}
Map.addLayer(least_cloudy, vis_cir, 'False Color (CIR)')
Map

"""***Agriculture (SWIR-1, NIR, Blue):***

This band combination uses SWIR-1 (6), near-infrared (5), and blue (2). It’s commonly used for crop monitoring because of the use of short-wave and near-infrared. Healthy vegetation appears dark green (GIS Geography, 2025).
"""

# Agriculture composite:
vis_agriculture = {'bands': ['SWIR1', 'NIR', 'B'], 'min': 0.0, 'max': 0.3}

Map.addLayer(least_cloudy, vis_agriculture, 'Landsat Agriculture Composite')
Map

"""***You can try different band combinations based on the Landsat 8 following this website:*** https://gisgeography.com/landsat-8-bands-combinations/

**7. Clipping the image to the area of interest (AOI)**
"""

# Clipping to the AOI
least_cloudy_clip = least_cloudy.clip(roi)

# Visualising the clipped image in Natural color (R, G, B)
Map.addLayer(least_cloudy_clip, vis_nat, 'Phnom Penh Natural Color (RGB)')
Map.centerObject(least_cloudy_clip, zoom= 10)
Map

"""**8. Calculating different spectral indices**

***8.1 Normalized Difference Vegetation Index (NDVI):***

NDVI always ranges from -1 to +1. But there isn’t a distinct boundary for each type of land cover.

For example, when you have negative values, it’s highly likely that it’s water. On the other hand, if you have an NDVI value close to +1, there’s a high possibility that it’s dense green leaves.

But when NDVI is close to zero, there are likely no green leaves and it could even be an urbanized area (GIS Geography, 2025).

Read more about NDVI at: https://gisgeography.com/ndvi-normalized-difference-vegetation-index/
"""

# looking at the band names for easy integration of the formulas
bands = least_cloudy.bandNames().getInfo()
print(bands)

# Calculating NDVI
nir = least_cloudy_clip.select('NIR') # introducing NIR band
red = least_cloudy_clip.select('R') # introducing Red band

ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI') # calculation of ndvi

# Customised palette for NDVI
pal_ndvi = {'min': -1, 'max': 1, 'palette': ['0000ff', '00008b', 'ffff00', '008000']}

# Visualising the NDVI result
Map.addLayer(ndvi, pal_ndvi, 'NDVI')
Map

"""**8.2 Normalized difference water index (NDWI):**

Normalized difference water index (NDWI), is a satellite-derived index used to enhance the presence of open water features or monitor vegetation water content in images. NDWI range from -1 to 1, where high values correspond to high water content.  
Read more at: https://www.sciencedirect.com/science/article/pii/S0034425796000673
"""

# NDWI
green = least_cloudy_clip.select('G') # introducing Green band
ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI') # calculating NDWI

# Customised palette for NDWI
pal_ndwi = {'min': -1,'max': 1,'palette': ['008000', 'ffff00', '0000ff', '00008b']}

# Visualising the NDWI result
Map.addLayer(ndwi, pal_ndwi, 'NDWI')
Map

"""***The Normalized Difference Built-Up Index (NDBI):***

The Normalized Difference Built-Up Index (NDBI) is a remote sensing index calculated from satellite imagery to differentiate and map built-up (urban) areas from non-built-up areas by analyzing spectral reflectance in the Shortwave Infrared (SWIR) and Near-Infrared (NIR) bands. The formula, which subtracts NIR from SWIR and divides by their sum, results in values between -1 and 1, with positive values indicating built-up areas like buildings and roads.

Read more at: https://doi.org/10.1080/01431160304987
"""

swir1 = least_cloudy_clip.select('SWIR1') # introducing SWIR-1 band
ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename('NDBI') # calculating NDBI

# Customised palette for NDBI
pal_ndbi = {'min': -1,'max': 1,'palette': ['0000ff', '90ee90', 'a52a2a', '654321']}

# Visualising the NDBI result
Map.addLayer(ndbi, pal_ndbi, 'NDBI')
Map