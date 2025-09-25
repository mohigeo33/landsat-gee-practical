# -*- coding: utf-8 -*-
"""Practical 3: Thermal time series analysis from Landsat in GEE.ipynb

**About the script:**
This script is prepared for the forest information technology (FIT) 3rd semester students of Eberswalde University for Sustainable Development (HNEE).

*Developed by: Gulam Mohiuddin*

**1. Getting the Google Earth Engine Python API ready and activated**
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
ee.Initialize(project='ENTER YOUR PROJECT NAME HERE')

"""**2. Introducing area of interest (AOI)**"""

# Phnom Penh, Cambodia
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
year_end = 2024
month_start = 2
month_end = 3

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

# Calling the image collections
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

# Landsat 9
# Surface Reflectance (SR) Collection (Level 2, Collection 2, Tier 1)
imgCol_L9_SR = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')\
    .filterBounds(roi)\
    .filter(ee.Filter.calendarRange(year_start,year_end,'year'))\
    .filter(ee.Filter.calendarRange(month_start,month_end,'month'))\
    .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover))\
    .map(function_mask_clouds_l89)

size_L9_SR = imgCol_L9_SR.size()
print("Total LS 9 collection SR: ",size_L9_SR.getInfo())

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
imgCol_merge_L9_band_coh = imgCol_L9_SR.map(function_bands_l89)

"""***5. Clipping the collections to the area of interest (AOI)***"""

# Function 3: clipping the image collections to study area
def fun_clip(img):
  clip_img = img.clip(roi)
  return clip_img

# Applying clip function
imgCol_merge_L8_band_coh_clip = imgCol_merge_L8_band_coh.map(fun_clip)
imgCol_merge_L9_band_coh_clip = imgCol_merge_L9_band_coh.map(fun_clip)

"""***6. Merging the image collections into one collection***"""

# Merging the collection
imgCol_merge = imgCol_merge_L8_band_coh_clip.merge(imgCol_merge_L9_band_coh_clip)

imgCol_merge

"""***7. Extracting data for statistical analysis***"""

# Function 4: Extracting the tabular information from the processed images
def fctn_get_image_stats(img):
    img_lst = img.select('LST')

    # Calculate basic statistics
    img_lst_mean_value = img_lst.reduceRegion(ee.Reducer.mean(), roi, 30, crs='EPSG:4326', bestEffort=True, maxPixels=1e9).getInfo().get('LST')
    img_lst_max_value = img_lst.reduceRegion(ee.Reducer.max(), roi, 30, crs='EPSG:4326', bestEffort=True, maxPixels=1e9).getInfo().get('LST')
    img_lst_min_value = img_lst.reduceRegion(ee.Reducer.min(), roi, 30, crs='EPSG:4326', bestEffort=True, maxPixels=1e9).getInfo().get('LST')

    # get metadata
    img_date = img.date().getInfo().get('value')
    img_systemindex = img.get('system:index').getInfo()
    img_cloud_cover = img.get('CLOUD_COVER').getInfo()
    img_spacecraft = img.get('SPACECRAFT_ID').getInfo()

    # Create dictionary with all information
    img_all_info = {
        'system:index': img_systemindex,
        'date': img_date,
        'mean_lst': img_lst_mean_value,
        'min_lst': img_lst_min_value,
        'max_lst': img_lst_max_value,
        'cloud_cover': img_cloud_cover,
        'satellite': img_spacecraft
    }

    return img_all_info

# Converting image collection to list
doi = imgCol_merge
doiList = doi.toList(doi.size())
doiList_size = doiList.size().getInfo()
print('Total Images in Data of Interest (doi) dataset: ', doiList_size)

import pandas as pd
# Creating the dataframe
df = pd.DataFrame(columns=['SystemIndex', 'Millis', 'MeanLST', 'MaxLST', 'MinLST', 'Cloud', 'satellite'])

from tqdm import tqdm
# Iteration
for i in tqdm(range(doiList_size)):
    image = ee.Image(doiList.get(i))
    image_info = fctn_get_image_stats(image)

    # Create a temporary DataFrame for the new row
    temp_df = pd.DataFrame([{
        'SystemIndex': image_info['system:index'],
        'Millis': image_info['date'],
        'MeanLST': image_info['mean_lst'],
        'MaxLST': image_info['max_lst'],
        'MinLST': image_info['min_lst'],
        'Cloud': image_info['cloud_cover'],
        'satellite': image_info['satellite']
    }])

    # Append the new row using pd.concat()
    df = pd.concat([df, temp_df], ignore_index=True)

df

"""**8. Adding the date variables**"""

# Function 5: Add date variables to DataFrame.
def add_date_info(df):
  df['Timestamp'] = pd.to_datetime(df['Millis'], unit='ms')
  df['Year'] = pd.DatetimeIndex(df['Timestamp']).year
  df['Month'] = pd.DatetimeIndex(df['Timestamp']).month
  df['Day'] = pd.DatetimeIndex(df['Timestamp']).day
  df['DOY'] = pd.DatetimeIndex(df['Timestamp']).dayofyear
  return df

#adding the date variables
df = add_date_info(df)

df

"""Optional: save the data as csv in your drive so that next time you do not need to run the extraction again"""

# mounting the google dirive
from google.colab import drive
drive.mount('/content/drive')

# setting the working directory
os.chdir('/content/drive/MyDrive/Office/Teaching/ARSI_FIT_2025/25.09.2025')
print("Current working directory: {0}".format(os.getcwd()))

# export the dataframe to a CSV file
df.to_csv('lsttimeseries_ph_2014_2024.csv', index=False) # decide the data name urself

# Function 6: Loading the csv/txt file
def load_csv(filepath):
    data =  []
    col = []
    checkcol = False
    with open(filepath) as f:
        for val in f.readlines():
            val = val.replace("\n","")
            val = val.split(',')
            if checkcol is False:
                col = val
                checkcol = True
            else:
                data.append(val)
    df = pd.DataFrame(data=data, columns=col)
    return df

import pandas as pd
# Loading the data
myData = load_csv('lsttimeseries_ph_2014_2024.csv')
df = myData
print(df.head())

"""**9. Checking for artefacts**"""

df

# Converting the variables into numeric
#Mean
df["MeanLST"] = pd.to_numeric(df["MeanLST"])
meanlst = df["MeanLST"]

#Max
df["MaxLST"] = pd.to_numeric(df["MaxLST"])
maxlst = df["MaxLST"]

#Min
df["MinLST"] = pd.to_numeric(df["MinLST"])
minlst = df["MinLST"]

#Time and cloud
df['Year'] = pd.to_numeric(df['Year'])
df['Month'] = pd.to_numeric(df['Month'])
df['Cloud'] = pd.to_numeric(df['Cloud'])
df['DOY'] = pd.to_numeric(df['DOY'])

"""Different studies that used the Landsat analysis-ready LST data mentioned there are sometimes artefacts in this data. While there are no established systematic method till date to filter out these artefacts, we can use logic based on climatic reality for the study area to find a way to minimise the artefacts.

If you are interested to learn what can be a way of systematically filter out such artefacts, you can read at: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5373152

**The first thing we will do is filter out the values zero or lower LST which is unrealistic value for a tropical city.**
"""

# Checking for values <= 0 in LST columns
print("MinLST has values <= 0:", (df['MinLST'] <= 0).any())
print("MeanLST has values <= 0:", (df['MeanLST'] <= 0).any())
print("MaxLST has values <= 0:", (df['MaxLST'] <= 0).any())

"""**Since the result for MinLST came True it means it has such values that is zero or lower and we will filter them out now.**"""

# Filter the DataFrame to exclude rows where MinLST is <= 0
df_filtered = df[df['MinLST'] > 0].copy()

# Display the filtered DataFrame to verify
display(df_filtered)

"""**We will also filter out the data that has null values in the LST categories.**"""

import numpy as np
# Removing rows with null values in MinLST, MeanLST, MaxLST
df_filtered.replace(['', ' ', '-1'], np.nan, inplace = True)
df_filtered = df_filtered.dropna(subset=['MinLST', 'MeanLST', 'MaxLST'])
df_filtered

print(df_filtered.head())

"""**sensor wise images and their cloud%**"""

import altair as alt

sensor_domain = ['LANDSAT_8', 'LANDSAT_9']
sensor_colors = ['#1f78b4', '#e31a1c']  # Strong blue and red, high contrast

alt.Chart(df_filtered).mark_point(size=100, filled=True).encode(
    x=alt.X('Timestamp:T', title='Year'),
    y=alt.Y('Cloud:Q', title='Cloud cover in %'),
    color=alt.Color(
        'satellite:N',
        scale=alt.Scale(domain=sensor_domain, range=sensor_colors),
        legend=alt.Legend(title="Sensor Type")),
    tooltip=[
        alt.Tooltip('Timestamp:T', title='Time'),
        alt.Tooltip('Cloud:Q', title='Cloud cover (in %)'),
        alt.Tooltip('satellite:N', title='Sensor Type')
    ]
).properties(width=600, height=300)

"""**Distribution of LST values**"""

import matplotlib.pyplot as plt

# Select only the relevant columns
cols = ['MinLST', 'MeanLST', 'MaxLST']
df_plot = df_filtered[cols]

# Create the plot
plt.figure(figsize=(8, 6))
df_plot.boxplot(column=cols)

plt.ylabel("LST in [°C]", fontsize=12)
plt.title("", fontsize=14)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

"""**As you can see there is a outlier in MaxLST which is quite higher than a normal high LST values, we can also consider this as artefacts and now we will filter it out to make the data more realistic.**"""

df_filtered = df_filtered[df_filtered['MaxLST'] < 80]

"""**Histogram of the minimum, mean and maximum LST**"""

import matplotlib.pyplot as plt
import seaborn as sns

# Pick the columns
cols = ['MinLST', 'MeanLST', 'MaxLST']
colors = ['blue', 'green', 'orange']

# Create subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, col, color in zip(axes, cols, colors):
    sns.histplot(
        df_filtered[col],
        kde=True,          # add normal distribution-like curve
        stat="density",
        bins=30,
        color=color,
        ax=ax,
        alpha=0.6
    )
    ax.set_title(col)
    ax.set_xlabel("LST [°C]")
    ax.grid(True, linestyle='--', alpha=0.6)

axes[0].set_ylabel("Density")

plt.suptitle("Distribution of LST values", fontsize=16)
plt.tight_layout()
plt.show()

"""**Correlation between variables**"""

# Correlation matrix
cols_for_corr = ['MinLST', 'MeanLST', 'MaxLST', 'Cloud', 'DOY']
correlation_matrix = df_filtered[cols_for_corr].corr()

#Display the correlation matrix
# print(correlation_matrix)

# Optionally, visualize the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('')
plt.show()

"""**Cloud cover vs LST categories**"""

import matplotlib.pyplot as plt
import seaborn as sns

cols = ['MinLST', 'MeanLST', 'MaxLST']
colors = ['blue', 'green', 'orange']

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, col, color in zip(axes, cols, colors):
    sns.scatterplot(
        x='Cloud',
        y=col,
        data=df_filtered,
        color=color,
        ax=ax,
        alpha=0.6,
        edgecolor=None
    )
    ax.set_title(f"{col}")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel("Cloud (%) in the Landsat Scene" if col == 'MeanLST' else "")
    ax.set_ylabel("LST [°C]" if col == 'MinLST' else "")

plt.suptitle("", fontsize=16)
plt.tight_layout()
plt.show()

"""**Yearly LST trends**"""

import matplotlib.pyplot as plt

# Group by Year and take the mean (or median, depending on what you want)
df_yearly = df_filtered.groupby("Year")[["MinLST", "MeanLST", "MaxLST"]].mean().reset_index()

import numpy as np
from scipy.interpolate import make_interp_spline

plt.figure(figsize=(10, 6))

for col, color, label in zip(["MinLST", "MeanLST", "MaxLST"],
                             ["blue", "green", "orange"],
                             ["Min LST", "Mean LST", "Max LST"]):
    x = df_yearly["Year"].values
    y = df_yearly[col].values

    # Make dense x-values for smooth curve
    xnew = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=3)  # cubic spline
    y_smooth = spl(xnew)

    plt.plot(xnew, y_smooth, color=color, label=label)

plt.xlabel("Year")
plt.ylabel("LST [°C]")
plt.title("Yearly trends of LST metrics (smoothed)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

for col, color, label in zip(["MinLST", "MeanLST", "MaxLST"],
                             ["blue", "green", "orange"],
                             ["Min LST", "Mean LST", "Max LST"]):
    x = df_yearly["Year"].values.astype(float)
    y = df_yearly[col].values.astype(float)

    # --- Smooth line with spline ---
    xnew = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=3)  # cubic spline
    y_smooth = spl(xnew)
    plt.plot(xnew, y_smooth, color=color, label=label)

    # --- Trendline (linear regression fit) ---
    coeffs = np.polyfit(x, y, 1)  # degree 1 = linear
    y_trend = np.polyval(coeffs, x)
    plt.plot(x, y_trend, linestyle="--", color=color, alpha=0.7)

plt.xlabel("Year")
plt.ylabel("LST [°C]")
plt.title("Yearly trends of LST metrics (smoothed")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

"""**Yearly distribution of LST**"""

import matplotlib.pyplot as plt
import seaborn as sns

# Melt dataframe into long format for seaborn
df_long = df_filtered.melt(
    id_vars="Year",
    value_vars=["MinLST", "MeanLST", "MaxLST"],
    var_name="LST Type",
    value_name="LST Value"
)

plt.figure(figsize=(12, 6))
sns.boxplot(
    x="Year",
    y="LST Value",
    hue="LST Type",
    data=df_long,
    palette={"MinLST": "blue", "MeanLST": "green", "MaxLST": "orange"}
)

plt.title("Yearly Distribution of LST Metrics", fontsize=14)
plt.xlabel("Year")
plt.ylabel("LST [°C]")
plt.legend(title="LST Type")
plt.grid(True, axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Select the two specific images
image_2024_feb = imgCol_merge.filterMetadata('system:index', 'equals', '2_LC09_126052_20240217').first().select('LST')
image_2016_feb = imgCol_merge.filterMetadata('system:index', 'equals', '1_LC08_126052_20160219').first().select('LST')

image_2024_feb

image_2016_feb

# 3) Min/Max for each image
stats_2024 = image_2024_feb.reduceRegion(
    reducer=ee.Reducer.minMax(),
    geometry=roi,
    scale=30,
    bestEffort=True,
    maxPixels=1e13
)
stats_2016 = image_2016_feb.reduceRegion(
    reducer=ee.Reducer.minMax(),
    geometry=roi,
    scale=30,
    bestEffort=True,
    maxPixels=1e13
)

min_2024 = stats_2024.getNumber('LST_min').getInfo()
max_2024 = stats_2024.getNumber('LST_max').getInfo()
min_2016  = stats_2016.getNumber('LST_min').getInfo()
max_2016  = stats_2016.getNumber('LST_max').getInfo()

print(f"February 2024 LST in °C:   {min_2024:.2f} – {max_2024:.2f}")
print(f"February 2016 LST in °C: {min_2016:.2f} – {max_2016:.2f}")

# 4) Use a COMMON stretch across both images
vmin = round(min(min_2016, min_2024), 2)
vmax = round(max(max_2016, max_2024), 2)
print("Common LST range:", vmin, "to", vmax)

palette = ['0000ff','00ffff','00ff00','ffff00','ff8000','ff0000']
vis = {'min': vmin, 'max': vmax, 'palette': palette}

# 5) Map: show both layers
Map = geemap.Map()
Map.centerObject(image_2024_feb, 10)
Map.addLayer(image_2016_feb, vis, "LST — 2016 February")
Map.addLayer(image_2024_feb,  vis, "LST — 2024 February")

# Legend (2-decimal labels)
Map.add_colorbar_branca(colors=palette, vmin=vmin, vmax=vmax, caption="LST")

Map

# change map
change_lst = image_2024_feb.subtract(image_2016_feb)
# Keep only positive values in change_lst
positive_change_lst = change_lst.updateMask(change_lst.gt(1))

stats_change = positive_change_lst.reduceRegion(
    reducer=ee.Reducer.minMax(),
    geometry=roi,
    scale=30,
    bestEffort=True,
    maxPixels=1e13
)

min_change = stats_change.getNumber('LST_min').getInfo()
max_change = stats_change.getNumber('LST_max').getInfo()

print(f"range of changes in LST: {min_change:.2f} – {max_change:.2f}")

change_lst

# Create a color palette for the change in LST
# Green is lowest, red is highest
palette = ['00ff00', 'ffff00', 'ff0000']
vis = {'min': 1.00, 'max': 29.00, 'palette': palette}

# 5) Map: show both layers
Map = geemap.Map()
Map.centerObject(change_lst, 10)
Map.addLayer(positive_change_lst, vis, "changes in LST 2016-2024")


# Legend (2-decimal labels)
Map.add_colorbar_branca(colors=palette, vmin=1, vmax=29, caption="Changes in LST")

Map

# Define the export parameters
export_params_2024 = {
    'image': image_2024_feb.select('LST'),
    'description': 'LST_2024_feb',
    'folder': 'GEE_Export_FIT', # Specify a folder in your Google Drive
    'fileNamePrefix': '2024_feb',
    'scale': 30,
    'region': roi,
    'crs': 'EPSG:4326',
    'maxPixels': 1e13
}

export_params_2016 = {
    'image': image_2016_feb.select('LST'),
    'description': 'LST_2016_feb',
    'folder': 'GEE_Export_FIT', # Specify a folder in your Google Drive
    'fileNamePrefix': '2016_feb',
    'scale': 30,
    'region': roi,
    'crs': 'EPSG:4326',
    'maxPixels': 1e13
}

# Start the export tasks
task_2024 = ee.batch.Export.image.toDrive(**export_params_2024)
task_2016 = ee.batch.Export.image.toDrive(**export_params_2016)

task_2024.start()
task_2016.start()

print("Export tasks started for 2024_feb and 2016_feb.")
print("Check the Tasks tab in the GEE Code Editor or the Colab Tasks sidebar for progress.")

"""Clue for Google Earth Overlay

North = Ymax

South = Ymin

East = Xmax

West = Xmin
"""
