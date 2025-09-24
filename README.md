# Google Earth Engine (Python): Remote Sensing Practicals

**Author**: [Gulam Mohiuddin](https://www.linkedin.com/in/mohigeo33/) (2025)  

Repository of Google Earth Engine (Python API) workflows developed for MSc *Forest Information Technology (FIT)*, Eberswalde University for Sustainable Development (HNEE).

---

## 📑 Citation

If you use this code in your work, please cite the Zenodo archive:

> Mohiuddin, G. (2025). *Google Earth Engine Remote Sensing Practicals (Landsat & MODIS)* (v0.2.0) [Software]. [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17121725.svg)](https://doi.org/10.5281/zenodo.17121725)

---

## 📂 Practicals Included

### 🟢 Practical 1 — Basic Operations & Visualization (Landsat)
- Setup **Google Earth Engine API** in Python.  
- Define **AOI** using **OpenStreetMap** (Phnom Penh).  
- Load **Landsat 8 SR (C2 L2)** collection, apply **cloud masking**.  
- Apply **scale factors** and harmonize band names.  
- Identify **least-cloudy image**.  
- Visualize composites:
  - Natural Color (RGB)
  - Color Infrared (CIR)
  - Agriculture (SWIR1, NIR, Blue)  
- Compute & visualize indices: **NDVI, NDWI, NDBI**.  

### 🔵 Practical 2 — Estimating LST from MODIS
- Use **MOD11A1 (Terra, V6.1)** daily product.  
- Apply **QA filtering** to retain reliable pixels.  
- Convert thermal bands to °C.  
- Identify **best same-day acquisition** with valid **Day & Night LST**.  
- Visualize **daytime & nighttime LST** with a common color stretch.  

### 🔴 Practical 3 — Thermal Time Series Analysis (Landsat 8/9)
- Retrieve **Landsat 8 & 9 SR collections** (2014–2024).  
- Apply **cloud masking** and **scale/offset correction**.  
- Merge L8 & L9 collections into a unified dataset.  
- Clip imagery to AOI.  
- Extract **LST statistics (mean, min, max)** per image.  
- Build a **time series dataset** for long-term thermal monitoring.  

---

## ⚙️ Quick Start

### 1) Install dependencies

**pip:**
```bash
pip install -r requirements.txt
```

**conda:**
```bash
conda env create -f environment.yml
conda activate gee-practicals
```

### 2) Authenticate Earth Engine

```python
import ee
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()
```

### 3) Run the scripts / notebooks

All scripts are Jupyter/Colab-ready. For local Jupyter:

```bash
python -m ipykernel install --user --name gee-practicals
jupyter lab
```

Open the relevant script under `scripts/` and run cell by cell.

---

## 📂 Repository Layout

```
gee-practicals/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ requirements.txt
├─ environment.yml
└─ scripts/
   ├─ practical_1_basic_operation_and_visualisation_in_gee.py
   ├─ practical_2_estimating_lst_from_modis_in_gee.py
   └─ practical_3_thermal_time_series_analysis_from_landsat_in_gee.py
```

---

## ⚠️ Notes

- Lines starting with `!pip install ...` are **notebook magics**. Keep them if running in Colab/Jupyter, or replace with standard `subprocess` calls in pure Python scripts.  
- AOI is defined via **OSM geocoding** (`osmnx`). Change `"Phnom Penh, Cambodia"` to your own study area if needed.  
- MODIS & Landsat scale factors are already applied; indices and LST are ready to interpret.  

---

## 📜 License & Citation

- License: **MIT** (see `LICENSE`).  
- Citation info: see `CITATION.cff`.  
