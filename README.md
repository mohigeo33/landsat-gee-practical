# Landsat + GEE: Basic Operations & Visualization (Phnom Penh)

**Author of the code**: ***[Gulam Mohiuddin](https://www.linkedin.com/in/mohigeo33/)*** (2025)  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17121725.svg)](https://doi.org/10.5281/zenodo.17121725)

This repository contains a clean, ready-to-run **Google Earth Engine (Python)** workflow focused on **Landsat 8 Collection 2, Level-2** data:

- Select the **least-cloudy** image from a prepared collection (`imgCol_merge_L8_band_coh`).
- Define **AOI** using **OpenStreetMap** (Phnom Penh).
- Visualize **True Color / CIR / SWIR / Agriculture** composites.
- Compute & visualize **NDVI, NDWI, NDBI** with intuitive palettes.
- (Optional) Show **LST** from the same collection (`LST` band).

> Built for MSc *Forest Information Technology* / Advanced Remote Sensing Innovation (Eberswalde University for Sustainable Development).

---

## Quick Start

### 1) Install dependencies (choose one)

**pip:**

```bash
pip install -r requirements.txt
```

**conda:**

```bash
conda env create -f environment.yml
conda activate landsat-gee
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

### 3) Run the script / notebook

Run the script in a Jupyter/Colab environment (recommended) because it contains notebook-style `!pip` calls and map display:

```bash
python -m ipykernel install --user --name landsat-gee
jupyter lab
```

Open: `scripts/practical_1_basic_operation_and_visualisation_in_gee.py` and run cells if using a Jupyter-like environment (e.g., **Colab**).

> If you want a pure `.py` version (without notebook `!` commands), consider converting the installation lines to standard Python `subprocess`-based `pip` calls.

---

## Repository Layout

```
landsat-gee-practical/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ requirements.txt
├─ environment.yml
├─ .gitignore
└─ scripts/
   └─ practical_1_basic_operation_and_visualisation_in_gee.py
```

---

## Notes

- Your script expects a **preprocessed** collection named `imgCol_merge_L8_band_coh` where **scale/offset is already applied** and **bands renamed** to: `B, G, R, NIR, SWIR1, SWIR2` and thermal `LST`.
- In standard Python scripts, lines starting with `!` (e.g., `!pip install ...`) are **IPython/Notebook magics**. Keep the workflow in a notebook or change those lines to standard pip calls if running as a pure script.

---

## License & Citation

- License: **MIT** (see `LICENSE`).
- Citation info: see `CITATION.cff`.
