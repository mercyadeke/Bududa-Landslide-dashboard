# Bududa Landslide Risk & Early Warning Dashboard

Streamlit dashboard: landslide susceptibility (where) × rainfall alert rule (when) × population exposure (who)
for Bududa District, Mt. Elgon, Eastern Uganda.

## 1. Put your Earth Engine exports in `data/`

| File (name can have extra text, e.g. `-0000000000`) | From |
|---|---|
| `Bududa_factor_classes*.tif` | Steps 3+4 script export |
| `Bududa_population_30m*.tif` | Steps 3+4 script export |
| `Bududa_subcounty_exposure_geojson*.geojson` | Steps 3+4 script export |
| `Bududa_subcounty_buildings_csv*.csv` | Buildings-only script (optional) |
| `bududa_rain_rolling_alerts_2010_2024.csv` | already included |
| `bududa_landslide_events_2010_2024.csv` | already included |

`Bududa_susceptibility.tif` is not needed by the app – it re-computes the map from the factor classes
so the weight sliders work.

## 2. Run on your computer

```
pip install -r requirements.txt
streamlit run app.py
```

(On Windows, installing `geopandas` and `rasterio` is easiest with Anaconda:
`conda install -c conda-forge geopandas rasterio` then `pip install -r requirements.txt`.)

## 3. Deploy (public URL for the report)

1. Create a new **public** GitHub repository and upload everything in this folder, including `data/`
   and the hidden `.streamlit/` folder.
2. Go to https://share.streamlit.io → **Create app** → pick the repo, branch `main`, file `app.py` → **Deploy**.
3. Copy the app URL into the report and the dashboard footer.

## What the controls do

- **Rainfall scenario** – 30-day and 3-day rain → alert level (Normal / Watch / Warning)
- **Zones flagged for action** – Auto: Watch → Very high zones, Warning → High + Very high
- **Model weights** – re-weights the six factors and re-draws the susceptibility map live
- **Buffer radius** – click the map; counts people within the radius
- **Focus sub-county** – highlights it and shows its numbers
- **Years** – filters past landslides on the map and timeline
