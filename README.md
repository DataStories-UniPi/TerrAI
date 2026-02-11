# TerrAI
Source code for the paper "Towards Data-driven Nitrogen Estimation in Wheat Fields using Multispectral Images"

# Table of Contents
- Overview
- Prerequisites
- Installation
    - Using Docker Compose
    - Manual Python Environment
- Data Preparation
    - Parsing raw data
    - Weather aggregation
    - Patch creation & preprocessing
- Model Training, Inference & Visualization
- Contributors
- Acknowledgement

# Overview 
TerrAI is an efficient CNN-based solution for Targeted Spraying and Fertilization (TSF) over wheat fields. It ingests raw field data (Excel sheets, shapefiles, GeoTIFFs), enriches it with weather information, creates raster patches, trains a UNet model, and produces visualizations of forecasted prescriptions.

# Prerequisites
- Docker (>=28) – for creating the experimental testbed.
- Docker Compose (>=v2.34.0) – for automating container creation.
- GPU (optional) – if you want to accelerate training with CUDA, modify the Dockerfile accordingly.
- URL and API token for the weather service (required for `weather-data.py`).

# Installation
1. Clone the repository.
1. Build and start the service.
```bash
docker-compose up --build -d
```

The `docker-compose.yaml` defines the service (TerrAI testbed) that builds the image from the provided Dockerfile and mounts the src directory as a volume, giving you live access to the code inside the container. 

# Data Preparation
## Parse field data
```bash
python src/data-parsing.py
```

## Weather forecasts
```bash
python src/weather-data.py --url <URL> --token <TOKEN>
```

## Patch creation & pre-processing
```bash
python src/data-preprocessing.py \
    --spectral_bands BNDVI \
    --weather_bands "min T; T0" \
    --weather_bands "min T; T8" \
    --weather_bands "min T; T16" \
    --weather_bands "mean rel. hum.; T0" \
    --weather_bands "mean rel. hum.; T8" \
    --weather_bands "mean rel. hum.; T16" \
    --weather_bands "mean pressure; T0" \
    --weather_bands "mean pressure; T8" \
    --weather_bands "mean pressure; T16" \
    --weather_bands "max gust; T0" \
    --weather_bands "wind direction; T0" \
    --weather_bands "wind speed; T16" \
    --weather_bands "global energy; T0" \
    --kernel 8 \
    --stride 1 \
    --lands Wheat \
    --phases 2
```

# Model Training, Inference & Visualization
The high‑level training script is `src/training-terrai-v10.py`. In our experimental study, we used the following command for training the baseline TerrAI model
```bash
python training-unet-v10-raster-patch.py \
    --spectral_bands BNDVI \
    --weather_bands "min T; T0" \
    --weather_bands "min T; T8" \
    --weather_bands "min T; T16" \
    --weather_bands "mean rel. hum.; T0" \
    --weather_bands "mean rel. hum.; T8" \
    --weather_bands "mean rel. hum.; T16" \
    --weather_bands "mean pressure; T0" \
    --weather_bands "mean pressure; T8" \
    --weather_bands "mean pressure; T16" \
    --weather_bands "max gust; T0" \
    --weather_bands "wind direction; T0" \
    --weather_bands "wind speed; T16" \
    --weather_bands "global energy; T0" \
    --kernel 8 \
    --stride 1 \
    --land Wheat \
    --phase 2 \
    --use_augmentation \
    --gpuid 0 \
    --hidden_channels 24,36,48 \
    --dropout 0.25
```

Adjust the `--hidden_channels` parameter to `4/8/16` and `72/84/96` for training the small and large TerrAI variants, respectively. After training, use the following methods from `src/helper.py`
  - `model_inference` - folds model predictions back to full‑size rasters via `st_image_toolkit.unary_union`.
  - `visualize_model_results` - creates side‑by‑side plots of NDVI, ground‑truth prescription, and model prediction with colorbars. 

# Contributors
- Andreas Tritsarolis; Department of Informatics, University of Piraeus
- Tomaž Bokan; Innovation Technology Cluster
- Matej Brumen; Department of Electrical Enginnering & Computer Science, University of Maribor
- Domen Mongus; Department of Electrical Enginnering & Computer Science, University of Maribor
- Yannis Theodoridis; Department of Informatics, University of Piraeus

# Acknowledgement
This work was supported in part by the Horizon Europe Research and Innovation Programme of the European Union under grant agreement No. 101070416 (Green.Dat.AI; https://greendatai.eu). The authors would also like to acknowledge INESC-TEC for providing access to their Energy benchmarking tool.
