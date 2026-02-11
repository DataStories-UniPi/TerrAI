import geopandas as gpd
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
import argparse
import rasterio
import os

import torch
import st_image_toolkit as stt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Targeted Fertilization Data Preprocessing')
    parser.add_argument('--red_index', help='Index of the Red (R) spectral band', default=1, type=int, required=False)
    parser.add_argument('--green_index', help='Index of the Green (G) spectral band', default=2, type=int, required=False)
    parser.add_argument('--blue_index', help='Index of the Blue (B) spectral band', default=3, type=int, required=False)
    parser.add_argument('--nir_index', help='Index of the Near-InfraRed (NIR) spectral band', default=0, type=int, required=False)
    parser.add_argument('--spectral_bands', help='Additional spectral bands to include (Vegetation Indices)', 
                        default=['NIR', 'R', 'G', 'B'], type=str, required=False, action="append",
                        choices=(VI_BANDS_NAMES:=['NDVI', 'BNDVI', 'GNDVI', 'GBNDVI', 'GRNDVI', 'RBNDVI']))
    parser.add_argument('--weather_bands', help='Additional spectral bands to include (Weather Conditions)', default=[], type=str, required=False, action="append")
    parser.add_argument('--kernel', help='Patches\' kernel size', default=8, type=int, required=False)
    parser.add_argument('--stride', help='Patches\' kernel stride', default=1, type=int, required=False)
    parser.add_argument('--lands', help='Focus on specific land(s)', default=[], type=str, required=True, choices=['Wheat', 'Soybean', 'Barley'], action="append")
    parser.add_argument('--phases', help='Focus on specific fertilization phase', default=[], type=str, required=True, choices=['1', '2', '3'], action="append")
    args = parser.parse_args()


    # %%
    # Data Parsing
    DATA_PATH = os.path.join('.', 'data')
    GEOJSON_TIF_PATH = os.path.join(DATA_PATH, 'Prescriptions_2023 10 27')
    
    NO_DATA_VALUE = -9999

    WEATHER_BANDS_NAMES = [
        'min T', 'mean T', 'T', 'max T',
        'min rel. hum.', 'mean rel. hum.', 'rel. hum.', 'max rel. hum.',
        'min pressure', 'mean pressure', 'max pressure',
        'max gust', 'wind direction', 'wind speed',
        'precipitation', 'global energy'
    ]
    
    SPECTRAL_BANDS_NAMES, WEATHER_BANDS_RS_NAMES = [
        'NIR', 'R', 'G', 'B', *VI_BANDS_NAMES
    ], [
        f'{band}; T{time}' for band, time in np.array(
            np.meshgrid(WEATHER_BANDS_NAMES, ['0', '8', '16'])
        ).T.reshape(-1, 2)
    ]

    # Parse Dataset (Fertilization Data; Main Table)
    fertilization_data = pd.read_pickle(os.path.join(DATA_PATH, 'pkl', './Targeted fertilization v3.labels.weather.pkl'))
    

    # %%
    # Data Preprocessing  
    # # Drop Outlier GERKs, in terms of ammonitrat used
    target_median = fertilization_data.GERK_SHP.str.strip(' ').apply(lambda l: gpd.read_file(os.path.join(GEOJSON_TIF_PATH, l)).iloc[:, 0].median())

    q25, q75 = target_median.describe()[['25%', '75%']]
    iqr = q75 - q25

    fertilization_data.drop(
        target_median.loc[~target_median.between(q25 - 1.5*iqr, q75 + 1.5*iqr, inclusive='both')].index,
        inplace=True
    )

    # %%
    # # Feature engineering; Create additional bands (vegetation indices)
    # # Data aggregation; Integrate weather conditions for timestamp of fetilization phase
    
    # ## Index: GERK, Datetime | Features: NIR, R, G, B, ...; Target: Ammonitrat
    input_rasters, target_rasters = {}, {}

    for ix, rec in tqdm(enumerate(fertilization_data.itertuples()), desc='Calculating Vegetation Indices...'):
        input_raster = rasterio.open(os.path.join(GEOJSON_TIF_PATH, rec.NDVI_TIF.strip(' ')))
        target_raster = rasterio.open(os.path.join(GEOJSON_TIF_PATH, rec.LABEL_TIF.strip(' ')))

        # ### Get baseline bands from input raster
        input_raster_bands = input_raster.read((2,3,4,5))
        
        # ### Create additional bands - vegetation indices
        input_raster_bands = np.concatenate(
            (
                # NIR, R, G, B
                input_raster_bands, 
                # Normalized Difference Vegetation Index (NDVI)
                np.expand_dims(stt.ndvi_band(input_raster_bands[args.nir_index], input_raster_bands[args.red_index]), 0),    
                # Blue Normalized Difference Vegetation Index (BNDVI)
                np.expand_dims(stt.bndvi_band(input_raster_bands[args.nir_index], input_raster_bands[args.blue_index]), 0),  
                # Green Normalized Difference Vegetation Index (GNDVI)
                np.expand_dims(stt.gndvi_band(input_raster_bands[args.nir_index], input_raster_bands[args.green_index]), 0), 
                # Green-Blue Normalized Difference Vegetation Index (GBNDVI)
                np.expand_dims(stt.gbndvi_band(input_raster_bands[args.nir_index], input_raster_bands[args.green_index], input_raster_bands[args.blue_index]), 0), 
                # Green-Red Normalized Difference Vegetation Index (GRNDVI)
                np.expand_dims(stt.grndvi_band(input_raster_bands[args.nir_index], input_raster_bands[args.green_index], input_raster_bands[args.red_index]), 0),  
                # Red-Blue Normalized Difference Vegetation Index (RBNDVI)
                np.expand_dims(stt.rbndvi_band(input_raster_bands[args.nir_index], input_raster_bands[args.red_index], input_raster_bands[args.blue_index]), 0),   
            ), 
            axis=0
        )   

        input_weather_bands = rec.weather_bands

        # ### Mask input data \w ```no-data```
        input_raster_bands[input_raster_bands == NO_DATA_VALUE] = 0

        # ### Get Target from Output
        target_raster_band = target_raster.read((1,))
        # target_raster_band[target_raster_band == NO_DATA_VALUE] = 0
        
        # ### Add to Pandas dict.
        input_rasters[(*rec.Index, ix)] = np.concatenate((input_raster_bands, input_weather_bands))
        target_rasters[(*rec.Index, ix)] = target_raster_band


    # # Create Raster Dataset...
    raster_dataset = pd.DataFrame({
        'input':input_rasters,
        'output':target_rasters
    }).rename_axis(
        ['GERK', 'datetime', None]
    )

    raster_dataset.loc[:, ['Field name', 'Area (ha)', 'Crop', 'month', 'phase']] = fertilization_data[
        ['Field name', 'Area (ha)', 'Crop', 'month', 'phase']
    ].values

    # ## Save raster dataset for future reference
    raster_dataset.to_pickle(os.path.join(DATA_PATH, 'pkl', './Targeted fertilization v3.raster_dataset.weather.v3.pkl'))


    # %%
    # # Create Patches Dataset...
    input_patches, input_metadata, output_patches, output_metadata = {}, {}, {}, {}

    raster_dataset.input = raster_dataset.input.apply(lambda l: torch.Tensor(l).unsqueeze(0))
    raster_dataset.output = raster_dataset.output.apply(lambda l: torch.Tensor(l).unsqueeze(0))

    for land in tqdm(raster_dataset.itertuples(), desc='Creating Patches Dataset...'):
        try:
            land_input_samples, land_input_meta = stt.segment_images(
                land.input, kernel_size=args.kernel, stride=args.stride
            )
            land_output_samples, land_output_meta = stt.segment_images(
                land.output, kernel_size=args.kernel, stride=args.stride
            )

            input_patches[land.Index], output_patches[land.Index] = land_input_samples, land_output_samples
            input_metadata[land.Index], output_metadata[land.Index] = land_input_meta, land_output_meta

        except RuntimeError:
            tqdm.write(f'Segmentation Error | Land ID: {land.Index} | Phase: {land.phase} | Crop: {land.Crop} | Shape: {land.input.shape}')
            input_patches[land.Index], output_patches[land.Index] = None, None
            input_metadata[land.Index], output_metadata[land.Index] = None, None

    raster_dataset.loc[:, 'input_patches'] = pd.Series(input_patches)
    raster_dataset.loc[:, 'input_patches_metadata'] = pd.Series(input_metadata)
    raster_dataset.loc[:, 'output_patches'] = pd.Series(output_patches)
    raster_dataset.loc[:, 'output_patches_metadata'] = pd.Series(output_metadata)

    # ## Save raster patches for future reference
    raster_dataset.to_pickle(os.path.join(DATA_PATH, 'pkl', f'./Targeted fertilization v3.patches_dataset.kernel={args.kernel}_stride={args.stride}.weather.v3.pkl'))


    # %%
    # # Feature Selection
    all_feats = [*SPECTRAL_BANDS_NAMES, *WEATHER_BANDS_RS_NAMES, 'AMMONITRAT']
    # print(all_feats)

    selected_features = [*args.spectral_bands, *args.weather_bands]
    print(f'Selected Features: {selected_features}')

    selected_features_ix = [all_feats.index(i) for i in selected_features]

    # ## Focus on specific land and/or fertilization phase
    patches_dataset_clean = raster_dataset.loc[(raster_dataset.Crop.isin(args.lands)) & (raster_dataset.phase.isin(args.phases))].copy()

    # ## Drop records with no patches (i.e., drop NaN values)
    patches_dataset_clean.dropna(subset=['input_patches'], inplace=True)

    # ## Get the selected features
    patches_dataset_clean.input_patches = patches_dataset_clean.input_patches.apply(lambda l: l[:,:,selected_features_ix,:,:])

    # ## Update patches' metadata
    patches_dataset_clean.input_patches_metadata.apply(lambda l: l.update({'C':len(selected_features_ix)}))

    # ## Save filtered raster patches for future reference
    patches_dataset_clean.to_pickle(os.path.join(DATA_PATH, 'pkl', f'./Targeted fertilization v3.patches_dataset.kernel={args.kernel}_stride={args.stride}_bands={len(selected_features_ix)}.weather.land={"-".join(args.lands)}_ph={"-".join(args.phases)}.v3.pkl'))
