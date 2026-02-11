# %%
import geopandas as gpd
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
import os

import argparse
import rasterio

import st_image_toolkit as stt
import dataset as ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Aggregating Weather Data')
    parser.add_argument('--url', help='URL for Weather DB', type=str, required=True)    
    parser.add_argument('--token', help='API token for Weather DB', type=str, required=True)    
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

    # # Fertilization Data (Main Table)
    fertilization_data = pd.read_pickle('./Targeted fertilization v3.labels.pkl')
    fertilization_data.loc[:, 'month'] = fertilization_data.index.get_level_values(1).month


    # %%
    # Integrate Weather Data
    # ## Get list of available weather stations
    weather_stations = ds.query_weather_db(
        '''"
        SELECT
            A.*,
            D.Longitude,
            D.Latitude,
            D.Altitude
        FROM 
            objects as A
        JOIN (
            SELECT * 
            FROM datastreams
            WHERE DatastreamId IN (
                SELECT MAX(DatastreamId)
                FROM datastreams
                GROUP BY ObjectId
                WHERE ObjectId IN (120, 125)
            )
        ) as B ON A.ObjectId == B.ObjectId
        JOIN (
            SELECT 
                DatastreamId, 
                LocationId
            FROM Observations
            WHERE ObservationId IN (
                SELECT MAX(ObservationId)
                FROM Observations
                GROUP BY DatastreamId
            )
        ) C ON C.DatastreamId = B.DatastreamId
        JOIN Locations D on D.LocationId = C.LocationId
        "''', 
        args.url, 
        args.token
    )

    weather_stations = gpd.GeoDataFrame(weather_stations, geometry=gpd.points_from_xy(weather_stations.Longitude, weather_stations.Latitude), crs=4326)


    # ## Create Rasters' Weather Band
    fertilization_data_weather = {}

    for ix, row in tqdm(enumerate(fertilization_data.itertuples()), total=len(fertilization_data)):
        bands = stt.create_weather_bands(
            GEOJSON_TIF_PATH,
            row, 
            weather_stations.copy(),
            bands=WEATHER_BANDS_NAMES, 
            url=args.url, 
            token=args.token, 
            timeout=180
        )        
        fertilization_data_weather[row.Index] = bands

    fertilization_data_weather = pd.concat(fertilization_data_weather)


    # %%
    fertilization_data_weather_bands = fertilization_data_weather.loc[pd.IndexSlice[:,:,:,WEATHER_BANDS_NAMES]].copy()
    gerk_weather_bands_masked = {}

    for ix, row in tqdm(enumerate(fertilization_data.itertuples()), total=len(fertilization_data)):
        gerk_label = rasterio.open(os.path.join(GEOJSON_TIF_PATH, row.LABEL_TIF))

        gerk_weather_bands = np.ones(
            (len(WEATHER_BANDS_NAMES) * len(fertilization_data_weather_bands.columns), *gerk_label.read(1).shape)
        )
        gerk_weather_bands = np.multiply(
            gerk_weather_bands, 
            fertilization_data_weather_bands.xs(row.Index).values.reshape(-1, 1)[:, :, np.newaxis]
        )
        
        gerk_mask = np.ones_like(gerk_label.read(1))
        gerk_mask[gerk_label.read(1) == gerk_label.nodata] = 0

        gerk_weather_bands_masked[row.Index] = np.multiply(
            gerk_weather_bands,
            gerk_mask[np.newaxis, :, :]
        )

    fertilization_data.loc[:, 'weather_bands'] = pd.Series(gerk_weather_bands_masked)
    fertilization_data.to_pickle('./Targeted fertilization v3.labels.weather.pkl')
