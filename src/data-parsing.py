import pandas as pd

from tqdm.auto import tqdm
import argparse
import os

import st_image_toolkit as stt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Parsing ITC Sample Data')
    args = parser.parse_args()

    # %%
    # Data Parsing
    DATA_PATH = os.path.join('.', 'data')
    GEOJSON_TIF_PATH = os.path.join(DATA_PATH, 'Prescriptions_2023 10 27')

    # Fertilization Data (Main Table)
    fertilization_data = pd.read_excel(os.path.join(DATA_PATH,'Targeted fertilization v3.xlsx'))

    # Rename columns (for brevity)
    fertilization_data.rename({'SHP file name additional fertilization':'GERK_SHP'}, axis=1, inplace=True)

    # Getting GEOTIFF file paths
    fertilization_data[['NDVI_JSON', 'NDVI_TIF']] = fertilization_data['NDVI file name'].str.split(';').apply(pd.Series)
    fertilization_data['datetime'] = pd.to_datetime(
        fertilization_data.GERK_SHP.str.split('_').apply(lambda l: l[3]),
    )

    # Drop Un-necessary columns
    fertilization_data.drop(['NDVI file name'], axis=1, inplace=True)
    fertilization_data.set_index(['GERK', 'datetime'], inplace=True)

    # From domain experts...
    fertilization_data.drop(['No.'], axis=1, inplace=True)
    fertilization_data.loc[:, 'phase'] = fertilization_data.GERK_SHP.str.split('_PH').str[-1].str[0]    # OLD ONE --> .GERK_SHP.str.split('_').str[-1].str[2]

    # Save as Pickle (for future reference)
    fertilization_data.to_pickle('Targeted fertilization v3.pkl')

    tqdm.pandas()
    fertilization_data.loc[:, 'LABEL_TIF'] = fertilization_data.progress_apply(
        lambda l: stt.create_labels(
            os.path.join(GEOJSON_TIF_PATH, l.GERK_SHP),
            os.path.join(GEOJSON_TIF_PATH, l.NDVI_TIF.strip(' ')),
            os.path.join(GEOJSON_TIF_PATH, l.GERK_SHP.replace('zip', 'tif')),
        ), 
        axis=1
    )

    # Save as Pickle (for future reference)
    fertilization_data.dropna(subset=['LABEL_TIF']).to_pickle('./Targeted fertilization v3.labels.pkl')
