import numpy as np
import geopandas as gpd

import os
import fiona

import rasterio
from rasterio import features

import torch
import torch.nn.functional as F

import dataset as ds


def rvi_band(nir, red):
    '''Ratio Vegetation Index (RVI)'''
    nominator, denominator = nir, red
    return np.divide(nominator, denominator, out=np.zeros_like(nir), where=denominator != 0)


def grvi_band(nir, green):
    '''Green Ratio Vegetation Index (GRVI)'''
    nominator, denominator = nir, green
    return np.divide(nominator, denominator, out=np.zeros_like(nir), where=denominator != 0)


def msr_band(nir, red):
    '''Modified Simple Ratio (MSR) - Improves vegetation sensitivity'''
    nominator, denominator = (nir / red) - 1, np.sqrt((nir / red) + 1)
    return np.divide(nominator, denominator, out=np.zeros_like(nir), where=denominator != 0)  
    

def ndvi_band(nir, red):
    '''
        Normalized difference vegetation index (NDVI; Enhances contrast between soil and vegetation)
        Source: https://doi.org/10.1016/0034-4257(79)90013-0
    '''
    nominator, denominator = nir - red, nir + red
    return np.divide(nominator, denominator, out=np.zeros_like(nir), where=denominator != 0)


def wdrvi_band(nir, red):
    '''Wide dynamic range vegetation index (WDRVI)'''
    nominator, denominator = (0.1 * nir) - red, (0.1 * nir) + red
    return np.divide(nominator, denominator, out=np.zeros_like(nir), where=denominator != 0)


def gndvi_band(nir, green):
    '''
        Green normalized difference vegetation index (GNDVI)
        Source: https://doi.org/10.2134/agronj2001.933583x
    '''
    nominator, denominator = nir - green, nir + green
    return np.divide(nominator, denominator, out=np.zeros_like(nir), where=denominator != 0)


def bndvi_band(nir, blue):
    '''
        Blue Normalized Difference Vegetation Index (BNDVI)
        Source: https://doi.org/10.1016/S1672-6308(07)60027-4
    '''
    nominator, denominator = nir - blue, nir + blue
    return np.divide(nominator, denominator, out=np.zeros_like(nir), where=denominator != 0)


def gbndvi_band(nir, green, blue):
    '''
        Green-Blue Normalized Difference Vegetation Index (GBNDVI)
        Source: https://doi.org/10.1016/S1672-6308(07)60027-4
    '''
    nominator, denominator = nir - (green + blue), nir + (green + blue)
    return np.divide(nominator, denominator, out=np.zeros_like(nir), where=denominator != 0)


def grndvi_band(nir, green, red):
    '''
        Green-Blue Normalized Difference Vegetation Index (GBNDVI)
        Source: https://doi.org/10.1016/S1672-6308(07)60027-4
    '''
    nominator, denominator = nir - (green + red), nir + (green + red)
    return np.divide(nominator, denominator, out=np.zeros_like(nir), where=denominator != 0)


def rbndvi_band(nir, red, blue):
    '''
        Red-Blue Normalized Difference Vegetation Index (RBNDVI)
        Source: https://doi.org/10.1016/S1672-6308(07)60027-4
    '''
    nominator, denominator = nir - (red + blue), nir + (red + blue)
    return np.divide(nominator, denominator, out=np.zeros_like(nir), where=denominator != 0)


def evi_band(nir, red, blue):
    '''Enhanced vegetation index (EVI)'''
    nominator, denominator = 2.5 * (nir - red), nir + (6 * red) - (7.5 * blue) + 1
    return np.divide(nominator, denominator, out=np.zeros_like(nir), where=denominator != 0)


def create_labels(input_shapefile, template_raster, output_raster, bands=['Ammonitrat', 'Class']):
    # Forked from: https://gis.stackexchange.com/a/151861
    try:
        input_shapefile = gpd.read_file(input_shapefile)
        input_shapefile.columns = [x.title() if x != 'geometry' else x for x in input_shapefile.columns]
    except fiona.errors.DriverError as e:
        print(f'Missing Shapefile {input_shapefile}')
        return np.nan
    
    try:
        template = rasterio.open(template_raster)
    except rasterio.errors.RasterioIOError:
        print(f'Missing Raster {template_raster}')
        return np.nan
    
    input_shapefile.loc[:, 'area_m2'] = input_shapefile.to_crs(3857).area
    
    input_shapefile = input_shapefile.sort_values(
        'area_m2', ascending=True
    ).reset_index(
        level=0, drop=True
    ).reset_index(
        level=0, drop=False
    ).rename({'index':'Class'}, axis=1)

    # copy and update the metadata from the input raster for the output
    meta = template.meta.copy()
    meta.update()

    # Now burn the features into the raster and save it
    with rasterio.open(output_raster, 'w+', **meta) as out:
        for ix, band in enumerate(bands, start=1):
            out_arr = out.read(ix)

            # this is where we create a generator of geom, value pairs to use in rasterizing
            # shapes = ((geom, value) for geom, value in zip(input_shapefile.to_crs(meta['crs']).geometry, input_shapefile.iloc[:, ix-1]))
            shapes = ((geom, value) for geom, value in zip(input_shapefile.to_crs(meta['crs']).geometry, input_shapefile[band]))
            burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
            out.write_band(ix, burned)

    return os.path.basename(output_raster)


def create_weather_band(geotiff_dir, gerk, wb_stations, bands, **kwargs):    
    # ### Get weather information for each GERK (via weighted nearest neighbor query).
    gerk_geom = gpd.read_file(os.path.join(geotiff_dir, gerk.GERK_SHP))
    gerk_centroid = gerk_geom.unary_union.centroid

    _, wb_ix = wb_stations.sindex.nearest(gerk_centroid)
    sensor_id = wb_stations.iloc[wb_ix]['ObjectId'].values[0]

    gerk_weather = ds.wb_query_weather(
        sensor_id=sensor_id, 
        timestamp=gerk.Index[1], 
        weather_feats=bands,
        **kwargs
    )

    return gerk_weather


def segment_images(img, kernel_size, stride):
    '''
    Forked from: https://discuss.pytorch.org/t/fold-and-unfold-how-do-i-put-this-image-tensor-back-together-again/97374/4

    Input
    =====
      * img: torch.Tensor of size (B, C, H, W); B:Batch, C:Channels, H:Height, W:Width
      * kernel_size: the size (K) of the output patches
      * stride: the overlap of the kernel
    
    Output
    =====
      * patches: torch.Tensor of size (B, n_windows_H x n_windows_W, C, K, K);
    '''
    B, C, H, W = img.shape

    windows = img.unfold(2, size=kernel_size, step=stride).unfold(3, size=kernel_size, step=stride)  # torch.Size([B, C, n_windows_H, n_windows_W, K, K])
    
    # Create Patches from Image
    patches = windows.contiguous().view(B, C, -1, kernel_size, kernel_size)  # torch.Size([B, C, n_windows_H x n_windows_W, K, K])

    return patches.permute(0, 2, 1, 3, 4), {'C':C, 'H':H, 'W':W, 'n_windows_H':windows.shape[2], 'n_windows_W':windows.shape[3]}


def unary_union(patches, C, H, W, kernel_size, stride):
    '''
    Forked from: https://discuss.pytorch.org/t/fold-and-unfold-how-do-i-put-this-image-tensor-back-together-again/97374/4

    Input
    =====
      * patches: Contiguous torch.Tensor of size [B, n_windows_H x n_windows_W, C, K, K]; B:Batch, C:Channels, K:Kernel
      * C: Number of channels (i.e., bands)
      * H: torch.Tensor height 
      * W: torch.Tensor width
      * kernel_size: the size of the kernel
      * stride: the overlap of the kernel
    
    Output
    =====
      * img: torch.Tensor of size [B, C, H, W];
    '''
    # print(f'{patches.shape=}')
    patches_flat = patches.reshape(*patches.shape[:2], -1).permute(0, 2, 1)  # torch.Size([B, C x K x K, n_windows_H x n_windows_W])

    # print(f'{patches_flat.shape=}')
    folded = F.fold(patches_flat, output_size=(H, W), kernel_size=kernel_size, stride=stride)  # Construct initial image from patches (reduce with sum)
    counts = F.fold(torch.ones_like(patches_flat), output_size=(H, W), kernel_size=kernel_size, stride=stride)  # Count the overlaps of each patch in the constructed image
    
    # Divide the folded tensor by the counts tensor to get the original pixel values.
    result = (folded / counts)
    return result.view(patches.shape[0], C, H, W)
