import torch
import numpy as np

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import st_image_toolkit as stt


def weighted_sum(results, metric):
    # Weigh accuracy of each client by number of examples used
    metric_aggregated = [r.metrics[metric] * r.num_examples for _, r in results]
    examples = [r.num_examples for _, r in results]

    # Aggregate and print custom metric
    return sum(metric_aggregated) / sum(examples)


def model_inference(model, raster_data, data_transform, kernel, stride, device=torch.device('cpu'), no_data=-9999):
    pixel_mask = raster_data.output > 0

    pred_patches = model(
        data_transform(
            raster_data.input_patches.squeeze()
        ).to(device)
    )

    pred_union = torch.nan_to_num(
        stt.unary_union(
            pred_patches.unsqueeze(0),
            C=raster_data.output_patches_metadata['C'],
            H=raster_data.output_patches_metadata['H'],
            W=raster_data.output_patches_metadata['W'],
            kernel_size=kernel,
            stride=stride
        ),
        nan=no_data
    )
    pred_union[~pixel_mask] = no_data

    return pred_union


def visualize_model_results(raster_ndvi_band, raster_target_band, raster_pred_band, fig, axes, cmap):
    vmin, vmax = min(np.nanmin(raster_target_band.squeeze()), np.nanmin(raster_pred_band.squeeze())),\
                 max(np.nanmax(raster_target_band.squeeze()), np.nanmax(raster_pred_band.squeeze()))

    axes[0].set_title('BNDVI')
    axes[1].set_title('Actual Prescription')
    axes[2].set_title('Predicted Prescription')

    divider = make_axes_locatable(axes[0])
    cax_in = divider.append_axes('right', size='10%', pad=0.05)

    divider = make_axes_locatable(axes[1])
    cax_out = divider.append_axes('right', size='10%', pad=0.05)

    divider = make_axes_locatable(axes[2])
    cax_pred = divider.append_axes('right', size='10%', pad=0.05)

    map_in = axes[0].imshow(raster_ndvi_band.squeeze(), cmap=cmap, vmin=0, vmax=1)
    map_out = axes[1].imshow(raster_target_band.squeeze(), cmap='YlGn', vmin=vmin, vmax=vmax)
    map_pred = axes[2].imshow(raster_pred_band.squeeze(), cmap='YlGn', vmin=vmin, vmax=vmax)

    fig.colorbar(map_in, cax=cax_in, orientation='vertical')
    fig.colorbar(map_out, cax=cax_out, orientation='vertical')
    fig.colorbar(map_pred, cax=cax_pred, orientation='vertical')

    return axes
