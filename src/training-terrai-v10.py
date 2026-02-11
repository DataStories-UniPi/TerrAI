import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_percentage_error
import argparse
import os
import re 

import matplotlib
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T

import dataset as ds
import models as ml
import helper as hl
import train as tr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Targeted Fertilization Data Preprocessing')
    # Data Params
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
    # Model Params
    parser.add_argument('--bs', help='Batch Size', default=64, type=int, required=False)
    # parser.add_argument('--in_channels', help='Number of Input Bands', default=18, type=int, required=False)
    parser.add_argument('--out_channels', help='Number of Output Bands', default=1, type=int, required=False)
    parser.add_argument('--hidden_channels', help='Number of Hidden Channels', default='16,64', type=str, required=False)
    parser.add_argument('--dropout', help='Dropout rate at bottleneck (i.e., middle) layer', default=2e-1, type=float, required=False)
    parser.add_argument('--use_augmentation', help='Use data augmentation techniques', action='store_true')
    # Train Params
    parser.add_argument('--gpuid', help='GPU ID', default=0, type=int, required=False)
    parser.add_argument('--eta', help='Learning Rate', default=1e-3, type=float, required=False)
    parser.add_argument('--epochs', help='Number of Training Epochs', default=500, type=int, required=False)
    parser.add_argument('--patience', help='Number of Patience Epochs (for Early Stopping)', default=10, type=int, required=False)
    parser.add_argument('--skip_train', help='Skip training; Evaluate best model @ Test Set', action='store_true')
    args = parser.parse_args()


    # %%
    # Data Parsing
    DATA_PATH = os.path.join('.', 'data')
    GEOJSON_TIF_PATH = os.path.join(DATA_PATH, 'Prescriptions_2023 10 27')
    
    NO_DATA_VALUE = -9999

    WB_WEATHER_BANDS_NAMES = [
        'min T', 'mean T', 'T', 'max T',
        'min rel. hum.', 'mean rel. hum.', 'rel. hum.', 'max rel. hum.',
        'min pressure', 'mean pressure', 'max pressure',
        'max gust', 'wind direction', 'wind speed',
        'precipitation', 'global energy'
    ]
    
    SPECTRAL_BANDS_NAMES, WB_WEATHER_BANDS_RS_NAMES = [
        'NIR', 'R', 'G', 'B', *VI_BANDS_NAMES
    ], [
        f'{band}; T{time}' for band, time in np.array(
            np.meshgrid(WB_WEATHER_BANDS_NAMES, ['0', '8', '16'])
        ).T.reshape(-1, 2)
    ]

    ALL_FEATURES = [*SPECTRAL_BANDS_NAMES, *WB_WEATHER_BANDS_RS_NAMES, 'AMMONITRAT']
    
    # # Feature Selection
    selected_features = [*args.spectral_bands, *args.weather_bands]
    selected_features_ix = [ALL_FEATURES.index(i) for i in selected_features]

    # Parse Dataset (Fertilization Data; Main Table)
    fertilization_data = pd.read_pickle(os.path.join(DATA_PATH, 'pkl', './Targeted fertilization v3.labels.weather.pkl'))
    
    raster_dataset = pd.read_pickle(os.path.join(DATA_PATH, 'pkl', './Targeted fertilization v3.raster_dataset.weather.v3.pkl'))
    patches_dataset_clean = pd.read_pickle(os.path.join(DATA_PATH, 'pkl', f'./Targeted fertilization v3.patches_dataset.kernel={args.kernel}_stride={args.stride}_bands={len(selected_features_ix)}.weather.land={"-".join(args.lands)}_ph={"-".join(args.phases)}.v3.pkl'))


    # %%
    # Splitting Dataset to Train/Dev/Test based on GERKs 
    # (80% Train; 20% Test)
    gerks = patches_dataset_clean.index.get_level_values(0).unique() 

    gerk_train_ix, gerk_dev_ix, gerk_test_ix = ds.__train_test_split(
        gerks, 0.2, 0.2, seed=3407, stratify=False, shuffle=True
    )

    gerk_train_dev, gerk_test = gerks[gerk_train_ix + gerk_dev_ix], gerks[gerk_test_ix]

    # Save Train/Dev/Test GERKs
    pd.Series({
        'Train': gerks[gerk_train_ix], 
        'Dev': gerks[gerk_dev_ix],
        'Test': gerks[gerk_test_ix]
    }).to_pickle(os.path.join(DATA_PATH, 'pkl', f'./Targeted fertilization v3.patches_dataset.kernel={args.kernel}_stride={args.stride}_bands={len(selected_features_ix)}.weather.land={"-".join(args.lands)}_ph={"-".join(args.phases)}.v3.train-dev-test-splits.pkl'))

    # %%
    # Getting the correponding raster patches
    patches_dataset_train_dev, patches_dataset_test = patches_dataset_clean.loc[
        pd.IndexSlice[gerk_train_dev, :, :]
    ].copy(), patches_dataset_clean.loc[
        pd.IndexSlice[gerk_test, :, :]
    ].copy()

    # ## Plotting the distribution of [mean] Ammonitrat per Patch 
    fig, ax = plt.subplots(1, 1, figsize=(4, 4/1.618))

    out = pd.cut(
        patches_dataset_train_dev.output_patches.apply(
            lambda l: torch.mean(l.squeeze(0).squeeze(1), dim=(1,2)).numpy().tolist()
        ).explode(),
        bins=[NO_DATA_VALUE, 0, 50, 100, 150, 200, 250, 300], include_lowest=True
    )
    out.value_counts(sort=False).plot.bar(rot=45, ax=ax)
    ax.set_yscale('log')
    plt.savefig(os.path.join(DATA_PATH, 'fig', 'patch_dataset_ammonitrat_distribution.png'), dpi=300, bbox_inches='tight')

    # ## Split Patches to Train/Dev sets (80%/20%), stratified by (discretized/binned) Ammonitrat quantity
    patches_dataset_train, patches_dataset_dev = ds.train_test_split(
        patches_dataset_train_dev.explode(['input_patches', 'output_patches']).explode(['input_patches', 'output_patches']),
        test_size=0.2,
        random_state=3407,
        stratify=out
    )


    # %%
    # Create Torch Dataset and Dataloaders
    transforms = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5)
    ]) if args.use_augmentation else None

    patches_dataset_train_th = ds.ITC_Patches(
        [arr.unsqueeze(0) for arr in patches_dataset_train.input_patches.values.tolist()], 
        [arr.unsqueeze(0) for arr in patches_dataset_train.output_patches.values.tolist()],
        augmentation = transforms
    )
    patches_dataset_dev_th = ds.ITC_Patches(
        [arr.unsqueeze(0) for arr in patches_dataset_dev.input_patches.values.tolist()], 
        [arr.unsqueeze(0) for arr in patches_dataset_dev.output_patches.values.tolist()], 
        transform = patches_dataset_train_th.transform,
        augmentation = transforms
    )
    patches_dataset_test_th = ds.ITC_Patches(
        [arr.squeeze(0) for arr in patches_dataset_test.input_patches.values.tolist()], 
        [arr.squeeze(0) for arr in patches_dataset_test.output_patches.values.tolist()], 
        transform = patches_dataset_train_th.transform
    )

    patches_dataset_train_th_loader = torch.utils.data.DataLoader(patches_dataset_train_th, batch_size=args.bs, shuffle=True)
    patches_dataset_dev_th_loader = torch.utils.data.DataLoader(patches_dataset_dev_th, batch_size=args.bs, shuffle=True)
    patches_dataset_test_th_loader = torch.utils.data.DataLoader(patches_dataset_test_th, batch_size=args.bs, shuffle=False)


    # %%
    # Instantiate DL Model 
    model_params = dict(
        in_channels=len(selected_features_ix),
        out_channels=args.out_channels,
        hidden_channels=[int(i) for i in re.split(",\s*", args.hidden_channels)],
        dropout=args.dropout
    )

    device = torch.device(f'cuda:{args.gpuid}') if torch.cuda.is_available() else torch.device('cpu')
    model = ml.UNet(**model_params).to(device)
    print(model)


    # %%
    # Train DL Model 
    model_name_base = f"UNet_"+\
                      f"in-channels={model_params['in_channels']}_"+\
                      f"hidden-channels={'_'.join(map(str, model_params['hidden_channels']))}_"+\
                      f"out-channels={model_params['out_channels']}_"+\
                      f"patches-kernel={args.kernel}_stride={args.stride}_"+\
                      f"batch-size={args.bs}_{'ITC-sample-data'}_"+\
                      f"{'dropout_p={}_'.format(str(args.dropout).replace('.', '_')) if args.dropout else ''}"+\
                      f"patience={args.patience}_"+\
                      f"{'aug' if args.use_augmentation else ''}_"+\
                      f"land={'-'.join(args.lands)}_ph={'-'.join(args.phases)}"+\
                       ".v12.epoch{0}.pth"
    model_name_base_dir = os.path.join('.', 'data', 'pth', f'{model_name_base.split(".")[0]}')    
    os.makedirs(model_name_base_dir, exist_ok=True)

    save_path_epoch = os.path.join(model_name_base_dir, model_name_base)
    save_path_best = os.path.join(model_name_base_dir, model_name_base.format('best'))

    early_stop_params = dict(
        patience=args.patience,
        save_best=True,
        path=save_path_best
    )

    save_current_params = dict(
        path=save_path_epoch
    )

    # Forked from: https://discuss.pytorch.org/t/when-to-use-ignore-index/5935
    criterion = tr.RMSELoss_masked(ignore_index=torch.tensor([NO_DATA_VALUE]).to(device).float())
    # criterion = tr.RMSELoss_masked(ignore_index=torch.tensor([0]).to(device).float())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.eta)

    print(f'Model Path: {save_path_epoch}')

    if not args.skip_train:
        train_loss, dev_loss = tr.train_model(
            model, device, criterion, optimizer, args.epochs,
            patches_dataset_train_th_loader, patches_dataset_dev_th_loader, 
            evaluate_cycle=5, early_stop=True, save_current=True, 
            early_stop_params=early_stop_params, 
            save_current_params=save_current_params,
            evaluate_fun=None, evaluate_fun_params={}
        )


    # %% 
    # # Best Model Evaluation
    best_model = ml.UNet(**model_params).to(device)
    best_model_dict = torch.load(save_path_best)

    best_model.load_state_dict(best_model_dict['model_state_dict'])

    test_loss = tr.model_dev_loss(best_model, device=device, criterion=criterion, dev_loader=patches_dataset_test_th_loader)
    print(f'Best Model | Test Loss: {test_loss}')


    fig, ax = plt.subplots(len(patches_dataset_test), 3, figsize=(15, 2.5 * len(patches_dataset_test)))
    cmap = matplotlib.cm.RdYlGn
    cmap.set_bad(color='black')

    for axes, raster_data in zip(ax, patches_dataset_test.itertuples()):
        model_pred = hl.model_inference(
            best_model, raster_data, best_model_dict['transform'], args.kernel, args.stride, device=device
        )
        raster_metadata = fertilization_data.xs(raster_data.Index[:2])
        
        gerk, crop, phase, timestamp = raster_data.Index[0],\
                                    raster_metadata['Crop'].values[0], raster_metadata['phase'].values[0],\
                                    raster_data.Index[1].date()

        raster_ndvi_band, raster_target_band, raster_pred_band = np.copy(raster_data.input[:, 4]), np.copy(raster_data.output), np.copy(model_pred.detach().cpu().numpy())
        
        mape = mean_absolute_percentage_error(
            raster_target_band[~(raster_pred_band == NO_DATA_VALUE)],
            raster_pred_band[~(raster_pred_band == NO_DATA_VALUE)]
        )

        # raster_ndvi_band[raster_ndvi_band == 0] = np.nan
        raster_target_band[raster_target_band == NO_DATA_VALUE] = np.nan
        raster_pred_band[raster_pred_band == NO_DATA_VALUE] = np.nan

        hl.visualize_model_results(
            raster_ndvi_band, raster_target_band, raster_pred_band, fig, axes, cmap
        )

        axes[0].set_title(f"Input Soil-Health tensor ({axes[0].get_title()})")
        axes[1].set_title(f"GERK: {gerk} | {crop} | Phase: {phase}; {timestamp}\n{axes[1].get_title()} Map")
        axes[2].set_title(f"{axes[2].get_title()} Map | MAPE: {mape:.3f}")

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_PATH, 'fig', f'{model_name_base[:-13]}.test_set.png'), dpi=600)
