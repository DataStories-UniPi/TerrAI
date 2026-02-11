import pandas as pd
import numpy as np
from sklearn import metrics

# import pdb
import tqdm
import time
import torch
import torch.nn as nn

ROUND_DECIMALS = 5


class MSELoss_masked(torch.nn.MSELoss):
    def __init__(self, ignore_index, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = torch.isclose(target, self.ignore_index)
        return super().forward(input[~mask], target[~mask])


class RMSELoss_masked(torch.nn.Module):
    def __init__(self, eps=1e-3, ignore_index=None, **kwargs):
        super().__init__()
        self.mse = MSELoss_masked(ignore_index=ignore_index, **kwargs)
        self.eps = eps
        print(f'{self.eps=}')

    def forward(self, input, target):
        loss = torch.sqrt(self.mse(input, target) + self.eps)
        return loss


def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


def save_model(model, path, **kwargs):
    torch.save({
        'model_state_dict': model.state_dict(),
        **kwargs
    }, path)


def calc_loss(model, xb, yb, criterion, *args, **kwargs):
    y_pred = model(xb, *args)

    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        yb = yb.squeeze(1).long()
                
    loss = criterion(y_pred, yb.squeeze(dim=-2))
    return y_pred, loss
    

def model_backprop(model, xb, yb, criterion, optimizer, criterion_fun, *args, **kwargs):
    try:
        optimizer.zero_grad()
        _, loss = criterion_fun(model, xb, yb, criterion, *args, **kwargs)

        loss.backward()
        optimizer.step()
    except RuntimeError as err_runtime:
        print(err_runtime)
        # pdb.set_trace()
    return loss


def running_loss(loss, data_loader):
    loss = torch.Tensor(loss).sum()
    loss = loss / len(data_loader)
    return loss


def model_dev_loss(model, device, criterion, dev_loader, criterion_fun, criterion_fun_params):
    model.eval()
    with torch.no_grad():
        dev_loss = []
        for (xb, yb, *args) in (pbar := tqdm.tqdm(dev_loader, leave=False, total=len(dev_loader), dynamic_ncols=True)):
            if xb.shape[0] == 1: # Avoid division by zero in batch normalization 
                continue

            xb = xb.to(device).float()
            yb = yb.to(device).float()

            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                yb = yb.to(device).squeeze(1).long()

            args = (arg.to(device) for arg in args)
                
            # _, loss = calc_loss(model, xb, yb, criterion, *args)
            _, loss = criterion_fun(model, xb, yb, criterion, *args, **criterion_fun_params)
            dev_loss.append(loss)

            pbar.set_description(f'Dev Loss: {loss:.{ROUND_DECIMALS}f}')

    return running_loss(dev_loss, dev_loader)


def train_step(model, device, criterion, optimizer, train_loader, criterion_fun, criterion_fun_params):
    model.train()

    train_loss = []
    for j, (xb, yb, *args) in (pbar := tqdm.tqdm(enumerate(train_loader), leave=False, total=len(train_loader), dynamic_ncols=True)):
        if xb.shape[0] == 1: # Avoid division by zero in batch normalization 
            continue

        xb = xb.to(device).float()
        yb = yb.to(device).float()
        args = (arg.to(device) for arg in args)

        tr_loss = model_backprop(model, xb, yb, criterion, optimizer, criterion_fun, *args, **criterion_fun_params)
        train_loss.append(tr_loss)
        pbar.set_description(f'Train Loss: {tr_loss:.{ROUND_DECIMALS}f}')

    return running_loss(train_loss, train_loader)


def early_stopping(n_epochs_stop, min_loss, curr_loss, patience=5, min_delta=1e-4, save_best=False, **kwargs):
    if (min_loss - curr_loss) > min_delta:
        if save_best:
            print(f'Loss Decreased ({min_loss:.{ROUND_DECIMALS}f} -> {curr_loss:.{ROUND_DECIMALS}f}). Saving Model...', end=' ')
            save_model(**kwargs)
            print('Done!')

        return 0, curr_loss, False

    print(f'Loss Increased ({min_loss:.{ROUND_DECIMALS}f} -> {curr_loss:.{ROUND_DECIMALS}f}).')
    n_epochs_stop_ = n_epochs_stop + 1
    return n_epochs_stop_, min_loss, n_epochs_stop_ == patience


def train_model(model, device, criterion, optimizer, n_epochs,
                train_loader, dev_loader, criterion_fun=calc_loss, criterion_fun_params={}, evaluate_cycle=5, early_stop=True, save_current=True,
                evaluate_fun=None, evaluate_fun_params={}, early_stop_params={}, save_current_params={}):
    train_losses, dev_losses = [], []

    # Early Stopping Initial Param. Values
    min_loss, n_epochs_stop, stop = torch.tensor(float("Inf")), 0, False

    if save_current:
        save_path_template = save_current_params['path']

    # training loop
    for i in range(n_epochs):
        t_start = time.process_time()
        train_loss = train_step(model, device, criterion, optimizer, train_loader, criterion_fun, criterion_fun_params)
        dev_loss = model_dev_loss(model, device, criterion, dev_loader, criterion_fun, criterion_fun_params)
        t_end = time.process_time() - t_start

        train_losses.append(train_loss.numpy())
        dev_losses.append(dev_loss.numpy())

        epoch_summary = {
            'model': model,
            'epoch': i,
            'transform': train_loader.dataset.transform,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_losses,
            'dev_loss': dev_losses
        }

        if early_stop:
            early_stop_params.update(epoch_summary)
            n_epochs_stop, min_loss, stop = early_stopping(n_epochs_stop, min_loss, dev_loss, **early_stop_params)
        
        if save_current:
            save_current_params.update(epoch_summary)
            save_current_params['path'] = save_path_template.format(i)
            save_model(**save_current_params)

        print(f'Epoch #{i+1}/{n_epochs} | '
              f'Train Loss: {train_loss:.{ROUND_DECIMALS}f} | '
              f'Validation Loss: {dev_loss:.{ROUND_DECIMALS}f} | '
              f'Time Elapsed: {t_end:.{ROUND_DECIMALS}f}')

        if evaluate_cycle != -1 and i % evaluate_cycle == 0 and evaluate_fun is not None:
            evaluate_fun(model, device, criterion, dev_loader,
                         desc='Evaluation @ Dev Set...', **evaluate_fun_params)

        if stop:
            print(f'Training Stopped at Epoch #{i+1}')
            break
    return train_losses, dev_losses
