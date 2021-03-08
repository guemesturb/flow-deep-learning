# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 23:26:49 2021
@author: aguemes
"""


import numpy as np
import matplotlib.pyplot as plt


def main():

    filename = f"{prediction_path}/Ret{Ret}_resolution_predictions_subsampling-{subsampling_lr:02d}_wall-channels-{channels}.npz" 

    with np.load(filename) as data:

        X_lr_target = data['X_lr_target']
        X_hr_target = data['X_hr_target']
        X_hr_predic = data['X_hr_predic']

    rows = 1
    cols = 3
    ration = 0.5
    fig_width_pt = 400
    inches_per_pt = 1.0 / 72.27               
    golden_mean = (5 ** 0.5 - 1.0) / 2.0         
    fig_width = fig_width_pt * inches_per_pt 
    fig_height = fig_width / cols * rows * ration

    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)

    dx_lr = Lx / nx * subsampling_lr
    dz_lr = Lz / nz * subsampling_lr
    
    x_lr = np.arange(-dx_lr/2, Lx , dx_lr)
    z_lr = np.arange(-dz_lr/2, Lz , dz_lr)

    dx_hr = Lx / nx
    dz_hr = Lz / nz
    x_hr = np.arange(-dx_hr/2, Lx , dx_hr)
    z_hr = np.arange(-dz_hr/2, Lz , dz_hr)

    axs[0,0].pcolor(x_lr, z_lr, X_lr_target[0,0,:,:], cmap='RdBu_r', vmin=-2, vmax=2)
    axs[0,1].pcolor(x_hr, z_hr, X_hr_target[0,0,:,:], cmap='RdBu_r', vmin=-2, vmax=2)
    axs[0,2].pcolor(x_hr, z_hr, X_hr_predic[0,0,:,:], cmap='RdBu_r', vmin=-2, vmax=2)

    fig.savefig(f'figs/Ret{Ret}_resolution_instantaneous-field_subsamplig_{subsampling_lr:02d}_wall-channels-{channels}.pdf', dpi=600)

    return

if __name__ == '__main__':

    Ret = 180
    epochs = 100
    yp_wall = 0
    yp_flow = 15
    channels = 1
    batch_size = 8
    n_prefetch = 4
    n_res_block = 16
    subsampling_hr = 1
    subsampling_lr = 8

    Lx = 4 * np.pi
    Lz = 2 * np.pi

    if Ret == 180:

        nx = 192
        nz = 192

        prediction_path = "/storage2/alejandro/urban/re180/predictions/"

    main()