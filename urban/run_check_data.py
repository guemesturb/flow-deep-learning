# -*- coding: utf-8 -*-
"""
Created on Wed Mar 03 13:44:23 2021
@author: aguemes
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"]=''
import h5py
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
physical_devices = tf.config.list_physical_devices('GPU')
availale_GPUs = len(physical_devices) 

print('Using TensorFlow version: ', tf.__version__, ', GPU:', availale_GPUs)
print('Using Keras version: ', tf.keras.__version__)

if physical_devices:

  try:

    for gpu in physical_devices:

      tf.config.experimental.set_memory_growth(gpu, True)

  except RuntimeError as e:

    print(e)


def main():

    n_samples = []

    for dataset in datasets:

        path_mat = f"/storage2/alejandro/urban/urban/DATASET{dataset}"

        U = h5py.File(f"{path_mat}/U.mat", 'r')
        V = h5py.File(f"{path_mat}/W.mat", 'r')
        W = h5py.File(f"{path_mat}/W.mat", 'r')

        x = np.reshape(U['x'], (nz, ny, nx))[:, ::-1, :]
        y = np.reshape(U['y'], (nz, ny, nx))[:, ::-1, :]
        z = np.reshape(U['z'], (nz, ny, nx))[:, ::-1, :]

        n_samples.append(U['U'].shape[0])
    
    n_samples_tot = np.sum(n_samples)

    index = []
        
    for idx in np.arange(0,1.01,0.05):
        
        index.append(next(x for x, val in enumerate(list(y[0,:,0])) if val >= idx))

    index += [389] + list(range(391, 400)) 
    
    idx = 0

    for dataset, n_samp_dataset in zip(datasets, n_samples):

        path_mat = f"/storage2/alejandro/urban/urban/DATASET{dataset}"

        print(path_mat)

        U = h5py.File(f"{path_mat}/U.mat", 'r')
        V = h5py.File(f"{path_mat}/V.mat", 'r')
        W = h5py.File(f"{path_mat}/W.mat", 'r')

        u    = np.zeros((n_samp_dataset, nz_b, len(index), nx_b))
        v    = np.zeros((n_samp_dataset, nz_b, len(index), nx_b))
        w    = np.zeros((n_samp_dataset, nz_b, len(index), nx_b))

        for idx_samp in tqdm(range(n_samp_dataset)):

            u_temp = np.reshape(U[U['U'][idx_samp][0]][:], (nz, ny, nx))[83:117,::-1,21:38]
            v_temp = np.reshape(V[V['V'][idx_samp][0]][:], (nz, ny, nx))[83:117,::-1,21:38]  
            w_temp = np.reshape(W[W['W'][idx_samp][0]][:], (nz, ny, nx))[83:117,::-1,21:38]

            u[idx_samp, :, :, :] = u_temp[:, index, :]
            v[idx_samp, :, :, :] = v_temp[:, index, :]  
            w[idx_samp, :, :, :] = w_temp[:, index, :]  

            idx += 1

        u_mean = np.mean(u)
        v_mean = np.mean(v)
        w_mean = np.mean(w)
        print(u_mean)
        print(v_mean)
        print(w_mean)    
    

    return




if __name__ == '__main__':

    max_samples_per_tfr = 100

    nx = 100
    nz = 200
    ny = 400

    nx_b = 17
    ny_b = 400
    nz_b = 34
    

    datasets = [1,3,4,5,6,7, 8, 9, 10]
    
    save_path = "/storage2/alejandro/urban/urban/train/.tfrecords/"
    
    main()
