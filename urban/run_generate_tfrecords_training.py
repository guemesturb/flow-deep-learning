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
    
    delta_y_1 = y[0,1,0] - y[0,0,0]
    delta_y_2 = y[0,2,0] - y[0,1,0]

    u    = np.zeros((n_samples_tot, nz_b, len(index), nx_b))
    v    = np.zeros((n_samples_tot, nz_b, len(index), nx_b))
    w    = np.zeros((n_samples_tot, nz_b, len(index), nx_b))
    dudy = np.zeros((n_samples_tot, nz_b, nx_b))
    dwdy = np.zeros((n_samples_tot, nz_b, nx_b))

    idx = 0

    for dataset, n_samp_dataset in zip(datasets, n_samples):

        path_mat = f"/storage2/alejandro/urban/urban/DATASET{dataset}"

        print(path_mat)

        U = h5py.File(f"{path_mat}/U.mat", 'r')
        V = h5py.File(f"{path_mat}/V.mat", 'r')
        W = h5py.File(f"{path_mat}/W.mat", 'r')

        for idx_samp in tqdm(range(n_samp_dataset)):

            u_temp = np.reshape(U[U['U'][idx_samp][0]][:], (nz, ny, nx))[83:117,::-1,21:38]
            v_temp = np.reshape(V[V['V'][idx_samp][0]][:], (nz, ny, nx))[83:117,::-1,21:38]  
            w_temp = np.reshape(W[W['W'][idx_samp][0]][:], (nz, ny, nx))[83:117,::-1,21:38]

            u[idx, :, :, :] = u_temp[:, index, :]
            v[idx, :, :, :] = v_temp[:, index, :]  
            w[idx, :, :, :] = w_temp[:, index, :]  

            dudy[idx, :, :] = (-u_temp[:, 2, :] + 4 * u_temp[:, 1, :] - 3 * u_temp[:, 0, :]) / (3 * delta_y_2 - delta_y_1)
            dwdy[idx, :, :] = (-w_temp[:, 2, :] + 4 * w_temp[:, 1, :] - 3 * w_temp[:, 0, :]) / (3 * delta_y_2 - delta_y_1)
            
            idx += 1


    # Take into account grid volume

    y_reduced = y[0, index, 0]

    delta_x = x[0,0,2] - x[0,0,1]
    delta_z = z[2,0,0] - z[1,0,0]
    delta_y = np.expand_dims(np.array([0.5 * float(y_reduced[1]-y_reduced[0])] + list(0.5 * (y_reduced[2:]-y_reduced[:-2])) + [0.5 * float(y_reduced[-1]-y_reduced[-2])]), axis=(0,2))

    delta_v = delta_y * delta_x * delta_z

    filename = f'/storage2/alejandro/urban/urban/grid.npz'
    np.savez(filename, x=x[83:117,index,21:38], y=y[83:117,index,21:38], z=z[83:117,index,21:38], delta_v=delta_v)
    
    # Compute statistics

    u_mean = np.expand_dims(np.mean(u, axis=0), axis=0)
    v_mean = np.expand_dims(np.mean(v, axis=0), axis=0)
    w_mean = np.expand_dims(np.mean(w, axis=0), axis=0)
    u_std = np.expand_dims(np.std(u, axis=0), axis=0)
    v_std = np.expand_dims(np.std(v, axis=0), axis=0)
    w_std = np.expand_dims(np.std(w, axis=0), axis=0)
    
    avg_flow = np.concatenate((u_mean, v_mean, w_mean), axis = 0)
    std_flow = np.concatenate((u_std, v_std, w_std), axis = 0)

    filename = f'/storage2/alejandro/urban/urban/statistics_flow.npz'
    np.savez(filename, avg_flow=avg_flow, std_flow=std_flow)

    dudy_mean = np.expand_dims(np.mean(dudy, axis=0), axis=0)
    dwdy_mean = np.expand_dims(np.mean(dwdy, axis=0), axis=0)
    dudy_std = np.expand_dims(np.std(dudy, axis=0), axis=0)
    dwdy_std = np.expand_dims(np.std(dwdy, axis=0), axis=0)

    avg_wall = np.concatenate((dudy_mean, dwdy_mean), axis = 0)
    std_wall = np.concatenate((dudy_std, dwdy_std), axis = 0)

    filename = f'/storage2/alejandro/urban/urban/statistics_wall.npz'
    np.savez(filename, avg_wall=avg_wall, std_wall=std_wall)

    # Save mean values 
            
    C = np.concatenate(
        (
            np.reshape((u - u_mean) * delta_v, (n_samples_tot, nz_b * len(index) * nx_b)),
            np.reshape((v - v_mean) * delta_v, (n_samples_tot, nz_b * len(index) * nx_b)),
            np.reshape((w - w_mean) * delta_v, (n_samples_tot, nz_b * len(index) * nx_b)),
        ),
        axis=1
    )

    S = np.matmul(C, C.T)

    psi, delta, psiT = np.linalg.svd(S)

    sigma = np.diag(delta ** 0.5)

    phi = np.matmul(np.linalg.pinv(sigma), np.matmul(psiT,C))

    filename = f'/storage2/alejandro/urban/urban/pod_{nz_b}x{len(index)}x{nx_b}.npz'
    np.savez(filename, psi=psi, sigma=sigma, phi=phi)

    # Generate tfrecords

    n_sets = int(np.ceil(n_samples_tot / max_samples_per_tfr))

    tfrecords_filename_base = save_path + f'urban_datasets_'

    i_samp = 0

    for n_set in range(n_sets):

        if (n_set + 1) * max_samples_per_tfr > n_samples_tot:

            samples_per_tfr = n_samples_tot - n_set * max_samples_per_tfr

        else:

            samples_per_tfr = max_samples_per_tfr

        tfrecords_filename = tfrecords_filename_base + f'file_{n_set+1:03d}-of-{n_sets:03d}_samples{samples_per_tfr:03d}.tfrecords'
        writer = tf.io.TFRecordWriter(tfrecords_filename)

        for idx in range(samples_per_tfr):

            wall_hr_raw1 = np.float32(dudy[i_samp, :, :]).flatten().tolist()
            wall_hr_raw2 = np.float32(dwdy[i_samp, :, :]).flatten().tolist()

            flow_hr_raw1 = np.float32(u[i_samp, :, :, :]).flatten().tolist()
            flow_hr_raw2 = np.float32(v[i_samp, :, :, :]).flatten().tolist()
            flow_hr_raw3 = np.float32(w[i_samp, :, :, :]).flatten().tolist()

            flow_psi_raw = np.float32(psi[i_samp, :]).flatten().tolist()

            example = tf.train.Example(
                features = tf.train.Features(
                    feature={
                        'i_samp': _int64_feature(i_samp),
                        'n_x':  _int64_feature(nx),
                        'n_y':  _int64_feature(ny),
                        'n_z':  _int64_feature(nz),
                        'wall_hr_raw1':  _floatarray_feature(wall_hr_raw1),
                        'wall_hr_raw2':  _floatarray_feature(wall_hr_raw2),
                        'flow_hr_raw1':  _floatarray_feature(flow_hr_raw1),
                        'flow_hr_raw2':  _floatarray_feature(flow_hr_raw2),
                        'flow_hr_raw3':  _floatarray_feature(flow_hr_raw3),
                        'flow_psi_raw':  _floatarray_feature(flow_psi_raw),
                    }
                )
            )

            writer.write(example.SerializeToString())

            i_samp += 1 

        writer.close()

        print(f'Closing file {tfrecords_filename}')
    

    return


def _bytes_feature(value):

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):

    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _floatarray_feature(value):

    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


if __name__ == '__main__':

    max_samples_per_tfr = 100

    nx = 100
    nz = 200
    ny = 400

    nx_b = 17
    ny_b = 400
    nz_b = 34
    

    datasets = [1, 3, 4, 5, 6, 7, 8, 9, 10]
    
    save_path = "/storage2/alejandro/urban/urban/train/.tfrecords/"
    
    main()
