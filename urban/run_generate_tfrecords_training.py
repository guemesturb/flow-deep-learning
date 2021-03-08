# -*- coding: utf-8 -*-
"""
Created on Wed Mar 03 13:44:23 2021
@author: aguemes
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"]=''
import h5py
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

    U = h5py.File(f"{path_mat}/U.mat", 'r')
    W = h5py.File(f"{path_mat}/W.mat", 'r')

    x = np.reshape(U['x'], (nz, ny, nx))
    y = np.reshape(U['y'], (nz, ny, nx))
    z = np.reshape(U['z'], (nz, ny, nx))

    delta_x_1 = y[0,-2,0] - y[0,-1,0]
    delta_x_2 = y[0,-3,0] - y[0,-2,0]

    n_samples = U['U'].shape[0]

    n_sets = int(n_samples / max_samples_per_tfr)

    tfrecords_filename_base = save_path + f'urban_dataset{dataset:02d}_'

    i_samp = 0

    for n_set in range(n_sets):

        if (n_set + 1) * max_samples_per_tfr > n_samples:

            samples_per_tfr = np.sum(n_samples) - n_set * max_samples_per_tfr

        else:

            samples_per_tfr = max_samples_per_tfr

        tfrecords_filename = tfrecords_filename_base + f'file_{n_set+1:03d}-of-{n_sets:03d}_samples{samples_per_tfr:03d}.tfrecords'
        writer = tf.io.TFRecordWriter(tfrecords_filename)

        for idx in range(samples_per_tfr):

            u = np.reshape(U[U['U'][i_samp][0]][:], (nz, ny, nx))        
            w = np.reshape(W[W['W'][i_samp][0]][:], (nz, ny, nx))

            dudy = (-u[:,-3,:] + 4 * u[:,-2,:] - 3 * u[:, -1, :]) / (3 * delta_x_2 - delta_x_1)
            dwdy = (-w[:,-3,:] + 4 * w[:,-2,:] - 3 * w[:, -1, :]) / (3 * delta_x_2 - delta_x_1)

            wall_hr_raw1 = np.float32(dudy).flatten().tolist()
            wall_hr_raw2 = np.float32(dwdy).flatten().tolist()

            example = tf.train.Example(
                features = tf.train.Features(
                    feature={
                        'i_samp': _int64_feature(i_samp),
                        'n_x':  _int64_feature(nx),
                        'n_y':  _int64_feature(ny),
                        'n_z':  _int64_feature(nz),
                        'wall_hr_raw1':  _floatarray_feature(wall_hr_raw1),
                        'wall_hr_raw2':  _floatarray_feature(wall_hr_raw2),
                    }
                )
            )

            writer.write(example.SerializeToString())

            i_samp += 1 

        writer.close()

        print(f'Closing file {tfrecords_filename}')

    # import copy
    # cmap = copy.copy(matplotlib.cm.get_cmap('RdBu_r'))
    # cmap.set_bad(color='r')
    # print(n_samples)

    # for idx in range(n_samples):

    #     u = np.reshape(U[U['U'][idx][0]][:], (nz, ny, nx))        
    #     w = np.reshape(W[W['W'][idx][0]][:], (nz, ny, nx))

    #     a = u[:,-30,:]
    #     a[a==0] = np.nan
    #     print(a[100,:30])
    #     print(a[:100,20])
    #     print(a[100,30:50])

    #     dudy = (-u[:,-3,:] + 4 * u[:,-2,:] - 3 * u[:, -1, :]) / (3 * delta_x_2 - delta_x_1)
    #     dwdy = (-w[:,-3,:] + 4 * w[:,-2,:] - 3 * w[:, -1, :]) / (3 * delta_x_2 - delta_x_1)
        
    #     # plt.contourf(x[:,0,:],z[:,0,:],dudy-dudy.mean(),cmap='RdBu_r')
    #     fig, ax = plt.subplots()
    #     ax.contourf(x[:,0,:],z[:,0,:],a,cmap=cmap,vmin=-1,vmax=1, levels=np.linspace(-1,1,21))
    #     ax.set_facecolor('k')
    #     ax.axis('equal')
    #     ax.set_ylim([-1.5,1.5])
    #     ax.set_xlim([-1,5])
    #     fig.savefig('test.png')
    #     fig, ax = plt.subplots()
    #     ax.contourf(x[83:117,0,20:38],z[83:117,0,20:38],a[83:117,20:38],cmap=cmap,vmin=-1,vmax=1, levels=np.linspace(-1,1,21))
    #     ax.set_facecolor('k')
    #     ax.axis('equal')
    #     ax.set_ylim([-1.5,1.5])
    #     ax.set_xlim([-1,5])
    #     fig.savefig('test2.png')
    #     break

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

    dataset = 3

    path_mat = f"/storage2/alejandro/urban/urban/DATASET{dataset}"
    save_path = "/storage2/alejandro/urban/urban/train/.tfrecords/"
    
    main()