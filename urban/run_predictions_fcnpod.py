# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 15:55:18 2021
@author: aguemes
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import re
import time
import numpy as np
import tensorflow as tf
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

from tensorflow import keras
from tensorflow.keras import layers

def main():

    model_name = f"{models_path}flow-reconstruction_fcnpod_predictor.tf"
    predictor = keras.models.load_model(model_name)

    dataset_train, dataset_valid = load_tfrecords(save_path, validation_split, batch_size, n_prefetch, shuffle_buffer)

    n_samp = 500
    
    X_hr_target = np.zeros((n_samp, channels, nz, nx))
    Y_hr_target = np.zeros((n_samp, 3, nz, ny, nx))
    Y_hr_predic = np.zeros((n_samp, 3, nz, ny, nx))
    psi_predic = np.zeros((n_samp, n_required_modes))
    psi_target = np.zeros((n_samp, n_required_modes))

    itr = iter(dataset_valid)

    for i in range(0, n_samp, batch_size):

        print(i)

        (X_hr_target[i:(i+batch_size), :, :, :], Y_hr_target[i:(i+batch_size), :, :, :, :], psi_target[i:(i+batch_size), :]) = next(itr)
        psi_predic[i:(i+batch_size), :] = predictor.predict(X_hr_target[i:(i+batch_size), :, :, :])
        # print(psi_predic[i:(i+batch_size), :])
        # print(psi_predic[i:(i+batch_size), :])
        # kkk
    with np.load(f"/storage2/alejandro/urban/urban/pod_{nz}x{ny}x{nx}.npz") as data:

        sigma = data['sigma']
        phi = data['phi']
        psi = data['psi']

    with np.load('/storage2/alejandro/urban/urban/statistics_flow.npz') as data:

        std_out = data['std_flow']

    with np.load(f'/storage2/alejandro/urban/urban/grid.npz') as data:

        delta_v = data['delta_v']

    # C = np.matmul(psi_predic, np.matmul(sigma[:n_required_modes, :n_required_modes], phi[:n_required_modes, :]))
    C = np.matmul(psi_predic, np.matmul(sigma[:n_required_modes, :n_required_modes], phi[:n_required_modes, :]))
    
    Y_hr_predic = np.reshape(C, (n_samp,3,nz,ny,nx)) / delta_v  / std_out
    
    Y_hr_predic = np.nan_to_num(Y_hr_predic)
    Y_hr_target = np.nan_to_num(Y_hr_target)

    error_u = np.mean((Y_hr_predic[:,0,:,:,:] - Y_hr_target[:,0,:,:,:]) ** 2)
    error_v = np.mean((Y_hr_predic[:,1,:,:,:] - Y_hr_target[:,1,:,:,:]) ** 2)
    error_w = np.mean((Y_hr_predic[:,2,:,:,:] - Y_hr_target[:,2,:,:,:]) ** 2)
    print(error_u)
    print(error_v)
    print(error_w)

    import matplotlib.pyplot as plt

    rows = 1
    cols = 2
    ration = 2
    fig_width_pt = 400
    inches_per_pt = 1.0 / 72.27               
    golden_mean = (5 ** 0.5 - 1.0) / 2.0         
    fig_width = fig_width_pt * inches_per_pt 
    fig_height = fig_width / cols * rows * ration

    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
    axs[0,0].imshow(Y_hr_target[0,0,17,:,:], cmap='RdBu_r', vmin=-3, vmax=3, origin='lower', extent=(0.25,1.25,0,2))
    axs[0,1].imshow(Y_hr_predic[0,0,17,:,:], cmap='RdBu_r', vmin=-3, vmax=3, origin='lower', extent=(0.25,1.25,0,2))
    axs[0,0].set_xlabel('x/h')
    axs[0,1].set_xlabel('x/h')
    axs[0,0].set_ylabel('y/h')
    axs[0,0].set_title('Target')
    axs[0,1].set_title('Prediction')
    fig.savefig(f'test.pdf', dpi=600)

    C = np.matmul(psi_target, np.matmul(sigma[:n_required_modes, :n_required_modes], phi[:n_required_modes, :])) 
    Y_hr_target = np.reshape(C, (n_samp,3,nz,ny,nx)) / delta_v / std_out
    # scl = np.std(Y_hr_target, axis=0)
    # print(scl.shape)
    # print(scl[0,10,:,10])
    # Y_hr_target = Y_hr_target / scl
    # Y_hr_predic = Y_hr_predic / scl
    # Y_hr_predic = np.nan_to_num(Y_hr_predic)
    Y_hr_target = np.nan_to_num(Y_hr_target)

    print(Y_hr_target[0,0,10,:,10])
    print(Y_hr_predic[0,0,10,:,10])

    error_u = np.mean((Y_hr_predic[:,0,:,:,:] - Y_hr_target[:,0,:,:,:]) ** 2) 
    error_v = np.mean((Y_hr_predic[:,1,:,:,:] - Y_hr_target[:,1,:,:,:]) ** 2)
    error_w = np.mean((Y_hr_predic[:,2,:,:,:] - Y_hr_target[:,2,:,:,:]) ** 2)
    print(error_u)
    print(error_v)
    print(error_w)

    # C = np.matmul(psi, np.matmul(sigma, phi))
    # Y_hr_true = np.reshape(C, (2596,3,nz,ny,nx)) / delta_v / std_out
    # Y_hr_true = Y_hr_true[:500,:,:,:,:]
    # Y_hr_true = np.nan_to_num(Y_hr_true)

    # error_u = np.mean((Y_hr_true[20,0,:,:,:] - Y_hr_target[20,0,:,:,:]) ** 2)
    # error_v = np.mean((Y_hr_true[20,1,:,:,:] - Y_hr_target[20,1,:,:,:]) ** 2)
    # error_w = np.mean((Y_hr_true[20,2,:,:,:] - Y_hr_target[20,2,:,:,:]) ** 2)
    # print(error_u)
    # print(error_v)
    # print(error_w)
    



    return


def load_tfrecords(tfr_path, validation_split, batch_size, n_prefetch, shuffle_buffer):

    tfr_files = sorted([os.path.join(tfr_path, f) for f in os.listdir(tfr_path) if os.path.isfile(os.path.join(tfr_path, f))])

    regex = re.compile(f'.tfrecords')

    tfr_files = sorted([string for string in tfr_files if re.search(regex, string)])

    # Separating files for training and validation
    
    n_samples_per_tfr = np.array([int(s.split('.')[-2][-3:]) for s in tfr_files])

    n_samples_per_tfr = n_samples_per_tfr[np.argsort(-n_samples_per_tfr)]

    cumulative_samples_per_tfr = np.cumsum(np.array(n_samples_per_tfr))

    tot_samples_per_ds = sum(n_samples_per_tfr)

    n_tfr_loaded_per_ds = int(tfr_files[0].split('_')[-2][-3:])

    tfr_files = [string for string in tfr_files if int(string.split('_')[-2][:3]) <= n_tfr_loaded_per_ds]

    n_samp_train = int(sum(n_samples_per_tfr) * (1 - validation_split))
    n_samp_valid = sum(n_samples_per_tfr) - n_samp_train

    (n_files_train, samples_train_left) = np.divmod(n_samp_train, n_samples_per_tfr[0])

    if samples_train_left > 0:

        n_files_train += 1

    tfr_files_train = [string for string in tfr_files if int(string.split('_')[-2][:3]) <= n_files_train]

    n_tfr_left = np.sum(np.where(cumulative_samples_per_tfr < samples_train_left, 1, 0)) + 1


    if sum([int(s.split('.')[-2][-3:]) for s in tfr_files_train]) != n_samp_train:
        shared_tfr = tfr_files_train[-1]
        tfr_files_valid = [shared_tfr]
    else:
        shared_tfr = ''
        tfr_files_valid = list()

    tfr_files_valid.extend([string for string in tfr_files if string not in tfr_files_train])

    tfr_files_valid = sorted(tfr_files_valid)

    # Data preprocessing with tf.data.Dataset

    shared_tfr_out = tf.constant(shared_tfr)

    n_tfr_per_ds = tf.constant(n_tfr_loaded_per_ds)
    
    n_samples_loaded_per_tfr = list()

    if n_tfr_loaded_per_ds>1:

        n_samples_loaded_per_tfr.extend(n_samples_per_tfr[:n_tfr_loaded_per_ds-1])
        n_samples_loaded_per_tfr.append(tot_samples_per_ds - cumulative_samples_per_tfr[n_tfr_loaded_per_ds-2])

    else:

        n_samples_loaded_per_tfr.append(tot_samples_per_ds)

    n_samples_loaded_per_tfr = np.array(n_samples_loaded_per_tfr)
    
    tfr_files_train_ds = tf.data.Dataset.list_files(tfr_files_train, shuffle=False)

    tfr_files_val_ds = tf.data.Dataset.list_files(tfr_files_valid, shuffle=False)

    if n_tfr_left-1>0:

        samples_train_shared = samples_train_left - cumulative_samples_per_tfr[n_tfr_left-2]
        n_samples_tfr_shared = n_samples_loaded_per_tfr[n_tfr_left-1]

    else:

        samples_train_shared = samples_train_left
        n_samples_tfr_shared = n_samples_loaded_per_tfr[0]

    tfr_files_train_ds = tfr_files_train_ds.interleave(
        lambda x : tf.data.TFRecordDataset(x).take(samples_train_shared) if tf.math.equal(x, shared_tfr_out) else tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-2],sep='-')[0], tf.int32)-1)), 
        cycle_length=16, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    tfr_files_val_ds = tfr_files_val_ds.interleave(
        lambda x : tf.data.TFRecordDataset(x).skip(samples_train_shared).take(n_samples_tfr_shared - samples_train_shared) if tf.math.equal(x, shared_tfr_out) else tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-2],sep='-')[0], tf.int32)-1)),
        cycle_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    

    # Parsing datasets 
    dataset_train = tfr_files_train_ds.map(tf_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_train = dataset_train.batch(batch_size=batch_size)
    dataset_train = dataset_train.prefetch(n_prefetch)

    dataset_val = tfr_files_val_ds.map(tf_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_val = dataset_val.batch(batch_size=batch_size)
    dataset_val = dataset_val.prefetch(n_prefetch)   

    return dataset_train, dataset_val


@tf.function
def tf_parser(rec):
    '''
    This is a parser function. It defines the template for
    interpreting the examples you're feeding in. Basically, 
    this function defines what the labels and data look like
    for your labeled data. 
    '''
    features = {
        'i_samp': tf.io.FixedLenFeature([], tf.int64),
        'wall_hr_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'wall_hr_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'flow_hr_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'flow_hr_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'flow_hr_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'flow_psi_raw': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
    }  

    parsed_rec = tf.io.parse_single_example(rec, features)

    # Scaling data at wall

    avg_input_path = '/storage2/alejandro/urban/urban/statistics_wall.npz'

    print('The inputs are normalized to have a unit Gaussian distribution')

    with np.load(avg_input_path) as data:

        avg_in = tf.constant(data['avg_wall'].astype(np.float32))
        std_in = tf.constant(data['std_wall'].astype(np.float32))

    # Low-resolution processing --------------------------------------------------------
    
    inputs = (tf.reshape(parsed_rec['wall_hr_raw1'], (1, nz, nx)) - avg_in[0]) / std_in[0]
    
    for i_comp in range(1, channels):

        inputs = tf.concat((inputs, (tf.reshape(parsed_rec[f'wall_hr_raw{i_comp+1}'], (1, nz, nx)) - avg_in[i_comp]) / std_in[i_comp]), 0)

    # Scaling data at flow

    avg_input_path = '/storage2/alejandro/urban/urban/statistics_flow.npz'

    print('The inputs are normalized to have a unit Gaussian distribution')

    with np.load(avg_input_path) as data:

        avg_out = tf.constant(data['avg_flow'].astype(np.float32))
        std_out = tf.constant(data['std_flow'].astype(np.float32))
    
    # Low-resolution processing --------------------------------------------------------
    
    outputs = (tf.reshape(parsed_rec['flow_hr_raw1'], (1, nz, ny, nx)) - avg_out[0]) / std_out[0]
    
    for i_comp in range(1, 3):

        outputs = tf.concat((outputs, (tf.reshape(parsed_rec[f'flow_hr_raw{i_comp+1}'], (1, nz, ny, nx)) - avg_out[i_comp]) / std_out[i_comp]), 0)

    psi = parsed_rec['flow_psi_raw'][:n_required_modes]

    return inputs, outputs, psi


if __name__ == '__main__':

    nx = 17
    ny = 31
    nz = 34
    epochs = 300
    channels = 2
    batch_size = 1 
    n_prefetch = 4 
    shuffle_buffer = 100
    n_required_modes = 1
    validation_split = 0.2 
    models_path = "/storage2/alejandro/urban/urban/models/"
    save_path = "/storage2/alejandro/urban/urban/train/.tfrecords/"

    main()