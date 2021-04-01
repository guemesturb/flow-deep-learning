# -*- coding: utf-8 -*-
"""
Created on Wed Mar 03 13:44:23 2021
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


def main():

    dataset_train, dataset_valid = load_tfrecords(save_path, validation_split, batch_size, n_prefetch, shuffle_buffer)

    n_samp = int(2500 * validation_split)
    
    Y_hr_target = np.zeros((n_samp, 3, nz, ny, nx))
    psi_target = np.zeros((n_samp, 2596))

    itr = iter(dataset_train)

    for i in range(0, n_samp, batch_size):

        print(i)

        (Y_hr_target[i:(i+batch_size), :, :, :, :], psi_target[i:(i+batch_size), :]) = next(itr)

        break

    with np.load(f"/storage2/alejandro/urban/urban/pod_{nz}x{ny}x{nx}.npz") as data:

        sigma = data['sigma']
        phi = data['phi']
        psi = data['psi']

    with np.load(f'/storage2/alejandro/urban/urban/grid.npz') as data:
        delta_v = data['delta_v']
    avg_input_path = '/storage2/alejandro/urban/urban/statistics_flow.npz'
    with np.load(avg_input_path) as data:
        std_out = data['std_flow']

    C = np.matmul(psi, np.matmul(sigma, phi))
    U = C[:,:nx*ny*nz]
    U = np.reshape(U, (2596,nz,ny,nx))
    print((U/delta_v)[0,:5,20,5])
    print(Y_hr_target[0,0,:5,20,5])
    print(U.shape)

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
    # dataset_train = dataset_train.shuffle(shuffle_buffer)
    dataset_train = dataset_train.batch(batch_size=batch_size)
    dataset_train = dataset_train.prefetch(n_prefetch)

    dataset_val = tfr_files_val_ds.map(tf_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset_val = dataset_val.shuffle(shuffle_buffer)
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
    print(avg_out.shape)
    # Low-resolution processing --------------------------------------------------------
    
    outputs = (tf.reshape(parsed_rec['flow_hr_raw1'], (1, nz, ny, nx)) - avg_out[0])
    
    for i_comp in range(1, 3):

        outputs = tf.concat((outputs, (tf.reshape(parsed_rec[f'flow_hr_raw{i_comp+1}'], (1, nz, ny, nx)) - avg_out[i_comp])), 0)

    outputs2 = parsed_rec['flow_psi_raw']

    return outputs, outputs2


if __name__ == '__main__':

    nx = 17
    ny = 31
    nz = 34
    epochs = 300
    channels = 2
    batch_size = 1 
    n_prefetch = 4 
    shuffle_buffer = 100
    n_required_modes = 64
    validation_split = 0.2 
    models_path = "/storage2/alejandro/urban/urban/models/"
    save_path = "/storage2/alejandro/urban/urban/train/.tfrecords/"
    
    main()