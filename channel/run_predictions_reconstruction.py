# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 00:43:25 2021
@author: aguemes
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
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


def main():

    model_name = f"{models_path}Ret{Ret}_flow-reconstruction_yp{yp_flow:03d}_subsampling-{subsampling_lr:02d}_wall-channels-{channels}_predictor.tf"
    predictor = keras.models.load_model(model_name)

    dataset_test, nx, nz, n_samp = load_tfrecords(save_path, yp_flow, batch_size, n_prefetch, shuffle_buffer)

    X_lr_target = np.zeros((n_samp, channels, int(nz / subsampling_lr), int(nx / subsampling_lr)))
    Y_hr_target = np.zeros((n_samp, 3, nz, nx))
    Y_hr_predic = np.zeros((n_samp, n_required_modes, Nz, Nx))

    itr = iter(dataset_test)

    for i in range(0, n_samp, batch_size):

        print(i)

        (X_lr_target[i:(i+batch_size), :, :, :], Y_hr_target[i:(i+batch_size), :, :, :]) = next(itr)
        Y_hr_predic[i:(i+batch_size), :, :, :] = predictor.predict(X_lr_target[i:(i+batch_size), :, :, :])

    with np.load(f'/storage2/alejandro/urban/re{Ret}/PODbase_yp{yp_flow:03d}.npz') as data:

        phi_u   = data['phi'][:, :Px*Pz]
        phi_v   = data['phi'][:, Px*Pz:2*Px*Pz]
        phi_w   = data['phi'][:, 2*Px*Pz:]

    X_pred = np.zeros((n_samp * Nx * Nz, n_required_modes))

    nS=0

    for t in range(n_samp):
    
        for i in range(Nx):

            for k in range(Nz):
                        
                X_pred[nS, :] = Y_hr_predic[t, : ,k ,i]

                nS += 1

    U_pred = np.matmul(X_pred, phi_u[:n_required_modes, :])
    V_pred = np.matmul(X_pred, phi_v[:n_required_modes, :])
    W_pred = np.matmul(X_pred, phi_w[:n_required_modes, :])

    del X_pred

    Y_pred_u = np.zeros((n_samp, Nz*Pz, Nx*Px))
    Y_pred_v = np.zeros((n_samp, Nz*Pz, Nx*Px))
    Y_pred_w = np.zeros((n_samp, Nz*Pz, Nx*Px))

    nS = 0

    avg_outer_path = path_input + '/.avg/'
    std_out = []
    var_outer = ['u','v','w']

    for i in range(3):
        
        std_out.append(np.loadtxt(avg_outer_path + var_outer[i] + '_rms' + '.m')[ypos[str(yp_flow)], 1])
    
    for t in range(n_samp):

        for i in range(Nx):

            for k in range(Nz):

                Y_pred_u[t, Pz*k:Pz*(k+1), Px*i:Px*(i+1)] = np.reshape(U_pred[nS, :], (Pz, Px)) / std_out[0]
                Y_pred_v[t, Pz*k:Pz*(k+1), Px*i:Px*(i+1)] = np.reshape(V_pred[nS, :], (Pz, Px)) / std_out[1]
                Y_pred_w[t, Pz*k:Pz*(k+1), Px*i:Px*(i+1)] = np.reshape(W_pred[nS, :], (Pz, Px)) / std_out[2]
                
                nS += 1

    del U_pred, V_pred, W_pred

    Y_pred_u = np.expand_dims(Y_pred_u, axis=1)
    Y_pred_v = np.expand_dims(Y_pred_v, axis=1)
    Y_pred_w = np.expand_dims(Y_pred_w, axis=1)

    Y_hr_predic = np.concatenate(
        (Y_pred_u, Y_pred_v, Y_pred_w),
        axis=1
    )

    error_u = np.mean((Y_hr_target[:, 0, :, :] - Y_hr_predic[:, 0, :, :]) ** 2)
    print(error_u)
    error_v = np.mean((Y_hr_target[:, 1, :, :] - Y_hr_predic[:, 1, :, :]) ** 2)
    print(error_v)
    error_w = np.mean((Y_hr_target[:, 2, :, :] - Y_hr_predic[:, 2, :, :]) ** 2)
    print(error_w)

    filename = f"{prediction_path}/Ret{Ret}_flow-reconstruction_predictions_yp{yp_flow:03d}_subsampling-{subsampling_lr:02d}_wall-channels-{channels}.npz" 

    np.savez(
        filename, 
        X_lr_target=X_lr_target,
        Y_hr_target=Y_hr_target,
        Y_hr_predic=Y_hr_predic
    )

    return


def load_tfrecords(tfr_path, yp_flow, batch_size, n_prefetch, shuffle_buffer):

    tfr_files = sorted([os.path.join(tfr_path, f) for f in os.listdir(tfr_path) if os.path.isfile(os.path.join(tfr_path, f))])

    regex = re.compile(f'.tfrecords')

    tfr_files = sorted([string for string in tfr_files if re.search(regex, string)])

    regex = re.compile(f"yp{yp_flow}")

    tfr_files = sorted([string for string in tfr_files if re.search(regex, string)])
 
    regex = re.compile(f"subsampling{subsampling_lr}")

    tfr_files = sorted([string for string in tfr_files if re.search(regex, string)])

    # Separating files for training and validation
    
    n_samples_per_tfr = np.array([int(s.split('.')[-2][-3:]) for s in tfr_files])

    n_samples_per_tfr = n_samples_per_tfr[np.argsort(-n_samples_per_tfr)]

    cumulative_samples_per_tfr = np.cumsum(np.array(n_samples_per_tfr))

    tot_samples_per_ds = sum(n_samples_per_tfr)

    n_tfr_loaded_per_ds = int(tfr_files[0].split('_')[-2][-3:])

    tfr_files = [string for string in tfr_files if int(string.split('_')[-2][:3]) <= n_tfr_loaded_per_ds]

    n_samp_test = int(sum(n_samples_per_tfr))

    (n_files_test, samples_test_left) = np.divmod(n_samp_test, n_samples_per_tfr[0])

    if samples_test_left > 0:

        n_files_test += 1

    tfr_files_test = [string for string in tfr_files if int(string.split('_')[-2][:3]) <= n_files_test]


    (nx, nz, ny) = [int(val) for val in tfr_files[0].split('_')[-7].split('x')]

    # Data preprocessing with tf.data.Dataset

    n_tfr_per_ds = tf.constant(n_tfr_loaded_per_ds)
    
    n_samples_loaded_per_tfr = list()

    if n_tfr_loaded_per_ds>1:

        n_samples_loaded_per_tfr.extend(n_samples_per_tfr[:n_tfr_loaded_per_ds-1])
        n_samples_loaded_per_tfr.append(tot_samples_per_ds - cumulative_samples_per_tfr[n_tfr_loaded_per_ds-2])

    else:

        n_samples_loaded_per_tfr.append(tot_samples_per_ds)

    n_samples_loaded_per_tfr = np.array(n_samples_loaded_per_tfr)
    
    tfr_files_test_ds = tf.data.Dataset.list_files(tfr_files_test, seed=666)

    tfr_files_test_ds = tfr_files_test_ds.interleave(
        lambda x : tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-2],sep='-')[0], tf.int32)-1)), 
        cycle_length=16, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    

    # Parsing datasets 
    dataset_test = tfr_files_test_ds.map(tf_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset_test = dataset_test.shuffle(shuffle_buffer)
    dataset_test = dataset_test.batch(batch_size=batch_size)
    dataset_test = dataset_test.prefetch(n_prefetch)

    return dataset_test, nx, nz, n_samp_test


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
        'n_x': tf.io.FixedLenFeature([], tf.int64),
        'n_y': tf.io.FixedLenFeature([], tf.int64),
        'n_z': tf.io.FixedLenFeature([], tf.int64),
        'n_modes': tf.io.FixedLenFeature([], tf.int64),
        'subsampling': tf.io.FixedLenFeature([], tf.int64),
        'wall_lr_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'wall_lr_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'wall_lr_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'wall_hr_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'wall_hr_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'wall_hr_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'flow_hr_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'flow_hr_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'flow_hr_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        }

    parsed_rec = tf.io.parse_single_example(rec, features)

    nx_hr = tf.cast(parsed_rec['n_x'], tf.int32)
    nz_hr = tf.cast(parsed_rec['n_z'], tf.int32)
    n_modes = tf.cast(parsed_rec['n_modes'], tf.int32)
    subsampling = tf.cast(parsed_rec['subsampling'], tf.int32)
    nx_lr = nx_hr / subsampling
    nz_lr = nz_hr / subsampling 

    # Scaling data at wall

    avg_input_path = path_input + '/.avg_inputs/'

    print('The inputs are normalized to have a unit Gaussian distribution')

    with np.load(avg_input_path+f'stats_ds12x4200_dt135.npz') as data:

        avgs_in = tf.constant(data['mean_inputs'].astype(np.float32))
        std_in = tf.constant(data['std_inputs'].astype(np.float32))

    # Low-resolution processing --------------------------------------------------------

    if subsampling_lr == 1:

        inputs = tf.reshape((parsed_rec['wall_hr_raw1']-avgs_in[0])/std_in[0],(1,nz_hr, nx_hr))

        for i_comp in range(1, channels):

            inputs = tf.concat((inputs, tf.reshape((parsed_rec[f'wall_hr_raw{i_comp+1}']-avgs_in[i_comp])/std_in[i_comp],(1,nz_hr, nx_hr))),0)

    else:

        inputs = tf.reshape((parsed_rec['wall_lr_raw1']-avgs_in[0])/std_in[0],(1,nz_lr, nx_lr))

        for i_comp in range(1, channels):

            inputs = tf.concat((inputs, tf.reshape((parsed_rec[f'wall_lr_raw{i_comp+1}']-avgs_in[i_comp])/std_in[i_comp],(1,nz_lr, nx_lr))),0)

    # High-resolution processing --------------------------------------------------------

    # Output processing
    
    # Scaling data at flow

    avg_outer_path = path_input + '/.avg/'

    print('The outputs are normalized to have a unit Gaussian distribution')

    avgs_out = []
    std_out = []
    var_outer = ['u','v','w']

    for i in range(3):

        avgs_out.append(np.loadtxt(avg_outer_path + 'mean_' + var_outer[i] + '.m')[ypos[str(yp_flow)], 1])
        std_out.append(np.loadtxt(avg_outer_path + var_outer[i] + '_rms' + '.m')[ypos[str(yp_flow)], 1])

    # High-resolution processing --------------------------------------------------------

    outputs = tf.reshape((parsed_rec['flow_hr_raw1']-avgs_out[0])/std_out[0],(1,nz_hr, nx_hr))

    for i_comp in range(1, 3):

        outputs = tf.concat((outputs, tf.reshape((parsed_rec[f'flow_hr_raw{i_comp+1}']-avgs_out[i_comp])/std_out[i_comp],(1,nz_hr, nx_hr))),0)

    return inputs, outputs


if __name__ == '__main__':

    Ret = 180
    epochs = 100
    yp_wall = 0
    yp_flow = 30
    channels = 1
    batch_size = 8
    n_prefetch = 4
    n_res_block = 16
    subsampling_hr = 1
    subsampling_lr = 4
    shuffle_buffer = 4200

    if Ret == 180:

        Nx = 12
        Nz = 12
        Px = 16
        Pz = 16
        interv = 3
        n_required_modes = 64
        max_samples_per_tfr = 120
        save_path = "/storage2/alejandro/urban/re180/test/.tfrecords/"
        models_path = "/storage2/alejandro/urban/re180/models/"
        prediction_path = "/storage2/alejandro/urban/re180/predictions/"
        path_input = '/storage3/luca/PhD/015-Madrid/simson/015-Ret180_192x192x65/Test/'
        ref_file = '/storage3/luca/PhD/015-Madrid/simson/015-Ret180_192x192x65/Test/uxz_yp0_0.pl'
        n_samples = (4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200)
        ypos = {'0':0, '10':10, '15':12, '20':14, '30':17, '50':23, '80':30, '100':34, '120':39, '150':47}

    main()