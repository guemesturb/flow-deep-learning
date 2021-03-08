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

    dataset_train, dataset_valid, nx, nz = load_tfrecords(save_path, yp_flow, validation_split, batch_size, n_prefetch, shuffle_buffer)

    """
        Training loop
    """

    model_name = f"Ret{Ret}_flow-reconstruction_yp{yp_flow:03d}_subsampling-{subsampling_lr:02d}_wall-channels-{channels}"

    start_time = time.time()

    predictor_optimizer = tf.keras.optimizers.Adam(1e-3, epsilon=0.1)
    
    predictor = model_predictor()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        epsilon=0.1
    )

    predictor.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])

    with open(f'logs/log_{model_name}.log','w') as fd:

        fd.write(f"epoch,loss,val_loss\n")

    train_loss = tf.metrics.Mean()
    valid_loss = tf.metrics.Mean()

    for epoch in range(1, epochs + 1):

        train_loss.reset_states()
        valid_loss.reset_states()

        for (X_target, Y_target) in dataset_train:

            pred_loss, mae = predictor.train_on_batch(X_target, Y_target)
            train_loss.update_state(pred_loss)
            
        for (X_target, Y_target) in dataset_valid:

            valid_pred_loss, valid_mae = predictor.test_on_batch(X_target, Y_target)
            valid_loss.update_state(valid_pred_loss)
        
        end_time = time.time()

        # if epoch > 10:
        
        #     predictor.optimizer.lr = 0.001 * tf.math.exp(0.1 * (10 - epoch))

        with open(f'logs/log_{model_name}.log','a') as fd:

            fd.write(f"{epoch},{train_loss.result().numpy()},{valid_loss.result().numpy()}\n")

        print(f'Epoch {epoch:04d}/{epochs:04d}, loss: {train_loss.result().numpy()}, val_loss: {valid_loss.result().numpy()}, elapsed time from start: {end_time - start_time}')

    predictor.save(f'{models_path}{model_name}_predictor.tf')

    return


def model_predictor():

    if subsampling_lr == 1:

        inputs = keras.Input(shape=(channels, nz, nx), name='high-res-input')
        predictor = inputs

    else:

        model_name = f"Ret{Ret}_super-resolution_subsampling-{subsampling_lr:02d}_wall-channels-{channels}"
        model_name = f"{models_path}{model_name}_generator.tf"
        generator = keras.models.load_model(model_name)
        inputs = generator.input

        for l in generator.layers:

            generator.get_layer(l.name).trainable = False

        predictor = generator.layers[-1].output

    predictor = layers.Conv2D(filters=128, kernel_size=(5, 5), activation='relu', data_format='channels_first', padding='same', name='predic_01')(predictor)
    predictor = layers.BatchNormalization(axis=1, name='predic_02')(predictor)
    predictor = layers.MaxPooling2D(pool_size=[2, 2], data_format='channels_first', name='predic_03')(predictor)
    predictor = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', data_format='channels_first', kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01), padding='same', name='predic_04')(predictor)
    predictor = layers.BatchNormalization(axis=1, name='predic_05')(predictor)
    predictor = layers.MaxPooling2D(pool_size=[2, 2], data_format='channels_first', name='predic_06')(predictor)
    predictor = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', data_format='channels_first', kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01), padding='same', name='predic_07')(predictor)
    predictor = layers.BatchNormalization(axis=1, name='predic_08')(predictor)
    predictor = layers.MaxPooling2D(pool_size=[2, 2], data_format='channels_first', name='predic_09')(predictor)
    predictor = layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', data_format='channels_first', kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01), padding='same', name='predic_10')(predictor)
    predictor = layers.BatchNormalization(axis=1, name='predic_11')(predictor)
    predictor = layers.MaxPooling2D(pool_size=[2, 2], data_format='channels_first', name='predic_12')(predictor)
    predictor = layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', data_format='channels_first', kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01), padding='same', name='predic_13')(predictor)
    predictor = layers.BatchNormalization(axis=1, name='predic_14')(predictor)
    predictor = layers.Conv2D(filters=n_required_modes, kernel_size=(3, 3), activation='linear', data_format='channels_first', padding='same', name='predic_15')(predictor)

    predictor = keras.Model(inputs, predictor, name='CNN-POD')

    print(predictor.summary())

    return predictor


def load_tfrecords(tfr_path, yp_flow, validation_split, batch_size, n_prefetch, shuffle_buffer):

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

    (nx, nz, ny) = [int(val) for val in tfr_files[0].split('_')[-7].split('x')]

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
    
    tfr_files_train_ds = tf.data.Dataset.list_files(tfr_files_train, seed=666)

    tfr_files_val_ds = tf.data.Dataset.list_files(tfr_files_valid, seed=686)

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
    dataset_train = dataset_train.shuffle(shuffle_buffer)
    dataset_train = dataset_train.batch(batch_size=batch_size)
    dataset_train = dataset_train.prefetch(n_prefetch)

    dataset_val = tfr_files_val_ds.map(tf_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_val = dataset_val.shuffle(shuffle_buffer)
    dataset_val = dataset_val.batch(batch_size=batch_size)
    dataset_val = dataset_val.prefetch(n_prefetch)   

    return dataset_train, dataset_val, nx, nz


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
        'flow_pd_raw':  tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
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
    
    outputs = tf.reshape(parsed_rec['flow_pd_raw'],(n_modes, Nz, Nx))[:n_required_modes,:,:]

    return inputs, outputs


if __name__ == '__main__':

    Ret = 180
    epochs = 30
    yp_wall = 0
    yp_flow = 30
    channels = 1
    batch_size = 8
    n_prefetch = 4
    n_res_block = 8
    subsampling_hr = 1
    subsampling_lr = 4
    shuffle_buffer = 4200
    validation_split = 0.2
    nx = 192
    nz = 192

    if Ret == 180:

        Nx = 12
        Nz = 12
        interv = 3
        n_required_modes = 64
        max_samples_per_tfr = 120
        save_path = "/storage2/alejandro/urban/re180/train/.tfrecords/"
        models_path = "/storage2/alejandro/urban/re180/models/"
        path_input = '/storage3/luca/PhD/015-Madrid/simson/015-Ret180_192x192x65/Train/'
        ref_file = '/storage3/luca/PhD/015-Madrid/simson/015-Ret180_192x192x65/Train/uxz_yp0_0.pl'
        n_samples = (4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200)

    main()