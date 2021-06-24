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

    model_name = f"Ret{Ret}_super-resolution_subsampling-{subsampling_lr:02d}_wall-channels-{channels}"

    start_time = time.time()

    optimizer = tf.keras.optimizers.Adam(1e-4)
    
    generator =  model_generator(nx, nz, channels, subsampling_lr, n_res_block, optimizer)
    discriminator = model_discriminator(nx, nz, channels, optimizer)
    gan = get_gan_network(discriminator, channels, nx, nz, subsampling_lr, generator, optimizer)

    train_gen_loss = tf.metrics.Mean()
    train_disc_loss = tf.metrics.Mean()
    valid_gen_loss = tf.metrics.Mean()
    valid_disc_loss = tf.metrics.Mean()
 
    with open(f'logs/log_{model_name}.log','w') as fd:

        fd.write(f"epoch,gen_loss,disc_loss,val_gen_loss,val_disc_loss\n")

    for epoch in range(1, epochs + 1):

        train_gen_loss.reset_states()
        train_disc_loss.reset_states()
        valid_gen_loss.reset_states()
        valid_disc_loss.reset_states()

        for (lr_target, hr_target) in dataset_train:

            hr_predic = generator.predict(lr_target)

            fake_output = discriminator.predict(hr_predic)

            real_data_Y = np.ones(fake_output.shape) - np.random.random_sample(fake_output.shape)*0.2 
            fake_data_Y = np.random.random_sample(fake_output.shape)*0.2

            discriminator.trainable = True

            d_loss_real = discriminator.train_on_batch(hr_target, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(hr_predic, fake_data_Y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            discriminator.trainable = False

            gan_loss = gan.train_on_batch(lr_target, [hr_target, real_data_Y])
            
            train_gen_loss.update_state(gan_loss[0])
            train_disc_loss.update_state(discriminator_loss)

        for (lr_target, hr_target) in dataset_valid:

            hr_predic = generator.predict(lr_target)

            fake_output = discriminator.predict(hr_predic)

            real_data_Y = np.ones(fake_output.shape) - np.random.random_sample(fake_output.shape)*0.2 
            fake_data_Y = np.random.random_sample(fake_output.shape)*0.2

            d_loss_real = discriminator.test_on_batch(hr_target, real_data_Y)
            d_loss_fake = discriminator.test_on_batch(hr_predic, fake_data_Y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            gan_loss = gan.test_on_batch(lr_target, [hr_target, real_data_Y])
            
            valid_gen_loss.update_state(gan_loss[0])
            valid_disc_loss.update_state(discriminator_loss)
        
        end_time = time.time()

        # if epoch % 10 == 0:

        #     generator.save(f'{models_path}{model_name}_generator_epoch-{epoch:03d}-of-{epochs}.tf')
        #     discriminator.save(f'{models_path}{model_name}_discriminator_epoch-{epoch:03}-of-{epochs}.tf')

        with open(f'logs/log_{model_name}.log','a') as fd:

            fd.write(f"{epoch},{train_gen_loss.result().numpy()},{train_disc_loss.result().numpy()},{valid_gen_loss.result().numpy()},{valid_disc_loss.result().numpy()}\n")

        print(f'Epoch {epoch:04d}/{epochs:04d}, gen_loss: {train_gen_loss.result().numpy()}, disc_loss: {train_disc_loss.result().numpy()}, val_gen_loss: {valid_gen_loss.result().numpy()}, val_disc_loss: {valid_disc_loss.result().numpy()}, elapsed time from start: {end_time - start_time}')

    generator.save(f'{models_path}{model_name}_generator.tf')
    discriminator.save(f'{models_path}{model_name}_discriminator.tf')

    return


def model_generator(nx, nz, channels, subsampling, n_res_block, optimizer):

    inputs = keras.Input(shape=(channels, int(nz/ subsampling), int(nx / subsampling)), name='low-res-input')

    conv_1 = layers.Conv2D(filters=64, kernel_size=9, strides=1, activation='linear', data_format='channels_first', padding='same')(inputs)

    prelu_1 = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[2,3])(conv_1)

    res_block = prelu_1

    for index in range(n_res_block):

        res_block = res_block_gen(res_block, 3, 64, 1)


    conv_2 = layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same", data_format='channels_first')(res_block)
    batch_1 = layers.BatchNormalization(axis=1, momentum = 0.5)(conv_2)
    add_1 = layers.Add()([prelu_1, batch_1])

    up_sampling = add_1

    for index in range(int(np.log2(subsampling))):

        up_sampling = up_sampling_block(up_sampling, 3, 256, 1)

    conv_3 = layers.Conv2D(filters = channels, kernel_size = 9, strides = 1, padding = "same", data_format='channels_first')(up_sampling)
    outputs = conv_3


    model = keras.Model(inputs, outputs, name='Generator')

    model.compile(loss="mean_squared_error", optimizer=optimizer)

    print(model.summary())

    return model


def res_block_gen(model, kernal_size, filters, strides):

    gen = model
    
    model = layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", data_format='channels_first')(model)
    model = layers.BatchNormalization(axis=1, momentum = 0.5)(model)
    # Using Parametric ReLU
    model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[2,3])(model)
    model = layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", data_format='channels_first')(model)
    model = layers.BatchNormalization(axis=1, momentum = 0.5)(model)
        
    model = layers.Add()([gen, model])
    
    return model


def up_sampling_block(model, kernal_size, filters, strides):

    # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
    # Even we can have our own function for deconvolution (i.e one made in Utils.py)
    #model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", data_format='channels_first')(model)
    # model = layers.UpSampling2D(size = 2, data_format='channels_first')(model)
    model = SubpixelConv2D(model.shape, scale=2)(model)
    model = layers.LeakyReLU(alpha = 0.2)(model)
    
    return model


def SubpixelConv2D(input_shape, scale=4):
    
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                int(input_shape[1] / (scale ** 2)),
                input_shape[2] * scale,
                input_shape[3] * scale]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.nn.depth_to_space(x, scale, data_format='NCHW')


    return layers.Lambda(subpixel, output_shape=subpixel_shape)


def model_discriminator(nx, nz, channels, optimizer):

    inputs = keras.Input(shape=(channels, nz, nx), name='high-res-input')
        
    model = layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same", data_format='channels_first')(inputs)
    model = layers.LeakyReLU(alpha = 0.2)(model)
    
    model = discriminator_block(model, 64, 3, 2)
    model = discriminator_block(model, 128, 3, 1)
    model = discriminator_block(model, 128, 3, 2)
    model = discriminator_block(model, 256, 3, 1)
    model = discriminator_block(model, 256, 3, 2)
    model = discriminator_block(model, 512, 3, 1)
    model = discriminator_block(model, 512, 3, 2)
    
    model = layers.Flatten()(model)
    model = layers.Dense(1024)(model)
    model = layers.LeakyReLU(alpha = 0.2)(model)

    model = layers.Dense(1)(model)
    model = layers.Activation('sigmoid')(model) 
    
    model = keras.Model(inputs=inputs, outputs = model, name='Discriminator')

    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    print(model.summary())
    
    return model


def discriminator_block(model, filters, kernel_size, strides):
        
    model = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", data_format='channels_first')(model)
    model = layers.BatchNormalization(axis=1, momentum = 0.5)(model)
    model = layers.LeakyReLU(alpha = 0.2)(model)
    
    return model


def get_gan_network(discriminator, channels, nx, nz, subsampling_lr, generator, optimizer):
    
    discriminator.trainable = False
    gan_input = keras.Input(shape=(channels, int(nz/ subsampling_lr), int(nx / subsampling_lr)), name='low-res-input')
    x = generator(gan_input)
    gan_output = discriminator(x)
    
    gan = keras.Model(inputs=gan_input, outputs=[x,gan_output])
    
    gan.compile(loss=["mean_squared_error", "binary_crossentropy"], loss_weights=[1., 1e-3], optimizer=optimizer)
    
    print(gan.summary())
    
    return gan


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
        'subsampling': tf.io.FixedLenFeature([], tf.int64),
        'wall_lr_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'wall_lr_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'wall_lr_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'wall_hr_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'wall_hr_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'wall_hr_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        }

    parsed_rec = tf.io.parse_single_example(rec, features)

    nx_hr = tf.cast(parsed_rec['n_x'], tf.int32)
    nz_hr = tf.cast(parsed_rec['n_z'], tf.int32)
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

    inputs = tf.reshape((parsed_rec['wall_lr_raw1']-avgs_in[0])/std_in[0],(1,nz_lr, nx_lr))

    for i_comp in range(1, channels):

        inputs = tf.concat((inputs, tf.reshape((parsed_rec[f'wall_lr_raw{i_comp+1}']-avgs_in[i_comp])/std_in[i_comp],(1,nz_lr, nx_lr))),0)

    # High-resolution processing --------------------------------------------------------

    outputs = tf.reshape((parsed_rec['wall_hr_raw1']-avgs_in[0])/std_in[0],(1,nz_hr, nx_hr))

    for i_comp in range(1, channels):

        outputs = tf.concat((outputs, tf.reshape((parsed_rec[f'wall_hr_raw{i_comp+1}']-avgs_in[i_comp])/std_in[i_comp],(1,nz_hr, nx_hr))),0)

    return inputs, outputs


if __name__ == '__main__':

    Ret = 180
    epochs = 20
    yp_wall = 0
    yp_flow = 15
    channels = 3
    batch_size = 8
    n_prefetch = 4
    n_res_block = 16
    subsampling_hr = 1
    subsampling_lr = 2
    shuffle_buffer = 4200
    validation_split = 0.2

    if Ret == 180:

        Nx = 12
        Nz = 12
        interv = 3
        max_samples_per_tfr = 120
        save_path = "/storage2/alejandro/urban/re180/train/.tfrecords/"
        models_path = "/storage2/alejandro/urban/re180/models/"
        path_input = '/storage3/luca/PhD/015-Madrid/simson/015-Ret180_192x192x65/Train/'
        ref_file = '/storage3/luca/PhD/015-Madrid/simson/015-Ret180_192x192x65/Train/uxz_yp0_0.pl'
        n_samples = (4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200)

    main()