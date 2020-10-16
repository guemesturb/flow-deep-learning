import numpy as np
import matplotlib.pyplot as plt


class CNN_Model():


    def __init__(self, gpus=''):


        import os
        os.environ["CUDA_VISIBLE_DEVICES"]=gpus
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

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

        self.tf = tf

        return


    def build_model(self, subsampling, nx, nz, channels):

        inputs = self.tf.keras.Input(shape=(channels, int(nx / subsampling), int(nz / subsampling)), name='low-res-input')

        outputs = self.tf.keras.layers.Conv2DTranspose(
            filters=1, 
            kernel_size=(3, 3), 
            strides=2,
            data_format='channels_first', 
            padding='same',
            name='high-res-output'
        )(inputs)

        self.model = self.tf.keras.Model(inputs, outputs, name='MMH')

        optimizer = self.tf.keras.optimizers.Adam(
            learning_rate=0.001,
            epsilon=0.1
        )

        self.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])

        print(self.model.summary())

        return

    
    def train_model(self, dataset_train, dataset_val, epochs, n_samples, batch_size):

        self.model.fit(
            dataset_train,
            epochs=epochs,
            steps_per_epoch=int(np.ceil(np.sum(n_samples)/batch_size)),
            validation_data=dataset_val,
            validation_steps=int(np.ceil(np.sum(n_samples)/batch_size)),
            verbose=1
        )

        self.model.save(
            "dummy_model.tf", save_format='tf'
        )

        return




