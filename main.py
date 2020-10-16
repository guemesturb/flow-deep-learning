# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:34:59 2020
@author: aguemes
"""


from libs.models.cnn.cnn import CNN_Model
from libs.tfrecordshandler import tfrecord_handler



def main():

    """
        Generate TFRecords
    """

    re180data = tfrecord_handler(
        Ret=Ret, 
        path_input=path_input, 
        tfr_path=tfr_path, 
        max_samples_per_tfr=max_samples_per_tfr, 
        n_samples=n_samples, 
        subsampling=subsampling, 
        mode=mode,
        validation_split=validation_split
    )

    if generate:
    
        re180data.generate_tfrecords(
            interv=interv,
            ref_file=ref_file
        )

    """
        Load TFRecord pipelines
    """

    dataset_train, dataset_val = re180data.read_TFRecord(
        channels=channels,
        shuffle_buffer=shuffle_buffer,
        epochs=2,
        batch_size=batch_size,
        n_prefetch=n_prefetch
    )

    """
        Model training 
    """
    
    model = CNN_Model()

    model.build_model(subsampling, 192, 192, channels)

    model.train_model(dataset_train, dataset_val, epochs, n_samples, batch_size)

    return


if __name__ == '__main__':

    """
        Dataset parameters
    """

    Ret = 180
    mode = 'Train'
    subsampling = 2
    generate = False
    channels=1

    if mode == 'Train':

        # Training

        ref_file = '/storage3/luca/PhD/015-Madrid/simson/015-Ret180_192x192x65/Train/uxz_yp0_0.pl'
        path_input = '/storage3/luca/PhD/015-Madrid/simson/015-Ret180_192x192x65/Train/'

        n_samples = (
            4200, 4200, 4200, 4200, 4200, 4200,
            4200, 4200, 4200, 4200, 4200, 4200
        )

        interv = 3

    elif mode == 'Test':

        # Test

        ref_file = '/storage3/luca/PhD/015-Madrid/simson/015-Ret180_192x192x65/Test/uxz_yp0_0.pl'
        path_input = '/storage3/luca/PhD/015-Madrid/simson/015-Ret180_192x192x65/Test/'
        
        n_samples = (
            1680, 1680
        )

        interv = 1

    """
        Tensorflow parameters
    """

    validation_split = 0.2
    epochs = 1
    shuffle_buffer = 4500
    n_prefetch = 4
    batch_size = 8
    max_samples_per_tfr = 420
    tfr_path = '/storage2/alejandro/srgan/'

    """
        Main execution logic
    """

    main()