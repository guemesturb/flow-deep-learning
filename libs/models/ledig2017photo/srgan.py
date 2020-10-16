# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:42:16 2020
@author: aguemes
"""


import os
import re
import math
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

physical_devices = tf.config.list_physical_devices('GPU')
availale_GPUs = len(physical_devices) 

print('Using TensorFlow version: ', tf.__version__, ', GPU:', availale_GPUs)
print('Using Keras version: ', tf.keras.__version__)


class SRGAN():


    def __init__(self, yp, Ret, path_input, tfr_path, max_samples_per_tfr, n_samples, validation_split, outer_channels, inner_channels, subsampling):

        print('Initializing the SRGAN class')

        self.yp = yp
        self.tfr_path = tfr_path
        self.Ret = Ret
        self.max_samples_per_tfr = max_samples_per_tfr
        self.n_samples = n_samples
        self.validation_split = validation_split
        self.outer_channels = outer_channels
        self.inner_channels = inner_channels
        self.path_input = path_input
        self.subsampling = subsampling
        self.var_outer = {0: 'u', 1: 'v', 2: 'w'} 
        self.var_inner = {0: 'u', 1: 'v', 2: 'w'} 

        print(f'Friction Reynolds number of the data is {self.Ret}')

        return

    
    def generate_tfrecords_training(self, ref_file, interv):

        print(f'Generating TFRecords for')

        Re, xl, zl, nx_, ny_, nz_, h_size = self.header_reader(ref_file)

        dy_w = self.wall_dy(ny_)

        X_LR = self.preprocess_planes(self.yp, interv, nx_, nz_, h_size, dy_w, self.subsampling, target=False)
        X_HR = self.preprocess_planes(self.yp, interv, nx_, nz_, h_size, dy_w, subsampling=1, target=False)

        self.write_TFRecord(X_LR, X_HR, nx_, nz_, ny_, interv, self.tfr_path)  

        return


    def load_tfrecords_training(self, epochs, shuffle_buffer, n_prefetch, batch_size):

        tfr_files = [os.path.join(self.tfr_path,f) for f in os.listdir(self.tfr_path) if os.path.isfile(os.path.join(self.tfr_path,f))]

        regex = re.compile(f'yp-outer{self.yp_outer}_yp-inner{self.yp_inner}')
        
        tfr_files = [string for string in tfr_files if re.search(regex, string)]

        regex = re.compile(f'Ret{self.Ret}')

        tfr_files = [string for string in tfr_files if re.search(regex, string)]
    
        # Separating files for training and validation
        
        n_samples_per_tfr = np.array([int(s.split('_')[-2][14:]) for s in tfr_files if int(s.split('_')[-2][4:7])==0])
        
        n_samples_per_tfr = n_samples_per_tfr[np.argsort(-n_samples_per_tfr)]
        
        cumulative_samples_per_tfr = np.cumsum(np.array(n_samples_per_tfr))

        tot_samples_per_ds = sum(n_samples_per_tfr)
                        
        print(f'WARNING: The maximum number of samples per file is fixed to {self.max_samples_per_tfr}, the number of samples from each SIMSON dataset is {tot_samples_per_ds}, make sure that it corresponds to the actual number of files in the TFRecords')

        n_tfr_loaded_per_ds = np.sum(np.where(cumulative_samples_per_tfr<self.n_samples[0],1,0))+1

        tfr_files = [string for string in tfr_files if int(string.split('_')[-1][:3])<=n_tfr_loaded_per_ds]

        n_samp_train = int(sum(self.n_samples) * (1 - self.validation_split))
        n_samp_valid = sum(self.n_samples) - n_samp_train
    
        (n_files_train, samples_train_left) = np.divmod(n_samp_train,self.n_samples[0])

        tfr_files_train = [string for string in tfr_files if int(string.split('_')[-2][4:7])<=n_files_train]

        n_tfr_left = np.sum(np.where(cumulative_samples_per_tfr<samples_train_left,1,0))+1

        tfr_files_train = [string for string in tfr_files_train if ((int(string.split('_')[-2][4:7])<n_files_train) or (int(string.split('_')[-2][4:7])==n_files_train and int(string.split('_')[-1][:3])<=n_tfr_left))]

        tfr_files_train = sorted(tfr_files_train, key=lambda s: (int(s.split('_')[-2][4:7]), int(s.split('_')[-1][:3])))

        if sum([int(s.split('_')[-2][14:]) for s in tfr_files_train]) != n_samp_train:
            shared_tfr = tfr_files_train[-1]
            tfr_files_valid = [shared_tfr]
        else:
            shared_tfr = ''
            tfr_files_valid = list()

        tfr_files_valid.extend([string for string in tfr_files if string not in tfr_files_train])

        tfr_files_valid = sorted(tfr_files_valid, key=lambda s: (int(s.split('_')[-2][4:7]), int(s.split('_')[-1][:3])))

        (nx_, nz_, ny_) = tf.constant([int(val) for val in tfr_files[0].split('_')[-6].split('x')])
        
        # Old setting again

        nx = nx_
        nz = nz_ 

        input_shape = (self.outer_channels, nx.numpy(), nz.numpy())

        # Data preprocessing with tf.data.Dataset

        shared_tfr_out = tf.constant(shared_tfr)

        n_tfr_per_ds = tf.constant(n_tfr_loaded_per_ds)
        n_samples_loaded_per_tfr = list()

        if n_tfr_loaded_per_ds>1:
            n_samples_loaded_per_tfr.extend(n_samples_per_tfr[:n_tfr_loaded_per_ds-1])
            n_samples_loaded_per_tfr.append(self.n_samples[0] - cumulative_samples_per_tfr[n_tfr_loaded_per_ds-2])
        else:
            n_samples_loaded_per_tfr.append(self.n_samples[0])

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
            lambda x : tf.data.TFRecordDataset(x).take(samples_train_shared) if tf.math.equal(x,shared_tfr_out) else tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-1],sep='-')[0], tf.int32)-1)), 
            cycle_length=16, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        tfr_files_val_ds = tfr_files_val_ds.interleave(
            lambda x : tf.data.TFRecordDataset(x).skip(samples_train_shared).take(n_samples_tfr_shared - samples_train_shared) if tf.math.equal(x,shared_tfr_out) else tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-1],sep='-')[0], tf.int32)-1)),
            cycle_length=16,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        
        # Parsing datasets ------------------------------------------------------------
        self.dataset_train = tfr_files_train_ds.map(self.tf_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.dataset_train = self.dataset_train.shuffle(shuffle_buffer)
        self.dataset_train = self.dataset_train.repeat(epochs)
        self.dataset_train = self.dataset_train.batch(batch_size=batch_size)
        self.dataset_train = self.dataset_train.prefetch(n_prefetch)

        self.dataset_val = tfr_files_val_ds.map(self.tf_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.dataset_val = self.dataset_val.shuffle(shuffle_buffer)
        self.dataset_val = self.dataset_val.repeat(epochs)
        self.dataset_val = self.dataset_val.batch(batch_size=batch_size)
        self.dataset_val = self.dataset_val.prefetch(n_prefetch)   

        return


    def header_reader(self, plname):
        """
        
        Args:
        File name of a 2D xz-field generated using SIMSON     
        Returns:
        xl, zl        Physical size of the domain
        nx, ny, nz    Number of spectral modes in x,z directions
                        and number of points in y direction
        """
        f = open(plname,'rb')

        simtype = np.dtype([ ('eor1','<i4'), ('Re', '<f8'), ('eor2', '<i4'),
                            ('xl', '<f8'), ('zl', '<f8'), ('t', '<f8'),
                            ('eor3','<f8'), ('eor4','<i4',2),
                            ('nx', '<i4'), ('ny', '<i4'), ('nz', '<i4'), ('eor5','<i4',3), 
                            ('tpl', '<i4'), ('ivar', '<i4'), ('cpl', '<f8'), ('fltype', '<i4'), 
                            ('dstar', '<f8'), ('eor6','<i4',2) ])
        f.seek(0)  
        recs = np.fromfile(f, dtype=simtype, count=1)

        # Geometry of the problem
        Re = recs[0][1]
        xl = recs[0][3]
        zl = recs[0][4]

        nx = recs[0][8]
        ny = recs[0][9]
        nz = recs[0][10]
    
        header_size = recs[0].nbytes
        f.close()
        
        return Re, xl, zl, nx, ny, nz, header_size


    def preprocess_planes(self, yp, interv, nx, nz, header_size, dy_w, subsampling, target=False):
        """
        TO DO
        Args:
        yp           Height of the plane to be predicted in y+ units
        n_samp       Array with the number of snapshots in every testset
        nx, nz       Size of the snapshots
        header_size 
        dy_w         Distance of the first plane from the wall
                        (to compute the finite difference approximation of the 
                        derivative at the wall)
        target       Indicates whether we are extracting features
                        
        Returns:
        The fields for each velocity component u, v, w, stored as a 3-channel image, to be used
        as feature/target for the model
        """
        # n_ds = len(n_samp)
        n_samp_tot = np.sum(self.n_samples)

        xz_planetype = np.dtype([ ('t','<f8'), ('eor7','<f8'), ('eor8','<i4',2),
                                ('velxz', '<f8', (nx, nx)), ('eor9','<i4',2)])
        
        pl_size = xz_planetype.itemsize  
        
        # Memory allocation for the dataset

        N_VARS_IN = 3
        N_VARS_OUT = 3
        VARS_NAME_IN = ('u','w','p')
        VARS_NAME_OUT = ('u','v','w')
        
        nxd = int(nx / subsampling)
        nzd = int(nz / subsampling)
        
        if target == False:
            nfiles = N_VARS_IN
            pl_storage = np.ndarray((n_samp_tot,N_VARS_IN,nxd,nzd),dtype='float')
            l_pad = 0
        else:
            nfiles = N_VARS_OUT
            pl_storage = np.ndarray((n_samp_tot,N_VARS_OUT,nx,nz),dtype='float')
            l_pad = 0
        
        k_pl = 0 # Global plane counter
                
        for i_ds in range(len(self.n_samples)):

            if i_ds == 0:
                print('[LOADING TRAINING DATA]')
            print('Reading from file '+str(i_ds))

            if target == False:
                plname = [VARS_NAME_IN[0]+'xz_yp'+str(yp)+'_'+str(i_ds)+'.pl',
                            VARS_NAME_IN[1]+'xz_yp'+str(yp)+'_'+str(i_ds)+'.pl',
                            VARS_NAME_IN[2]+'xz_yp'+str(yp)+'_'+str(i_ds)+'.pl'
                            ]
            else:
                plname = [VARS_NAME_OUT[0]+'xz_yp'+str(yp)+'_'+str(i_ds)+'.pl',
                            VARS_NAME_OUT[1]+'xz_yp'+str(yp)+'_'+str(i_ds)+'.pl',
                            VARS_NAME_OUT[2]+'xz_yp'+str(yp)+'_'+str(i_ds)+'.pl'
                            ]

            f = []
                        
            for i_file in range(nfiles):
                
                f.append(open(self.path_input+plname[i_file],'rb'))
                f[i_file].seek(header_size + pl_size)  
                
            for i_pl in range(0, self.n_samples[i_ds]):
                for i_var in range(len(f)):          
                    xz_rec = np.fromfile(f[i_var], dtype=xz_planetype, count=1)
                    # DATA AUGMENTATION NOT IMPLEMENTED FOR THIS PROBLEM
                    j_pl = [i_pl]
                    if np.isnan(xz_rec[0][3]).any():
                        print('Nan! ') 
                    # derivatives are needed only in WallRecon
                    if target == False: 
                        # Pressure must not be divided by the differential
                        if i_var != 2: 
                            pl_storage[k_pl+j_pl[0],i_var] = np.pad(xz_rec[0][3]/dy_w,\
                                        ((int(l_pad/2),int(l_pad/2)),(int(l_pad/2),int(l_pad/2))), 'wrap')[::subsampling,::subsampling]
                        else:
                            pl_storage[k_pl+j_pl[0],i_var] = np.pad(xz_rec[0][3],\
                                        ((int(l_pad/2),int(l_pad/2)),(int(l_pad/2),int(l_pad/2))), 'wrap')[::subsampling,::subsampling]
                    else:
                        pl_storage[k_pl+j_pl[0],i_var] = np.pad(xz_rec[0][3],\
                                    ((int(l_pad/2),int(l_pad/2)),(int(l_pad/2),int(l_pad/2))), 'wrap')[::subsampling,::subsampling]


                    f[i_var].seek((interv-1)*pl_size,1)
                
            k_pl = k_pl+self.n_samples[i_ds]
                
                    
        for i_file in range(nfiles):
            f[i_file].close()

        return pl_storage
    

    def wall_dy(self, ny):
        #   ny                  # Number of point in y direction
        nfyd = 0            # Aliasing flag
        nyp = ny+nfyd*ny/2  # Actual number of points
        
        dy = 1 - math.cos(math.pi*1./float(nyp-1))
        
        return dy


    def write_TFRecord(self, X_LR, X_HR, nx_, nz_, ny_, interv, save_path):
        """
        This function writes in the TFRecord format the selected data, they can be 
        either inputs or outputs for the CNN.
        
        Parameters
        ----------
        data_in : 4D array, size [n_samples, n_components=3, nz, nx]
            Array with the input data, it must contain:
            (du/dy, dw/dy, p) 
        data_out: 4D array, size [n_samples, n_modes, n_winz, n_winx]
            Array with the output data, i.e. the POD modes in the windows    
        yp_files_in/out : integer
            Indicate where the input and output planes are located
        max_samples_per_file : integer
            Indicates the maximum number of planes saved i each TFRecord
        Returns
        -------
        None.
        """
        
        tfrecords_filename_base = save_path + f'Ret{self.Ret}_{nx_}x{nz_}x{ny_}_dt{int(0.45*100*interv)}_subsampling{self.subsampling}_yp{self.yp}_'
        
        i_samp = 0 # global index from the samples (used for debugging, irrespective of the interval of sampling)

        for i_file in range(len(self.n_samples)):

            num_smp = min(self.max_samples_per_tfr, self.n_samples[i_file])

            i_smp_file = 0
            
            if num_smp != self.max_samples_per_tfr:

                if num_smp == 0:

                    continue

                tfrecords_filename = tfrecords_filename_base + f'file{i_file:03d}samples{num_smp}_{1:03d}-of-{1:03d}.tfrecords'

                writer = tf.io.TFRecordWriter(tfrecords_filename)
                
                for i_s in range(num_smp):

                    # Take a single snapshot

                    field_lr = X_LR[i_samp]
                    field_hr = X_HR[i_samp] 

                    # Store important details about the field 
                      
                    f_nz_lr = field_lr.shape[1]   
                    f_nx_lr = field_lr.shape[2]
                    
                    f_nz_hr = field_hr.shape[1]
                    f_nx_hr = field_hr.shape[2]

                    field_lr_raw1 = np.float32(field_lr[0]).flatten().tolist()
                    field_lr_raw2 = np.float32(field_lr[1]).flatten().tolist()
                    field_lr_raw3 = np.float32(field_lr[2]).flatten().tolist()

                    field_hr_raw1 = np.float32(field_hr[0]).flatten().tolist()
                    field_hr_raw2 = np.float32(field_hr[1]).flatten().tolist()
                    field_hr_raw3 = np.float32(field_hr[2]).flatten().tolist()

                    example = tf.train.Example(features = tf.train.Features(feature={
                        'i_sample': _int64_feature(i_smp_file),
                        'nx_outer': _int64_feature(f_nx_lr),
                        'nz_outer': _int64_feature(f_nz_lr),
                        'nx_inner': _int64_feature(f_nx_hr),
                        'nz_inner': _int64_feature(f_nz_hr),
                        'outer_raw1': _floatarray_feature(field_lr_raw1),
                        'outer_raw2': _floatarray_feature(field_lr_raw2),
                        'outer_raw3': _floatarray_feature(field_lr_raw3),
                        'inner_raw1': _floatarray_feature(field_hr_raw1),
                        'inner_raw2': _floatarray_feature(field_hr_raw2),
                        'inner_raw3': _floatarray_feature(field_hr_raw3),
                        }))    
                
                    writer.write(example.SerializeToString())
                    i_samp += 1
                    i_smp_file += interv
                
                writer.close()
                print(f'Closing file {tfrecords_filename}')

            else:

                n_files, smp_left = divmod(self.n_samples[i_file], self.max_samples_per_tfr)
                
                for i_file_out in range(n_files):

                    if i_file_out == n_files:

                        num_smp = smp_left

                    tfrecords_filename = tfrecords_filename_base + f'file{i_file:03d}samples{num_smp}_{i_file_out+1:03d}-of-{n_files:03d}.tfrecords'

                    writer = tf.io.TFRecordWriter(tfrecords_filename)

                    for i_s in range(num_smp):

                        # Take a single snapshot

                        field_lr = X_LR[i_samp]
                        field_hr = X_HR[i_samp] 

                        # Store important details about the field 
                        
                        f_nz_lr = field_lr.shape[1]   
                        f_nx_lr = field_lr.shape[2]
                        
                        f_nz_hr = field_hr.shape[1]
                        f_nx_hr = field_hr.shape[2]

                        field_lr_raw1 = np.float32(field_lr[0]).flatten().tolist()
                        field_lr_raw2 = np.float32(field_lr[1]).flatten().tolist()
                        field_lr_raw3 = np.float32(field_lr[2]).flatten().tolist()

                        field_hr_raw1 = np.float32(field_hr[0]).flatten().tolist()
                        field_hr_raw2 = np.float32(field_hr[1]).flatten().tolist()
                        field_hr_raw3 = np.float32(field_hr[2]).flatten().tolist()

                        example = tf.train.Example(features = tf.train.Features(feature={
                            'i_sample': self._int64_feature(i_smp_file),
                            'nx_outer': self._int64_feature(f_nx_lr),
                            'nz_outer': self._int64_feature(f_nz_lr),
                            'nx_inner': self._int64_feature(f_nx_hr),
                            'nz_inner': self._int64_feature(f_nz_hr),
                            'outer_raw1': self._floatarray_feature(field_lr_raw1),
                            'outer_raw2': self._floatarray_feature(field_lr_raw2),
                            'outer_raw3': self._floatarray_feature(field_lr_raw3),
                            'inner_raw1': self._floatarray_feature(field_hr_raw1),
                            'inner_raw2': self._floatarray_feature(field_hr_raw2),
                            'inner_raw3': self._floatarray_feature(field_hr_raw3),
                            }))    
                    
                        writer.write(example.SerializeToString())
                        i_samp += 1
                        i_smp_file += interv
                    
                    writer.close()
                    print(f'Closing file {tfrecords_filename}')

        return

    
    def _bytes_feature(self, value):

        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def _int64_feature(self, value):

        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def _float_feature(self, value):

        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


    def _floatarray_feature(self, value):

        return tf.train.Feature(float_list=tf.train.FloatList(value=value))


    def res_block_gen(self, model, kernal_size, filters, strides):
    
        gen = model
        
        model = layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same",
            data_format='channels_first')(model)
        model = layers.BatchNormalization(momentum = 0.5)(model)
        # Using Parametric ReLU
        model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[2,3])(model)
        model = layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same",
            data_format='channels_first')(model)
        model = layers.BatchNormalization(momentum = 0.5)(model)
            
        model = layers.Add()([gen, model])
        
        return model


    def up_sampling_block(self, model, kernal_size, filters, strides):
    
        # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
        # Even we can have our own function for deconvolution (i.e one made in Utils.py)
        #model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
        model = layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same",
            data_format='channels_first')(model)
        model = layers.UpSampling2D(size = 2,
            data_format='channels_first')(model)
        model = layers.LeakyReLU(alpha = 0.2)(model)
        
        return model


    def make_genrator_model(self):

        l1 = keras.Input(shape=(self.outer_channels, int(192 /self.subsampling), int(192 /self.subsampling)), name='low-res-input')

        l2 = layers.Conv2D(
            filters=64, 
            kernel_size=(9, 9), 
            strides=(1,1),
            activation='linear',
            data_format='channels_first', 
            padding='same',
        )(l1)

        l3 = layers.PReLU(
            alpha_initializer='zeros', 
            alpha_regularizer=None, 
            alpha_constraint=None, 
            shared_axes=[2,3]
        )(l2)

        l4 = l3

        for index in range(16):

	        l4 = self.res_block_gen(l4, 3, 64, 1)


        l5 = layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same",
            data_format='channels_first')(l4)
        l6 = layers.BatchNormalization(momentum = 0.5)(l5)
        l7 = layers.Add()([l3, l6])

        # Using 2 UpSampling Blocks
        l8 = l7
        for index in range(1):

            l8 = self.up_sampling_block(l8, 3, 256, 1)

        l9 = layers.Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same",
            data_format='channels_first')(l8)
        l10 = layers.Activation('tanh')(l9)


        model = keras.Model(l1, l10, name='MMH')

        return model


    def make_discriminator_model(self):

        l1 = keras.Input(shape=(self.outer_channels, int(192 /self.subsampling), int(192 /self.subsampling)), name='low-res-input')
        
        model = layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(dis_input)
        model = layers.LeakyReLU(alpha = 0.2)(model)
        
        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)
        
        model = Flatten()(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha = 0.2)(model)
    
        model = Dense(1)(model)
        model = Activation('sigmoid')(model) 
        
        discriminator_model = Model(inputs = dis_input, outputs = model)
        
        return discriminator_model

    def srgan(self):

        generator = self.make_genrator_model()

        discriminator

        print(generator.summary())

        return

    def train(self):

        model = self.srgan()

        # print(model.summary())
        return
