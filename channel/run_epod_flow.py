# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 12:41:08 2021
@author: aguemes
"""


import os 
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import math
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

    Re, xl, zl, nx, ny, nz, h_size = header_reader(ref_file)
    
    dy_w = wall_dy(ny)

    X_HR = preprocess_planes(yp_wall, n_samples, interv, nx, nz, h_size, dy_w, subsampling_hr, target=False)
    psi, rk, basis = process_POD(X_HR, Nx, Nz, nx, nz, yp_flow, Ret)
    Y_HR = preprocess_planes(yp_flow, n_samples, interv, nx, nz, h_size, dy_w, subsampling_hr, target=True)
    Y_HR = Y_HR - np.expand_dims(np.mean(Y_HR, axis=(0, 2, 3)), axis=(0,2,3))

    Px = int(nx / Nx)
    Pz = int(nz / Nz)

    X = np.zeros((Y_HR.shape[0] * Nx * Nz, Px * Pz * Y_HR.shape[1]))

    n_t = Y_HR.shape[0]

    nS = 0

    for t in range(Y_HR.shape[0]):

        for i in range(Nx):

            for k in range(Nz):
                
                X[nS, :] = np.concatenate(
                    (
                        Y_HR[t, 0, Pz*k:Pz*(k+1), Px*i:Px*(i+1)].flatten(), 
                        Y_HR[t, 1, Pz*k:Pz*(k+1), Px*i:Px*(i+1)].flatten(), 
                        Y_HR[t, 2, Pz*k:Pz*(k+1), Px*i:Px*(i+1)].flatten()
                    )
                )

                nS += 1
    del Y_HR

    L = np.matmul(psi[:, :rk].T, X)

    np.savez(f'wallEPODbase_yp{yp_flow:03d}.npz' , L=L, basis=basis)
    
    return


def header_reader(plname):
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


def wall_dy(ny):
    #   ny                  # Number of point in y direction
    nfyd = 0            # Aliasing flag
    nyp = ny+nfyd*ny/2  # Actual number of points
    
    dy = 1 - math.cos(math.pi*1./float(nyp-1))
    
    return dy


def preprocess_planes(yp, n_samples, interv, nx, nz, header_size, dy_w, subsampling, target=False):
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
    n_samp_tot = np.sum(n_samples)

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
            
    for i_ds in range(len(n_samples)):

        if i_ds == 0:
            print(F"[LOADING  DATA]")
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
            
            f.append(open(path_input+plname[i_file],'rb'))
            f[i_file].seek(header_size + pl_size)  
            
        for i_pl in range(0, n_samples[i_ds]):
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
            
        k_pl = k_pl+n_samples[i_ds]
            
                
    for i_file in range(nfiles):
        f[i_file].close()

    return pl_storage
    

def process_POD(Y, Nx, Nz, nx, nz, yp_flow, Ret):

    Y = Y - np.expand_dims(np.mean(Y, axis=(0, 2, 3)), axis=(0,2,3))

    Px = int(nx / Nx)
    Pz = int(nz / Nz)

    X = np.zeros((Y.shape[0] * Nx * Nz, Px * Pz * Y.shape[1]))

    n_t = Y.shape[0]

    nS = 0

    for t in range(Y.shape[0]):

        for i in range(Nx):

            for k in range(Nz):
                
                X[nS, :] = np.concatenate(
                    (
                        Y[t, 0, Pz*k:Pz*(k+1), Px*i:Px*(i+1)].flatten(), 
                        Y[t, 1, Pz*k:Pz*(k+1), Px*i:Px*(i+1)].flatten(), 
                        Y[t, 2, Pz*k:Pz*(k+1), Px*i:Px*(i+1)].flatten()
                    )
                )

                nS += 1
    del Y

    if not os.path.exists(f'/storage2/alejandro/urban/re{Ret}/wallPODbase_yp{yp_flow:03d}.npz'):
        print('[COMPUTING SPATIAL CORRELATION MATRIX]')
        C = np.matmul(X.T, X)
        print('[COMPUTING POD]')
        phiT, sigma, phi = np.linalg.svd(C, full_matrices=False)
        del C
        del phiT
        print('[COMPUTING TEMPORAL COEFFICIENTS]')
        psi = np.matmul(X, phi.T)

        sigma = np.diag(sigma)
        rk = np.linalg.matrix_rank(sigma)

        print(rk)
        rk = int(rk)
        basis = np.matmul(phi[:rk, :].T, np.linalg.inv(sigma[:rk, :rk]))
        del sigma
        # np.savez(f'/storage2/alejandro/urban/re{Ret}/wallEPODbase_yp{yp_flow:03d}.npz' , L=L)
        del phi
        del X

    return psi, rk, basis


def write_TFRecord(X_LR, X_HR, Y_HR, Y_PD, Ret, nx, nz, ny, interv, yp_flow, subsampling_lr, save_path, n_samples, max_samples_per_tfr):
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
    
    tfrecords_filename_base = save_path + f'Ret{Ret}_{nx}x{nz}x{ny}_dt{int(0.45*100*interv)}_subsampling{subsampling_lr}_yp{yp_flow}_'

    n_sets = int(np.ceil(np.sum(n_samples) / max_samples_per_tfr))

    i_samp = 0
    i_samp_file = 0

    for n_set in range(n_sets):

        if (n_set + 1) * max_samples_per_tfr > np.sum(n_samples):

            samples_per_tfr = np.sum(n_samples) - n_set * max_samples_per_tfr

        else:

            samples_per_tfr = max_samples_per_tfr

        tfrecords_filename = tfrecords_filename_base + f'file_{n_set+1:03d}-of-{n_sets:03d}_samples{samples_per_tfr:03d}.tfrecords'
        writer = tf.io.TFRecordWriter(tfrecords_filename)

        for idx in range(samples_per_tfr):

            wall_lr = X_LR[i_samp]
            wall_hr = X_HR[i_samp] 
            flow_hr = Y_HR[i_samp]
            flow_pd = Y_PD[i_samp] 
            n_modes = flow_pd.shape[0]

            wall_lr_raw1 = np.float32(wall_lr[0]).flatten().tolist()
            wall_lr_raw2 = np.float32(wall_lr[1]).flatten().tolist()
            wall_lr_raw3 = np.float32(wall_lr[2]).flatten().tolist()

            wall_hr_raw1 = np.float32(wall_hr[0]).flatten().tolist()
            wall_hr_raw2 = np.float32(wall_hr[1]).flatten().tolist()
            wall_hr_raw3 = np.float32(wall_hr[2]).flatten().tolist()

            flow_hr_raw1 = np.float32(flow_hr[0]).flatten().tolist()
            flow_hr_raw2 = np.float32(flow_hr[1]).flatten().tolist()
            flow_hr_raw3 = np.float32(flow_hr[2]).flatten().tolist()

            flow_pd_raw = np.float32(flow_pd).flatten().tolist()

            example = tf.train.Example(
                features = tf.train.Features(
                    feature={
                        'i_samp': _int64_feature(i_samp_file),
                        'n_x':  _int64_feature(nx),
                        'n_y':  _int64_feature(ny),
                        'n_z':  _int64_feature(nz),
                        'n_modes': _int64_feature(n_modes),
                        'subsampling':  _int64_feature(subsampling_lr),
                        'wall_lr_raw1':  _floatarray_feature(wall_lr_raw1),
                        'wall_lr_raw2':  _floatarray_feature(wall_lr_raw2),
                        'wall_lr_raw3':  _floatarray_feature(wall_lr_raw3),
                        'wall_hr_raw1':  _floatarray_feature(wall_hr_raw1),
                        'wall_hr_raw2':  _floatarray_feature(wall_hr_raw2),
                        'wall_hr_raw3':  _floatarray_feature(wall_hr_raw3),
                        'flow_hr_raw1':  _floatarray_feature(flow_hr_raw1),
                        'flow_hr_raw2':  _floatarray_feature(flow_hr_raw2),
                        'flow_hr_raw3':  _floatarray_feature(flow_hr_raw3),
                        'flow_pd_raw':  _floatarray_feature(flow_pd_raw)
                    }
                )
            )

            writer.write(example.SerializeToString())

            i_samp += 1 
            i_samp_file += interv

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

    Ret = 180
    yp_wall = 0
    yp_flow = 15
    subsampling_hr = 1
    subsampling_lr = 1

    if Ret == 180:

        Nx = 12
        Nz = 12
        interv = 3
        max_samples_per_tfr = 120
        save_path = "/storage2/alejandro/urban/re180/train/.tfrecords/"
        path_input = '/storage3/luca/PhD/015-Madrid/simson/015-Ret180_192x192x65/Train/'
        ref_file = '/storage3/luca/PhD/015-Madrid/simson/015-Ret180_192x192x65/Train/uxz_yp0_0.pl'
        n_samples = (4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200)
        n_samples = (4200, 4200, 4200, 4200, 4200, 4200)

    main()