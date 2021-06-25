# **Reconstruction of wall turbulence data thorugh deep learning**

This repository contains several Python implementation of deep neural networks for the resolution enhancement of turbulent wall measurements and the reconstruction of turbulent velocity fields from wall data.

Currently, one case is available:

*   Turbulent-channel flow: DNS data from a turbulent channel flow with friction Reynolds number ![formula](https://render.githubusercontent.com/render/math?math=Re_{\tau}=180) 

## **Installation**

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required dependencies.

```bash
pip install -r requirements.txt
```

## **Usage**

To generate the tfrecord files, execute:

```console
guest@local:~$ python run_generate_tfrecords_training.py 
guest@local:~$ python run_generate_tfrecords_testing.py 
```

To run the training procedure, execute:

```console
guest@vegeta:~$ python run_training_resolution_gan.py
guest@vegeta:~$ python run_training_reconstruction_gan.py
guest@vegeta:~$ python run_training_reconstruction_fcnpod.py
```

To compute the prediction of the testing dataset, execute:

```console
guest@vegeta:~$ python run_predictions -c channel -u 4 -m architecture01 -l 1e-4
```

## **Publications**
This repository has been used for the following scientific publications:

*To be announced*

## **Authorship**
This repository has been developed in collaboration between the KTH Royal Institute of Technology and Universidad Carloss III de Madrid. The following researches and students are acknowledged for their contributions:
- Alejandro Güemes
- Hampus Tober
- Beril Sirmacek
- Stefano Discetti
- Andrea Ianiro
- Hossein Azizpour

## **Contributing**
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## **License**
[Creative Commons](https://creativecommons.org)
