# **Reconstruction of turbulent flows thorugh deep learning**

This repository contains several Python implementations of deep neural networks for the resolution enhancement of turbulent wall measurements and the reconstruction of turbulent velocity fields from wall data.

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
guest@king:~$ python run_generate_tfrecords_training.py 
guest@king:~$ python run_generate_tfrecords_testing.py 
```

To run the training procedure, execute:

```console
guest@king:~$ python run_training_resolution_gan.py
guest@king:~$ python run_training_reconstruction_gan.py
guest@king:~$ python run_training_reconstruction_fcnpod.py
```

To compute the prediction of the testing dataset, execute:

```console
guest@king:~$ python run_predictions_resolution_gan.py
guest@king:~$ python run_predictions_reconstruction_gan.py
guest@king:~$ python run_predictions_reconstruction_fcnpod.py
```

## **Publications**
This repository has been used for the following scientific publications:

Güemes, A., Discetti, S., Ianiro, A., Sirmacek, B., Azizpour, H., & Vinuesa, R. (2021). From coarse wall measurements to turbulent velocity fields through deep learning. *arXiv preprint arXiv:2103.07387.*

## **Authorship**
This repository has been developed in collaboration between the KTH Royal Institute of Technology and Universidad Carloss III de Madrid. The following researches and students, alphabetically lister, are acknowledged for their contributions:
- Alejandro Güemes
- Andrea Ianiro
- Beril Sirmacek
- Hampus Tober
- Hossein Azizpour
- Ricardo Vinuesa
- Stefano Discetti

## **Contributing**
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## **License**
[Creative Commons](https://creativecommons.org)
