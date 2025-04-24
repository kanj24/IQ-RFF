# IQ-RFF

This program uses tensorflow 2.10.0 ([Docker: joseasanchezviloria/my-repo:tensorflow-CAAI-FAU-HPC](https://hub.docker.com/layers/joseasanchezviloria/my-repo/tensorflow-CAAI-FAU-HPC/images/sha256-359a14f949900b1e40539eda574afb32e5f5c2c6f969807306d2ba7e74acc330)). 

This program is built to use IQ data packeted into different sizes. Ensure that the data is structured like shown in the format below in HDF5 files containing two tables:
	1. Data: 128bit complex values
	2. Label: 32bit integers
There should be an equal number of packets per device. 


Depending on the choice of inputs (spectrogram/slices) and models (resent/transformer/lstm) follow instructions in the comments of to include or exclude relevant functionalities. 

To run, simply setup the location of your datasets along with packet and device information in main.py and run the script from the shell. 

Code inspired and adapted from [1],[2],[3].

References: 

[1] G. Shen, J. Zhang, A. Marshall and J. R. Cavallaro, "Towards Scalable and Channel-Robust Radio Frequency Fingerprint Identification for LoRa," in IEEE Transactions on Information Forensics and Security, vol. 17, pp. 774-787, 2022, doi: 10.1109/TIFS.2022.3152404.

[2] G. Shen, J. Zhang, A. Marshall, M. Valkama and J. R. Cavallaro, "Toward Length-Versatile and Noise-Robust Radio Frequency Fingerprint Identification," in IEEE Transactions on Information Forensics and Security, vol. 18, pp. 2355-2367, 2023, doi: 10.1109/TIFS.2023.3266626.

[3] G. Reus-Muns, D. Jaisinghani, K. Sankhe and K. R. Chowdhury, "Trust in 5G Open RANs through Machine Learning: RF Fingerprinting on the POWDER PAWR Platform," GLOBECOM 2020 - 2020 IEEE Global Communications Conference, Taipei, Taiwan, 2020, pp. 1-6, doi: 10.1109/GLOBECOM42002.2020.9348261.

