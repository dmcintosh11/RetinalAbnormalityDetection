# RetinalAbnormalityDetection

This project is a way to identify individuals with diseased eyes based off of a retinal scan. I explored using a CNN architecture as well as an AutoEncoder for anomaly detection.

## AnomalyDetectorCNNAttempts.py
This file containts several attempted CNN models while trying to work around the limitations of my local compute.

## AnomalyDetectorAEAttempt1.py
This file is one version of the AutoEncoder anomaly detection model

## AnomalyDetectorAEAttempt2.py
This file is another version of the AutoEncoder anomaly detection model attempt


## Autoencoder.py
This file is an unfinished attempt at utilizing an autoencoder in a non conventional way to train on only the normal healthy retinal scan data, so that it compresses and expands data poorly on diseased retinal scans since they are abnormal and look different than what the model was originally trained on
