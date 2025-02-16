# RetinalAbnormalityDetection

This project is a way to identify individuals with diseased eyes based off of a retinal scan. I explored using a CNN architecture as well as an AutoEncoder for anomaly detection.

When researching how to conduct this project, I came across using an AutoEncoder in a non conventional way to train on only the normal healthy retinal scan data, so that it learns to effectively compress and expand healthy image scans. This means the reconstructive loss for diseased retinal scans will be higher since they are abnormal from what the model was originally trained on. That would allow us to use a reconstructive loss threshold to classify as healthy or abnormal.

## AnomalyDetectorCNNAttempts.py
This file containts several attempted CNN models while trying to work around the limitations of my local compute.

## AnomalyDetectorAEAttempt1.py
This file is an attempt at using the AutoEncoder anomaly detection model

## AnomalyDetectorAEAttempt2.py
This file is another attempt at using the AutoEncoder anomaly detection model attempt
