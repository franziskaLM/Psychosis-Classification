Psychosis Classification Using fMRI Data (Based on Kaggle Competition)
 
Project Overview

This project focuses on classifying schizophrenia (SZ) and bipolar disorder (BP) using resting-state fMRI data. Functional Network Connectivity (FNC) matrices derived from Intrinsic Connectivity Networks (ICNs) serve as inputs to deep learning models.

Data

Resting-state fMRI data from 183 BP and 288 SZ subjects
Brain activity represented across 105 Regions of Interest (ROIs)
FNC matrices calculated using Pearson and Partial correlation from ICN time courses
Synthetic training data generated via Mix-Up to address class imbalance
Models

Dense Neural Networks (DNN) using vectorized FNC matrices
Convolutional Neural Networks (CNN) with one or two input channels (Pearson and Partial correlations)
Usage

Best performance achieved by CNN with two-channel input and synthetic training data (ROC-AUC ~0.64)
Synthetic data improves class balance and model prediction accuracy
DNN performance declines when trained with synthetic data
Repository Structure

Data/ or Original Data/: raw and preprocessed fMRI data files
Scripts/: Python scripts for training and evaluation
Results/: evaluation outputs, plots, and logs
References

Data sourced from Kaggle competition [https://www.kaggle.com/competitions/psychosis-classification-with-rsfmri] and methodology inspired by Meszlényi et al. (2021).
