# AUTOMATING-THE-DETECTION-OF-FORGED-BANKNOTES-USING-K-MEANS-CLUSTERING

## About: 
The purpose of the project is to classify banknotes as Forged or Genuine depending on the parameters given in the dataset. These parameters are the characteristics that are derived from different samples of banknotes using the wavelet transforming tool measured using devices. An unsupervised learning metric, K-means clustering will be used to do cluster analysis on the dataset.

## Description of the Dataset:
We are using a simplified version of the banknotes dataset available at OpenML. It contains 2 features — V1 (variance) and V2 (skewness) which are numerical values which represent the features of the notes. The dataset is a mix of values for both Forged and Genuine banknotes. Hence, we are using Kmeans cluster algorithm to filter out both the categories. 

## Files and folders in the project :
Banknote_authentication.py - Contains the python code for the project
Banknote_authentication.csv - The Dataset retrived from [OpenML](https://www.openml.org/d/1462)
AUTOMATING THE DETECTION OF FORGED BANKNOTES.pdf - Report on the project

## Setting up the project:
The following python libraries needs to be installed for the program to run:
* Numpy
* Pandas
* Sklearn
* Matplotlib

## Running the program :
Run the program using Python Interpreter (Anaconda Distribution is recommended)
