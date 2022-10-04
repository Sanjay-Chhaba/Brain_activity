## Realtime Prediction of Muscle Movement Using EEG-based Brain-Machine Interface

## Intoduction

This project aims to predict a person's intended hand and arm movement using their brain activity. The data was collected during an experiment where subjects performed a series of grasp and lift trials while wearing a 64 channel EEG headset. 


The below figure represents a downsampled version of the electrodes’ activity in a timeframe of 50 millisecond before, and 100 millisecond after the event onset for the first task (Called “HandStart”). The sampling rate is 500 Hz, which means the readings were collected every 2 milliseconds. 

![alt text](https://github.com/Atlaskz/Bionic-AI-Predicting-Grasp-and-Lift-Motions/blob/main/Images/e1.png?style=centerme)


The above graph has a distinct peak reflecting the action potential collected by electrodes placed on the motor cortex. Such peaks are not always observable when using a non invasive EEG device. This is mainly due to the noise and artifacts resulting from the large distance between the electrodes and the signal source. The following figures represent tasks 2 and 3 of the same trial:


![alt text](https://github.com/Atlaskz/Bionic-AI-Predicting-Grasp-and-Lift-Motions/blob/main/Images/e2.png?style=centerme)

![alt text](https://github.com/Atlaskz/Bionic-AI-Predicting-Grasp-and-Lift-Motions/blob/main/Images/e3.png?style=centerme)

In order to make such peaks more defined and observable, noise reduction methods such as Independent Component Analysis are used. However, this wasn't the main focus of this project.

## Training

Due to the timeseries nature of EEG data, previous timepoints are used as features during training. The input to the model is a 3d array of data in batches of size 2000. Each batch includes a single reading and 511 previous timepoints (window size = 512). However, to downsample the data, every other time point was used, resulting in a window size of 256. Given that data was colected from 32 electrodes, the shape of the arrays used in training were 2000x256x32.

There are 6 depended variables each representing one of the 6 tasks. When training the model, a binary cross entropy loss was used rather than categorical to allow for independent prediction of each task. This is because events overlapped at some instances of a each trial (see the figure above).

In order to compare performance, 3 scenarios were designed: 

1. Training a model for each subject individually
2. Training a general model on all subjects and testing on a new subject 
3. Training the Model once on all subjects and a second time on each subject individually

The aim of creating these scenarios was to explore the potential of creating a general model that performs well on new subjects.

For each scenario, a base model (Logistic Regression) and a Deep Learning Model (1D Convolutional Neural Network) were trained.

## Results

Below is a graph of the AUC score from the 6 models. AUC was used as the performance metric due to is its ability to measure separability of the 2 classes (0s and 1s), which is important for imbalanced data.

<p align="center">
  <img src="https://github.com/Atlaskz/Bionic-AI-Predicting-Grasp-and-Lift-Motions/blob/main/Images/results.png">
</p>

As shown in the figure above, scenario 1 performed the worst while the advanced model from scenario 3 performed best. The aim of this project was to explore the potential of creating a general model that works well on new subjects. As seen in results from scenario 2, a CNN can predict these moevments for a new subject with an average accuarcy of X%.

The above graph shows the true events (in orange) and the corresponding predictions (in blue). The lower graph represents the activity of the 32 electrodes during the trial.

In the top graph, the least overlap is observed in event 5, which could be a result of the custom loss function created for finding a threshold for the binary classification. However, each event occurs in a timeframe of 75 milliseconds, which means event 5's prediction is delayed by less than 0.075 seconds only.


## Discussion

In this project, the use of deep learning for decoding brain activity was explored. It was obsevered that with enough data and a large number of subjects, a general model can perform well on new subjects. However, for this to be used in prosthetics, anything less than 100% accuracy is not good enough. Hence, sufficient data needs to be collected and such models would have to be tuned for maximizingtheir performance.

