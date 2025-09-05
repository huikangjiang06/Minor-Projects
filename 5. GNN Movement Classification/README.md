# Project Description

This project aims to implement an automatic classification system for figure skating jump actions. Given the time-series skeletal joint data of figure skaters, the task is to build a deep learning model capable of accurately classifying different types of jump actions. This system can be used for competition analysis, athlete training assistance, and other scenarios.

# Data Description

The data file is FS_data_for_HDGCN_jump.npz, located in the ./train_data/ directory. The dataset includes training and test sets, with the following data structure:

x_train: Training set skeletal joint data, with a shape of (N_train, C, T, V, M)

N_train: Number of training samples
C: Number of channels (e.g., 3, representing x, y, z coordinates)
T: Number of time frames
V: Number of skeletal joints
M: Number of people (typically 1)
y_train: Training set labels, with a shape of (N_train, num_class), in one-hot encoding
x_test: Test set skeletal joint data, with the same structure as above
y_test: Test set labels, with the same structure as above (used only for evaluation)
The total number of label categories is 17, corresponding to 17 types of figure skating jump actions.

# Figure Skating Jump Action Classification System

This system is based on deep learning methods and uses a simplified convolutional neural network (SimpleHDGCN) to extract spatiotemporal features from skeletal joint time-series data and perform classification. The main workflow is as follows:

Data Loading and Preprocessing: Read the npz file and convert it into a format suitable for model input.
Model Structure: Use multiple layers of convolution, pooling, and fully connected layers to extract spatiotemporal features and achieve action classification.
Training and Evaluation: Train the model on the training set and evaluate its accuracy on the test set.
Prediction and Result Saving: Predict on the test set and save the prediction results as submission.csv, with the format index,label.
This project is suitable for beginners and researchers to quickly get started with figure skating action recognition tasks. The code structure is clear and easy to extend and improve.