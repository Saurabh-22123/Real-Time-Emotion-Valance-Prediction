# Valence Regression:- RF Baseline vs LSTM

## Overview
This project predicts valence which is a number that measures how positive or negative an emotion is, using visual features.
The two machine learning models which are built and compared here:

**1. Random Forest (RF):** a classical model used as the baseline
**2. LSTM:** a deep learning model used as the main model

Both models are tested on the same data.


## What This Project Does
1. Loads three CSV files: train, validation, and test sets
2. Cleans the data by replacing missing and infinite values
3. Scales the features so all values are on the same range
4. Trains a Random Forest model as a baseline
5. Trains an LSTM neural network using PyTorch
6. Compares both models using three error metrics: MSE, RMSE, and MAE
7. Plots the results as a bar chart


## Results
|     Model     |      MSE      |    RSME      |     MAE       |
| ------------- | ------------- |------------- | ------------- |
| RF (Baseline) |     0.0821    |    0.2866    |    0.2368     |
|     LSTM      |     0.1131    |    0.3362    |    0.2901     |

The Random Forest performed better than the LSTM. This is common when the data has no time sequence, the LSTM loses its main advantage on static tabular features.


## Learning
**1. Data leakage:** the scaler must be fitted only on training data. Fitting it on test data gives falsely good results.

**2. Always build a baseline first:** a simple model shows whether a complex model is actually needed.

**3. LSTMs need sequences:** when each sample is just one time step, the LSTM behaves like a basic neural network and often loses to tree-based models.

**4. Dropout prevents overfitting:** a dropout rate of 0.5 was applied after the LSTM layer to improve generalisation.

**5. PyTorch basics:** zero gradients → forward pass → compute loss → backpropagate → update weights.


## Real-World Uses
|             Area            |                  How It Helps                   |
| --------------------------- | ----------------------------------------------- |
|         Mental health       | Detects mood changes from facial or visual data |
|         Driver safety       |        Flags stress or fatigue while driving    |
|         Content apps        |     Recommends content based on emotional tone  |
|         Advertising         | Measures emotional response to images or videos |
|  Human-computer interaction |   Adapts system behaviour based on user emotion |



