# Deep ECG

## Introduction

Build a deep neural network for ECG authentication

The whole project contains 3 different methods to rebuild model and execute ECG authentication:

- Original CNN

    This rebuilds model from a pre-trained CNN model and use its **original weights**. The best performance is **99.63%**

- Binary Neural Network

    This rebuilds model from a pre-trained CNN model and use **binary weights**. The weights are in this format:

    <div align=center> <img src="https://latex.codecogs.com/gif.latex?%5Cpm%7B1%7D"> </div>

    The best performance is **88.85%**

- Exponentiation Neural Network

    This rebuilds model from a pre-trained CNN model and use **exponent weights**. The weights are in this format:

    <div align=center> <img src="https://latex.codecogs.com/gif.latex?%5Cfrac%7BINTEGER%7D%7B2%5En%7D"> </div>

    **The performance increases with larger 'n' value**

Performance of all structures as shown:

<div align=center> <img src=performance.png width = 60% height = 60%> </div>

## Requirements and Installation

- Database

    [PTB Diagnostic ECG Database](https://physionet.org/content/ptbdb/1.0.0/)

    Download and unzip it to the root directory. The structure should be like this:

    ```txt
    - DeepECG
        - ...
        - ptb-diagnostic-ecg-database-1.0.0
            - patient001
            - ...
            - patient294
            - ...
        - ...
    ```

- Libraries

    Install the necessary libraries:

    ```cmd
    pip install -r requirements.txt
    ```

## Usage

- Step 1: Data Processing

    To precess data from original dataset manually, run:

    ```cmd
    python DataGeneration.py
    ```

    Or you can download the pre-processed dataset from the link:

    [PTB Processed Dataset](https://drive.google.com/file/d/1W1LkLuK3uwxJskv_1KAMDvivPbCyGgU4/view?usp=sharing)

    Put it on the root directory:

    ```txt
    - DeepECG
        - ...
        - PTB_dataset.csv
        - ...
    ```

- Step 2: CNN Modelling

    To build CNN model for human recognition, run:

    ```cmd
    python ModelCreation.py
    ```

    This model has been trained and uploaded as 'model.h5'

- Step 3: Authentication

    To execute authentication section, run:

    ```cmd
    python Authentication.py
    ```

## Update

- 2021/01/20

    optimize structure, improve performance, test generalization ability
