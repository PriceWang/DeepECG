<!--
 * @Author: Guoxin Wang
 * @Date: 2024-02-07 07:47:56
 * @LastEditors: Guoxin Wang
 * @LastEditTime: 2024-02-07 11:58:25
 * @FilePath: /DeepECG/README.md
 * @Description: 
 * 
 * Copyright (c) 2024 by Guoxin Wang, All Rights Reserved. 
-->
# Deep ECG

## Introduction

Build a deep neural network for ECG authentication

[Low Complexity ECG Biometric Authentication for IoT Edge Devices](https://ieeexplore.ieee.org/document/9332012)

The whole project contains 3 different methods to rebuild model and execute ECG authentication:

- Original CNN

    This rebuilds model from a pre-trained CNN model and use its **original weights**. The best performance is **99.63%**

- Binary Neural Network

    This rebuilds model from a pre-trained CNN model and use **binary weights**. The weights are in this format:

    <div align=center> <img src="https://latex.codecogs.com/svg.latex?\pm1"> </div>

    The best performance is **88.85%**

- Exponentiation Neural Network

    This rebuilds model from a pre-trained CNN model and use **exponent weights**. The weights are in this format:

    <div align=center> <img src="https://latex.codecogs.com/svg.latex?\frac{INTEGER}{2^n}"> </div>

    **The performance increases with larger 'n' value**

## Requirements and Installation

- Database

    [PTB Diagnostic ECG Database](https://physionet.org/content/ptbdb/1.0.0/)

- Libraries

    Install the necessary libraries:

    ```cmd
    conda env create -n deepecg --file environment.yml
    ```

    Activate environment:

    ```cmd
    conda activate deepecg
    ```

## Usage

- Step 1: Data Processing

    To precess data from original dataset manually, run:

    ```cmd
    python DataGeneration.py \
        --data_path ${data_path} \
        --prefix ${prefix} \
        --output_path ${output_path}
    ```

    Or you can download the pre-processed dataset from the link:

    [PTB Processed Dataset](https://drive.google.com/file/d/1W1LkLuK3uwxJskv_1KAMDvivPbCyGgU4/view?usp=sharing)

- Step 2: CNN Modelling

    To build CNN model for human recognition, run:

    ```cmd
    python ModelCreation.py \
        --save_path ${save_path} \
        --data_path ${data_path}
    ```

- Step 3: Authentication

    To execute authentication section, run:

    ```cmd
    python Authentication.py \
        --model_path ${model_path} \
        --data_path ${data_path}
    ```

## Update

- 2021/01/20

    optimize structure, improve performance, test generalization ability

- 2021/02/05

    optimize path search

- 2021/03/16 *

    update path structure

- 2021/05/10

    pruning
