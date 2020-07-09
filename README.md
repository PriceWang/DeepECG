# BinaryDeepECG

## Introduction

Binary CNN for ECG Authentication

## Requirements and Installation

- Database

    PTB Diagnostic ECG Database

    <https://physionet.org/content/ptbdb/1.0.0/>

    Download and unzip it to the root directory. The structure should be like this:

    ```txt
    - BinaryDeepECG
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

### Step 1: Data Processing

To precess data from original dataset manually, run:

```cmd
python DataGeneration.py
```

Or you can download the pre-processed dataset from the link:

<https://drive.google.com/file/d/1XcaSe04qPCxuQQyD_WXPDmCOV8rqBboe/view?usp=sharing>

Put it on the root directory:

```txt
- BinaryDeepECG
    - ...
    - PTB_dataset.csv
    - ...
```

### Step 2: CNN Modelling

To build CNN model for human recognition, run:

```cmd
python ModelCreation.py
```

This model has been trained and uploaded as 'model.h5'

### Step 3: Authentication
