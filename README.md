# FE-DGCAN

## Description

- `00_train.py`

  - "Pre-training mode":

    - This script pre-trains a model for each machine type using the directory `dev_data/<machine_type>/train/`.

  - "Unsupervised contrastive classification learning mode":

    - This script performs contrastive learning for each machine type using the directory `eval_data/<machine_type>/train/`.

- `01_test.py`

  - "Training dataset detection results":

    - This script generates a CSV file for each section, containing anomaly scores for each WAV file in the `dev_data/<machine_type>/test/` directory.
    - The CSV file will be stored in the `result/` directory.
    - It also generates a CSV file containing AUC, pAUC, precision, recall, and F1 scores for each section.

  - "Testing dataset detection results":

    - This script generates a CSV file for each section, containing anomaly scores for each WAV file in the `eval_data/<machine_type>/test/` directory.
    - The CSV file will be stored in the `result/` directory.

## Usage

### 1. Clone the Repository

Clone this repository from GitHub.

### 2. Download the Dataset

Download the datasets:

- "Development dataset":

  - Download `dev_data_<machine_type>.zip` from [https://zenodo.org/record/6355122](https://zenodo.org/record/6355122).

- "Additional training dataset", which is used for training and evaluation.
- "Evaluation dataset", which is used for testing.

### 3. Unzip dataset

Unzip the downloaded files and make the directory structure as follows::

- ./dcase2022_task2_baseline_ae
  - /00_train.py
  - /01_test.py
  - /common.py
  - /keras_model.py
  - /baseline.yaml
  - /readme.md
  - /dgcan.py
  - /feature_enhancement.py
  - /train_process.py
- /dev_data
  - /fan
    - /train (only normal clips)
      - /section*00_source_train_normal_0000*<attribute>.wav
      - ...
      - /section*00_source_train_normal_0989*<attribute>.wav
      - /section*00_target_train_normal_0000*<attribute>.wav
      - ...
      - /section*00_target_train_normal_0009*<attribute>.wav
      - /section*01_source_train_normal_0000*<attribute>.wav
      - ...
      - /section*02_target_train_normal_0009*<attribute>.wav
    - /test
      - /section*00_source_test_normal_0000*<attribute>.wav
      - ...
      - /section*00_source_test_normal_0049*<attribute>.wav
      - /section*00_source_test_anomaly_0000*<attribute>.wav
      - ...
      - /section*00_source_test_anomaly_0049*<attribute>.wav
      - /section*00_target_test_normal_0000*<attribute>.wav
      - ...
      - /section*00_target_test_normal_0049*<attribute>.wav
      - /section*00_target_test_anomaly_0000*<attribute>.wav
      - ...
      - /section*00_target_test_anomaly_0049*<attribute>.wav
      - /section*01_source_test_normal_0000*<attribute>.wav
      - ...
      - /section*02_target_test_anomaly_0049*<attribute>.wav
    - attributes_00.csv (attribute csv for section 00)
    - attributes_01.csv (attribute csv for section 01)
    - attributes_02.csv (attribute csv for section 02)
  - /gearbox (The other machine types have the same directory structure as fan.)
  - /bearing
  - /slider (`slider` means "slide rail")
  - /ToyCar
  - /ToyTrain
  - /valve
- /eval_data
  - /fan
    - /train (after launch of the additional training dataset)
      - /section*03_source_train_normal_0000*<attribute>.wav
      - ...
      - /section*03_source_train_normal_0989*<attribute>.wav
      - /section*03_target_train_normal_0000*<attribute>.wav
      - ...
      - /section*03_target_train_normal_0009*<attribute>.wav
      - /section*04_source_train_normal_0000*<attribute>.wav
      - ...
      - /section*05_target_train_normal_0009*<attribute>.wav
    - /test (after launch of the evaluation dataset)
      - /section_03_test_0000.wav
      - ...
      - /section_03_test_0199.wav
      - /section_04_test_0000.wav
      - ...
      - /section_05_test_0199.wav
    - attributes_03.csv (attribute csv for train data in section 03)
    - attributes_04.csv (attribute csv for train data in section 04)
    - attributes_05.csv (attribute csv for train data in section 05)
  - /gearbox (The other machine types have the same directory structure as fan.)
  - /bearing
  - /slider (`slider` means "slide rail")
  - /ToyCar
  - /ToyTrain
  - /valve

### 4. Modify Parameters

You can change the parameters for feature extraction and model definitions by editing `baseline.yaml`.

### 5. Run the Training Script (for Development Dataset)

Run the training script `00_train.py`.
Use the option `-d` to specify the development dataset `dev_data/<machine_type>/train/`.

Options:

| Parameter | Description |                                                  |
| --------- | ----------- | ------------------------------------------------ |
| `-h`      | `--help`    | Application help.                                |
| `-v`      | `--version` | Display application version.                     |
| `-d`      | `--dev`     | Development dataset mode.                        |
| `-e`      | `--eval`    | Additional training and evaluation dataset mode. |

`00_train.py` trains a model for each machine type and stores the trained model in the `model/` directory.

### 6. Run the Testing Script (for Development Dataset)

Run the testing script `01_test.py`.
Use the option `-d` to specify the development dataset `dev_data/<machine_type>/test/`.

```
$ python 01_test.py -d
```

The options for `01_test.py` are the same as for `00_train.py`.
`01_test.py` computes anomaly scores for each WAV file in `dev_data/<machine_type>/source_test/` and `dev_data/<machine_type>/target_test/`.
Each section’s CSV file (containing anomaly scores) will be stored in the `result/` directory.
If in "pre-training mode", the script will output a corresponding CSV file containing AUC, pAUC, precision, recall, and F1 scores.

### 7. Check the Results

You can view the anomaly scores in the CSV file `anomaly_score_<machine_type>_section_<section_index>_test.csv` located in the `result/` directory.
Each anomaly score corresponds to a WAV file in the `dev_data/<machine_type>/test/` directory.

`anomaly_score_fan_section_00_test.csv`

```
section_00_source_test_normal_0000_m-n_W.wav  -30.378235
section_00_source_test_normal_0001_m-n_X.wav  -32.738876
section_00_source_test_normal_0002_m-n_W.wav  -31.964493
section_00_source_test_normal_0003_m-n_W.wav  -32.687504
section_00_source_test_normal_0004_m-n_X.wav  -25.423658
section_00_source_test_normal_0005_m-n_W.wav  -35.16318
  ...
```

Threshold-processed anomaly detection results can be viewed in the CSV file `decision_result_<machine_type>_section_<section_index>_test.csv`:

`decision_result_fan_section_00_test.csv`

```
section_00_source_test_normal_0000_m-n_W.wav,0
section_00_source_test_normal_0001_m-n_X.wav,0
section_00_source_test_normal_0002_m-n_W.wav,0
section_00_source_test_normal_0003_m-n_W.wav,0
section_00_source_test_normal_0004_m-n_X.wav,0
section_00_source_test_normal_0005_m-n_W.wav,0
  ...
```

Additionally, you can check performance metrics such as AUC, pAUC, precision, recall, and F1 score in the `result.csv` file:

`result.csv`

```
fan
                AUC(source)  AUC(target)   pAUC  precision(source)  precision(target) recall(source) recall(target) F1 score(source)  F1 score(target)
00                 0.858        0.622     0.725       1.0              0.625            0.15           0.15           0.260              0.241
01                 0.749        0.762     0.712       0.939            0.62             0.31           0.31           0.466              0.413
02                 0.691        0.577     0.653       0.863            0.686            0.57           0.57           0.686              0.622
arithmetic mean    0.766        0.620     0.763       0.934            0.643            0.343          0.343          0.471              0.426
harmonic mean      0.760        0.690     0.556       0.930            0.642            0.257          0.257          0.403              0.3677

  ...
valve
                AUC(source)  AUC(target)   pAUC  precision(source)  precision(target) recall(source) recall(target) F1 score(source)  F1 score(target)
00                 0.786        0.675     0.768       0.928            0.812            0.13           0.13           0.228              0.224
01                 0.563        0.637     0.744       0.833            0.833            0.2            0.2            0.322              0.322
02                 0.768        0.806     0.857       1.0              1.0              0.73           0.73           0.843              0.843
arithmetic mean    0.706        0.639     0.657       0.920            0.881            0.353          0.353          0.464              0.463
harmonic mean      0.690        0.610     0.630       0.915            0.874            0.213          0.213          0.346              0.342

                                                                   AUC(source)  AUC(target)   pAUC   precision(source)  precision(target) recall(source)  recall(target) F1 score(source)  F1 score(target)
"arithmetic mean over all machine types  sections  and domains"     0.679          0.531      0.576     0.685               0.580            0.392            0.392          0.439             0.419
"harmonic mean over all machine types  sections  and domains"       0.641          0.477      0.563     0.00                0.00             0.0              0.0            0.0               0.0

```

## Dependencies

### Software Packages

- p7zip-full
- Python == 3.8.20
- FFmpeg

### Python Packages

- Keras == 2.10.0
- Keras-Preprocessing == 1.1.2
- matplotlib == 3.5.1
- numpy == 1.24.4
- PyYAML == 6.0
- scikit-learn == 1.0.2
- scipy == 1.10.1
- librosa == 0.9.1
- audioread == 3.0.1
- setuptools == 75.1.0
- tensorflow-gpu == 2.10.0
- tqdm == 4.63.0

## Citation

The model referenced the MobileNetV2 baseline system of the [DCASE 2022 Challenge Task 2](https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring), and if this baseline model is used, the following three papers must be cited according to the official rules:

- Kota Dohi, Keisuke Imoto, Noboru Harada, Daisuke Niizumi, Yuma Koizumi, Tomoya Nishida, Harsh Purohit, Takashi Endo, Masaaki Yamamoto, and Yohei Kawaguchi. Description and discussion on DCASE 2022 challenge task 2: unsupervised anomalous sound detection for machine condition monitoring applying domain generalization techniques. In arXiv e-prints: 2206.05876, 2022. [URL](https://arxiv.org/abs/2206.05876)
- Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, Shoichiro Saito, "ToyADMOS2: Another Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection under Domain Shift Conditions," in arXiv e-prints: 2106.02369, 2021. [URL](https://arxiv.org/abs/2106.02369)
- Kota Dohi, Tomoya Nishida, Harsh Purohit, Ryo Tanabe, Takashi Endo, Masaaki Yamamoto, Yuki Nikaido, and Yohei Kawaguchi. MIMII DG: sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task. In arXiv e-prints: 2205.13879, 2022. [URL](https://arxiv.org/abs/2205.13879)
