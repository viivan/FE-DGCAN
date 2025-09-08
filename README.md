# FE-DGCAN

此模型基于 MobileNetV2 的[DCASE2022 挑战任务 2](https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring)
的基准系统上添加修改

## 描述

- `00_train.py`
  - "预训练模式":
    - 该脚本通过使用目录 `dev_data/<machine_type>/train/` 来预训练每种机器类型的模型。
  - "无监督对比分类学习模式":
    - 该脚本通过使用目录 `eval_data/<machine_type>/train/` 来分类学习每种机器类型的模型。
- `01_test.py`
  - "训练数据集检测结果":
    - 该脚本为每个部分生成一个 csv 文件，其中包含 `dev_data/<machine_type>/test/` 目录下每个 wav 文件的异常得分。
    - csv 文件将存储在 `result/` 目录中。
    - 它还会生成一个 csv 文件，包括每个部分的 AUC、pAUC、精确率、召回率和 F1 分数。
  - "测试数据集检测结果":
    - 该脚本为每个部分生成一个 csv 文件，其中包含 `eval_data/<machine_type>/test/` 目录下每个 wav 文件的异常得分。
    - csv 文件将存储在 `result/` 目录中。

## 使用方法

### 1. 克隆代码仓库

从 Github 克隆此代码仓库。

### 2. 下载数据集

下载数据集：

- "开发数据集"
  - 从 [https://zenodo.org/record/6355122](https://zenodo.org/record/6355122) 下载 `dev_data_<machine_type>.zip`。
- "附加训练数据集"，即用于训练的评估数据集
- "评估数据集"，即用于测试的评估数据集

### 3. 解压数据集

解压下载的文件，并按以下目录结构组织：

- ./dcase2022_task2_baseline_mobile_net_v2
  - /00_train.py
  - /01_test.py
  - /common.py
  - /dgcan.py
  - /feature_enhancement.py
  - /train_process.py
  - /keras_model.py
  - /baseline.yaml
  - /readme.md
- /dev_data
  - /fan
    - /train (仅正常片段)
      - /section*00_source_train_normal_0000*<attribute>.wav
      - ...
      - /section*00_target_train_normal_0009*<attribute>.wav
    - /test
      - /section*00_source_test_normal_0000*<attribute>.wav
      - ...
      - /section*00_source_test_anomaly_0049*<attribute>.wav
    - attributes_00.csv (section 00 的属性 csv)
  - /gearbox (其他机器类型具有与 fan 相同的目录结构)
  - /bearing
  - /slider (`slider`表示“滑轨”)
  - /ToyCar
  - /ToyTrain
  - /valve
- /eval_data
  - /fan
    - /train
      - /section*03_source_train_normal_0000*<attribute>.wav
      - ...
    - /test
      - /section_03_test_0000.wav
      - ...
    - attributes_03.csv (训练数据的属性 csv)
  - /gearbox (其他机器类型具有与 fan 相同的目录结构)
  - /bearing
  - /slider (`slider`表示“滑轨”)
  - /ToyCar
  - /ToyTrain
  - /valve

### 4. 更改参数

通过编辑 `baseline.yaml` 来更改特征提取和模型定义的参数。

### 5. 运行训练脚本（用于开发数据集）

运行训练脚本 `00_train.py`。  
使用选项 `-d` 来指定开发数据集 `dev_data/<machine_type>/train/`。

选项：

| 参数 | 描述        |
| ---- | ----------- | -------------------------- |
| `-h` | `--help`    | 应用程序帮助。             |
| `-v` | `--version` | 显示应用程序版本。         |
| `-d` | `--dev`     | 开发数据集模式。           |
| `-e` | `--eval`    | 附加训练和评估数据集模式。 |

`00_train.py` 为每种机器类型训练一个模型，并将训练后的模型存储在 `model/` 目录中。

### 6. 运行测试脚本（用于开发数据集）

运行测试脚本 `01_test.py`。  
使用选项 `-d` 来指定开发数据集 `dev_data/<machine_type>/test/`。

`01_test.py` 的选项与 `00_train.py` 相同。  
`01_test.py` 计算 `dev_data/<machine_type>/source_test/` 和 `dev_data/<machine_type>/target_test/` 目录下每个 wav 文件的异常得分。  
每个部分的 csv 文件（包含异常得分）将存储在 `result/` 目录中。  
如果模式为“预训练模式”，该脚本会输出对应 csv 文件，其中包含每个部分的 AUC、pAUC、精确率、召回率和 F1 分数。

### 7. 检查结果

您可以在 `result/` 目录中的 csv 文件 `anomaly_score_<machine_type>_section_<section_index>_test.csv` 中查看异常得分。  
每个异常分数对应 `dev_data/<machine_type>/test/` 目录中的一个 wav 文件。

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

阈值处理后的异常检测结果可以在 csv 文件 `decision_result_<machine_type>_section_<section_index>_test.csv` 中查看：

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

此外，您还可以检查 AUC、pAUC、精确率、召回率和 F1 分数等性能指标：

`result.csv`

```
fan
                AUC(source)  AUC(target)   pAUC  precision(source)  precision(target) recall(source) recall(target) F1 score(source)  F1 score(target)
00                 0.858        0.622     0.525       1.0              0.625            0.15           0.15           0.260              0.241
01                 0.749        0.362     0.512       0.939            0.62             0.31           0.31           0.466              0.413
02                 0.691        0.577     0.653       0.863            0.686            0.57           0.57           0.686              0.622
arithmetic mean    0.766        0.520     0.563       0.934            0.643            0.343          0.343          0.471              0.426
harmonic mean      0.760        0.490     0.556       0.930            0.642            0.257          0.257          0.403              0.3677

  ...
valve
                AUC(source)  AUC(target)   pAUC  precision(source)  precision(target) recall(source) recall(target) F1 score(source)  F1 score(target)
00                 0.786        0.475     0.568       0.928            0.812            0.13           0.13           0.228              0.224
01                 0.563        0.637     0.544       0.833            0.833            0.2            0.2            0.322              0.322
02                 0.768        0.806     0.857       1.0              1.0              0.73           0.73           0.843              0.843
arithmetic mean    0.706        0.639     0.657       0.920            0.881            0.353          0.353          0.464              0.463
harmonic mean      0.690        0.610     0.630       0.915            0.874            0.213          0.213          0.346              0.342

                                                                   AUC(source)  AUC(target)   pAUC   precision(source)  precision(target) recall(source)  recall(target) F1 score(source)  F1 score(target)
"arithmetic mean over all machine types  sections  and domains"     0.679          0.531      0.576     0.685               0.580            0.392            0.392          0.439             0.419
"harmonic mean over all machine types  sections  and domains"       0.641          0.477      0.563     0.00                0.00             0.0              0.0            0.0               0.0

```

## 依赖

### 软件包

- p7zip-full
- Python == 3.8.20
- FFmpeg

### Python 包

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


## 引用

如果使用了该基线模型，需要按照官网规则引用以下三篇论文：

- Kota Dohi, Keisuke Imoto, Noboru Harada, Daisuke Niizumi, Yuma Koizumi, Tomoya Nishida, Harsh Purohit, Takashi Endo, Masaaki Yamamoto, and Yohei Kawaguchi. Description and discussion on DCASE 2022 challenge task 2: unsupervised anomalous sound detection for machine condition monitoring applying domain generalization techniques. In arXiv e-prints: 2206.05876, 2022. [URL](https://arxiv.org/abs/2206.05876)
- Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, Shoichiro Saito, "ToyADMOS2: Another Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection under Domain Shift Conditions," in arXiv e-prints: 2106.02369, 2021. [URL](https://arxiv.org/abs/2106.02369)
- Kota Dohi, Tomoya Nishida, Harsh Purohit, Ryo Tanabe, Takashi Endo, Masaaki Yamamoto, Yuki Nikaido, and Yohei Kawaguchi. MIMII DG: sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task. In arXiv e-prints: 2205.13879, 2022. [URL](https://arxiv.org/abs/2205.13879)
