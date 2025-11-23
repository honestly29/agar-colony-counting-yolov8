![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Ultralytics YOLO](https://img.shields.io/badge/YOLOv8-Model-blue)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

# Colony Counting with YOLOv8

This project trains and evaluates a YOLOv8 model for detecting and counting bacterial colonies (CFUs) on Petri dish images. It uses the **AGAR (A microbial colony dataset for deep learning detection)** dataset and a custom preprocessing pipeline that converts the dataset into YOLO-compatible format and merges all bacterial species into a single “CFU” class.

The workflow includes dataset transformation, model training, fine-tuning, and evaluation.

## Project Structure

```
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── utils/
│   ├── config.py
├── train.py
├── fine_tune.py
├── evaluate_test.py
└── README.md
```

## Dataset Overview

This project uses the **AGAR dataset**:

Majchrowska, S., Pawłowski, J., Guła, G., Bonus, T., Hanas, A., Loch, A., Pawlak, A., Roszkowiak, J. and Drulis-Kawa, Z. (2021).  
**AGAR: A microbial colony dataset for deep learning detection.** arXiv:2108.01234.

Dataset website:  
https://agar.neurosys.com

Originally, AGAR is **not structured in YOLO format** and contains **multi-class species labels**, which must be transformed for CFU counting.

## Dataset Transformation Pipeline

The raw AGAR dataset is preprocessed through several steps to make it usable for YOLOv8 colony detection.

### 1. Convert JSON → YOLOv8 Label Files

Each AGAR annotation JSON file is parsed and converted to YOLO `.txt` format:

```
class x_center y_center width height
```

YOLO expects normalised coordinates, so all bounding boxes are scaled relative to the image dimensions.

### 2. Merge All Species into One Class (CFU)

AGAR contains several bacterial species.  
For colony counting, the species identity is not required.

Therefore, all species are merged into:

```
class 0: CFU
```

### 3. Remove Uncountable or Defective Plates

Some plates in AGAR are unsuitable for CFU detection, such as:

- Overgrown plates
- Contaminated plates
- Broken or incomplete dishes
- Plates with severe artefacts

These are removed to improve training quality.

### 4. Restructure Dataset into YOLO Format

The processed data is organised as:

```
data/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

## Data Configuration File (`data.yaml`)

```
train: images/train
val: images/val
test: images/test

names:
  0: CFU
```

## Installation

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install ultralytics
```

## Training the Model

```
python train.py
```

### Fine-tuning

```
python fine_tune.py
```

## Evaluation

```
python evaluate_test.py
```

The script reports metrics such as:

- MAE
- MSE
- MSLE
- sMAPE
- Precision/Recall
- Per-image colony count errors

## Configuration

`config.py` contains:

- Model choice
- Batch size
- Image size
- Learning rate
- Dataset paths
- Output directory

## Results

After training, YOLOv8 creates:

```
runs/detect/exp/
```

containing:

- Loss curves
- Precision–Recall curves
- Best model weights (`best.pt`)
- Example predictions on val/test sets

## Citation

```
@misc{majchrowska2021agar,
      title={AGAR a microbial colony dataset for deep learning detection},
      author={Sylwia Majchrowska and Jarosław Pawłowski and Grzegorz Guła and Tomasz Bonus and Agata Hanas and Adam Loch and Agnieszka Pawlak and Justyna Roszkowiak and Tomasz Golan and Zuzanna Drulis-Kawa},
      year={2021},
      eprint={2108.01234},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
