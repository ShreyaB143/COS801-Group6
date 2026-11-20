# Lightweight Crop Disease Diagnosis for Low-Cost Devices

COS801 Deep Learning Project (2026), Group 6

This repository contains the code and supporting material for a crop disease classification project focused on three practical deployment challenges:

1. **Efficiency:** compressing lightweight convolutional neural networks for low-cost devices.
2. **Robustness:** measuring and reducing the performance gap between controlled laboratory images and real-world field images.
3. **Reliability:** testing whether a model can abstain when an input is unfamiliar or out of distribution.

The experiments use **PlantVillage** as the laboratory domain and **PlantDoc** as the field domain. Only the 26 crop-condition classes shared by both datasets are included in the main classification task.

> **Important:** The datasets and large trained-model files are not stored in this repository. Follow the setup and data instructions below before running the notebook.


## Main experiments

The project includes:

- MobileNetV3-Small and EfficientNet-B0 baselines
- Local and global unstructured pruning
- Structured channel pruning
- Knowledge-distillation recovery after pruning
- INT8 post-training quantisation
- Mixed INT8/FP32 quantisation
- Quantisation-aware training
- Geometric, photometric and image-degradation augmentation
- Combined augmentation experiments
- Calibration and temperature scaling
- Out-of-distribution detection and selective prediction
- ONNX and TensorFlow Lite export
- Model size and CPU inference-time measurement


## Reproducibility design

The project uses the following controls to make results easier to reproduce:

- The shared task is fixed to 26 classes that occur in both PlantVillage and PlantDoc.
- Class indices are defined centrally in `shared_class_labels.py`.
- PlantVillage near-duplicates are grouped before splitting to reduce leakage between train, validation and test data.
- The PlantVillage split is fixed at 70% training, 15% validation and 15% testing.
- PlantDoc is used only as a field test set, not for training or model selection.
- Augmentation experiments use seeds `42`, `43` and `44`.
- The main input size is `224 x 224`.
- Images are normalised using the ImageNet mean and standard deviation.
- Macro-F1 is the primary metric because the class distribution is imbalanced.
- Accuracy, expected calibration error and per-class F1 are also reported.
- Test datasets are evaluated without augmentation.
- A baseline parity check is performed before compression comparisons.

## Data summary

| Split | Source | Images | Classes | Purpose |
|---|---:|---:|---:|---|
| `lab_train` | PlantVillage | 25,108 | 26 | Training |
| `lab_val` | PlantVillage | 5,385 | 26 | Model selection and calibration |
| `lab_test` | PlantVillage | 5,376 | 26 | Laboratory-domain evaluation |
| `field_test` | PlantDoc | 2,499 | 26 | Field-domain evaluation only |
| **Total** |  | **38,368** | **26** |  |

## Environment

### Recommended setup

The simplest option is to run the project in **Google Colab** with Google Drive mounted. A local Python environment can also be used.

Recommended environment:

- Python 3.10 or 3.11
- PyTorch and torchvision
- pandas
- NumPy
- scikit-learn
- Pillow
- matplotlib
- seaborn
- ONNX and ONNX Runtime
- TensorFlow, for TensorFlow Lite export

Exact package versions should be saved after the final successful run:

```bash
pip freeze > requirements.txt
```

Install the saved environment with:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the strongest reproducibility, also record the following in the experiment output:

- Python version
- PyTorch and torchvision versions
- CUDA and cuDNN versions, if a GPU is used
- GPU or CPU model
- operating system
- Git commit hash

## Dataset setup

###  Obtain the datasets

Download the following datasets from their official or approved distribution locations:

- **PlantVillage**, colour-image version
- **PlantDoc**, including its train and test folders

The datasets are not redistributed in this repository. Users are responsible for following the licences and terms of the original datasets.

The project report records PlantVillage as **CC BY-SA** and PlantDoc as **CC BY 4.0**. Confirm the current licence information at the original source before redistributing any data.

### Validate the 26 shared classes

Run:

```bash
python src/shared_class_labels.py
```

The script:

- maps PlantVillage and PlantDoc folder names to the same canonical labels;
- checks that all required folders exist;
- counts images by class and domain;
- reports excluded folders and thin classes; and
- exits with an error if a mapped class has no images on one side.

A successful run should end with:

```text
Checks passed.
```


## Running in Google Colab

###  Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Move to the repository

For a shared drive, adjust the path to match your own folder:

```python
%cd "/content/drive/Shareddrives/COS 801 Group 6/COS801-Group6"
```

###  Install dependencies

```python
!pip install -r requirements.txt
```


###  Run the notebook

Open the main notebook and select:

**Runtime > Run all**

The notebook should be run from top to bottom in a fresh runtime. Avoid executing later sections with stale variables from an earlier session.

## Pipeline order

The combined notebook follows this order:

1. Environment setup
2. Shared data-loading, evaluation, bootstrap and training functions
3. Split validation and local image-cache construction
4. Baseline scoring
5. Compression experiments
6. Augmentation experiments
7. Joint compression and augmentation analysis
8. ONNX and TensorFlow Lite export
9. Abstention handoff and final evaluation

Run the sections in this order because later experiments depend on baseline checkpoints, split files or outputs created earlier.

## Core experiment settings

The main augmentation sweep uses:

| Setting | Value |
|---|---|
| Model | MobileNetV3-Small |
| Initialisation | ImageNet weights |
| Output classes | 26 |
| Input size | 224 x 224 |
| Optimiser | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Learning-rate schedule | Cosine decay |
| Epochs | 10 |
| Batch size | 32 |
| Loss | Cross-entropy |
| Seeds | 42, 43 and 44 |
| Model selection | Best `lab_val` accuracy |
| Precision | FP32 |
| ECE bins | 15 equal-width bins |
| Bootstrap | 2,000 resamples |

The five augmentation arms are:

- `A0`: control augmentation
- `A1`: geometric augmentation
- `A2`: photometric augmentation
- `A3`: degradation augmentation
- `A4`: combined geometric, photometric and degradation augmentation


### Abstention deployment model

The selected deployment model used A4 combined augmentation and 40% global magnitude pruning. It retained laboratory behaviour but remained weak under field-domain shift. It performed better than the single-run baseline on non-plant rejection, but neither model reliably rejected unseen diseases on known crops.

These are validation targets, not guarantees of bit-for-bit identical results.

## Reproducing a clean run

Use this checklist for a full rerun:

- [ ] Clone the repository at a recorded commit.
- [ ] Create a fresh Python or Colab environment.
- [ ] Install `requirements.txt` without unresolved dependency errors.
- [ ] Place PlantVillage and PlantDoc in the expected folders.
- [ ] Confirm that the fixed split file is present.
- [ ] Run `shared_class_labels.py` and obtain `Checks passed.`
- [ ] Run `diagnose_paths.py` and resolve missing data.
- [ ] Confirm the expected split counts.
- [ ] Run the notebook from the first cell in a fresh runtime.
- [ ] Confirm the baseline parity check before compression.
- [ ] Run seeds 42, 43 and 44 for multi-seed experiments.
- [ ] Save metrics, checkpoints and figures to separate output folders.
- [ ] Record the environment and Git commit hash.
- [ ] Compare the final metrics with the reference values above.

## Outputs

A complete run should save, where applicable:

- baseline and compressed checkpoints;
- per-run metric files;
- aggregated metric tables;
- bootstrap confidence intervals;
- calibration results;
- per-class F1 results;
- pruning and quantisation summaries;
- ONNX and TensorFlow Lite models;
- model-size and CPU-latency measurements; and
- report figures.
