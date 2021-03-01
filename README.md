# AICore - Usecase 2

## Data Preparation

1. copy/move data into data_input
2. copy/move data into data_aux (e.g. prepared ArcticDEM data)

## Data Preprocessing

```bash
python setup_raw_data.py
```
```bash
python prepare_data.py
```

## Training a model

```bash
python train.py
```

## Running Inference

```bash
python inference.py logs/TrainedModel 20190727_160426_104e 20190709_042959_08_1057
```

## Configuration

Configuration is done via the `config.yml` file. Example config:

```yaml
# Model Specification
model:
  # Model Architecture. Available:
  # Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN]
  architecture: Unet
  # Model Encoder. Examples:
  # resnet18, resnet34, resnet50, resnet101, resnet152
  # Check https://github.com/qubvel/segmentation_models.pytorch#encoders for the full list of available encoders
  encoder: resnet34
  # Encoder weights to use (if transfer learning is desired)
  # `imagenet` is available for all encoders, some of them have more options available
  # `random` initializes the weights randomly
  # Check https://github.com/qubvel/segmentation_models.pytorch#encoders for the
  # full list of weights available for each encoder
  encoder_weights: random
# Loss Function to use. Available:
# JaccardLoss, DiceLoss, FocalLoss, LovaszLoss, SoftBCEWithLogitsLoss
loss_function: FocalLoss
# Data Configuration
data_threads: 4  # Number of threads for data loading
data_sources:  # Enabled input features
  - planet
  - ndvi
  - tcvis
  - relative_elevation
  - slope
datasets:
  train:
    augment: true
    augment_types:
      - "HorizontalFlip"
      - "VerticalFlip"
      - "Blur"
      - "RandomRotate90"
    shuffle: true
    scenes:
      - "20190618_201847_1035"
      - "20190618_201848_1035"
      - "20190623_200555_0e19"
  val:
    augment: false
    shuffle: false
    scenes:
      - "20190727_160426_104e"
  test:
    augment: false
    shuffle: false
    scenes:
      - "20190709_042959_08_1057"
# Training Parameters
batch_size: 4
learning_rate: 0.001
# Training Schedule
schedule:
  - phase: Training
    epochs: 30
    steps:
      - train_on: slump_tiles(train)
      - validate_on: val
      - log_images
# Visualization Configuration
visualization_tiles:
  20190727_160426_104e: [5, 52, 75, 87, 113, 139, 239, 270, 277, 291, 305]

```

