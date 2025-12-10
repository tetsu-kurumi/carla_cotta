# CoTTA for Semantic Segmentation in CARLA

This repository contains code for evaluating test-time adaptation methods (CoTTA and TTDA/Tent) for semantic segmentation in autonomous driving using the CARLA simulator. Our experiments evaluate these methods under continual domain shifts caused by varying weather and lighting conditions.

## Overview

We compare three approaches:
- **Static**: No adaptation (baseline)
- **TTDA (Tent)**: Test-Time Domain Adaptation using entropy minimization
- **CoTTA**: Continual Test-Time Adaptation with teacher-student architecture

All methods are evaluated on a SegFormer-B3 model trained on CARLA Town03 data.

## Prerequisites

### System Requirements
- Ubuntu 20.04 or 22.04 (recommended)
- NVIDIA GPU with CUDA support (minimum 8GB VRAM)
- Python 3.8+
- At least 50GB free disk space

### Dependencies

Install PyTorch and other dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers opencv-python h5py tqdm tensorboard numpy
```

### CARLA Simulator Setup

1. Install CARLA 0.9.16:
```bash
chmod +x INSTALL_CARLA.sh
./INSTALL_CARLA.sh
```

2. Or manually download CARLA 0.9.16 from the [official releases](https://github.com/carla-simulator/carla/releases/tag/0.9.16) and extract it to your preferred location.

3. Start CARLA server:
```bash
cd /path/to/CARLA_0.9.16
./CarlaUE4.sh -quality-level=Low -RenderOffScreen
```

## Replicating Our Experiments

You can either use our pre-collected data and trained models, or collect/train your own.

### Option 1: Using Provided Data and Models (Recommended)

We provide pre-collected datasets and trained model checkpoints:
- **Training data**: ~8,000 frames across 8 weather conditions (HDF5 format, ~5GB)
- **Evaluation data**: ~18,500 frames across 5 evaluation scenarios (HDF5 format, ~15GB)
- **Model checkpoint**: SegFormer-B3 trained on CARLA (.pth file, ~200MB)

**Download link**: [Link to Download](https://www.dropbox.com/scl/fo/zbk6umjpom6ko3beniat1/AGpWTy65ACD_qYISzAAX-Z0?rlkey=dm3poktd0gwhu62qanrhjcmap&st=rm739e65&dl=0)

After downloading:
1. Extract data to `./data/` directory
2. Place model checkpoint in `./checkpoints/` directory
3. Skip to [Running Evaluation](#running-evaluation)

### Option 2: Collecting Your Own Data

#### Step 1: Start CARLA Server

```bash
cd /path/to/CARLA_0.9.16
./CarlaUE4.sh -quality-level=Low -RenderOffScreen
```

#### Step 2: Collect Training Data

```bash
python collect_training_data.py \
    --output-dir ./data/training \
    --frames-per-weather 1000 \
    --map Town03 \
    --random-seed 12345
```

This will collect ~8,000 frames (1,000 per weather condition) in HDF5 format.

**Weather conditions used:**
- clear_noon, clear_sunset
- cloudy_noon, cloudy_sunset
- wet_noon, wet_sunset
- soft_rain_noon, wet_cloudy_noon

**Parameters:**
- Camera resolution: 800x600
- Field of view: 110 degrees
- Frame rate: 20 FPS (synchronous mode)
- Random seed: 12345 (for reproducibility)

#### Step 3: Collect Evaluation Data

```bash
python collect_eval_data.py \
    --output-dir ./data/evaluation \
    --map Town03 \
    --random-seed 54321
```

This collects data for 5 evaluation scenarios:
1. **Weather Progression** (2,001 frames): clear_noon → cloudy → light_rain → heavy_rain → fog
2. **Time Progression** (1,501 frames): clear_noon → sunset → dusk → night
3. **Combined** (2,001 frames): clear_noon → cloudy → light_rain → sunset → dusk → night
4. **Cyclic** (11,701 frames): 10 cycles of clear_noon → heavy_rain → fog → clear_noon
5. **Stress Test** (1,201 frames): Rapid shifts between clear_noon and challenging conditions

**Note:** Evaluation uses a different random seed (54321) to ensure distinct images from training data.

### Option 3: Training Your Own Model

If you collected your own training data or want to retrain:

#### Train SegFormer-B3

```bash
python train_SFB3_preload.py \
    --data-dir ./data/training \
    --output-dir ./checkpoints \
    --batch-size 2 \
    --epochs 50 \
    --learning-rate 6e-5 \
    --num-classes 29
```

**Training details:**
- Model: SegFormer-B3 (~47M parameters)
- Pre-trained weights: ADE20K (150 classes)
- Input resolution: 480x640
- Optimizer: AdamW with polynomial learning rate decay
- Loss: Cross-entropy

The best model (based on validation mIoU) will be saved as `best_model.pth`.

#### Alternative: Train Fast-SCNN

For faster training with a lighter model:

```bash
python train_scnn_preload.py \
    --data-dir ./data/training \
    --output-dir ./checkpoints \
    --batch-size 8 \
    --epochs 50 \
    --learning-rate 1e-3 \
    --num-classes 29
```

Fast-SCNN is faster but typically achieves lower accuracy than SegFormer-B3.

## Running Evaluation

### Run All Scenarios

```bash
python run_evaluation.py \
    --checkpoint ./checkpoints/best_model.pth \
    --data-dir ./data/evaluation \
    --output-dir ./results \
    --model-type segformer_b3 \
    --scenarios all
```

This will evaluate all three methods (Static, TTDA, CoTTA) across all five scenarios.

**Evaluation parameters:**
- Batch size: 1 (online adaptation)
- TTDA learning rate: 1e-5
- CoTTA learning rate: 1e-5
- CoTTA restoration probability: 0.1
- CoTTA teacher momentum: 0.999

### Run Specific Scenarios

To run individual scenarios:

```bash
# Weather progression only
python run_evaluation.py \
    --checkpoint ./checkpoints/best_model.pth \
    --data-dir ./data/evaluation \
    --output-dir ./results \
    --model-type segformer_b3 \
    --scenarios weather

# Time progression only
python run_evaluation.py \
    --checkpoint ./checkpoints/best_model.pth \
    --data-dir ./data/evaluation \
    --output-dir ./results \
    --model-type segformer_b3 \
    --scenarios time

# Available scenarios: weather, time, combined, cyclic, stress, all
```

### Expected Runtime

- Weather Progression: ~15-20 minutes
- Time Progression: ~10-15 minutes
- Combined: ~15-20 minutes
- Cyclic (10 cycles): ~90-120 minutes
- Stress Test: ~8-12 minutes

**Total time for all scenarios: ~2.5-3 hours**

## Analyzing Results

After evaluation completes, analyze the results:

```bash
python analyze_all_results.py \
    --results-dir ./results \
    --output-file ./results/summary.txt
```

This will generate:
- Per-scenario mIoU comparisons
- Variance/stability metrics
- Source domain retention analysis
- Visualization plots (if matplotlib is installed)

### Understanding the Output

The analysis reports three key metrics:

1. **Mean IoU (mIoU)**: Average intersection-over-union across all semantic classes
   - Higher is better
   - Range: 0.0 to 1.0

2. **Standard Deviation**: Prediction stability across frames
   - Lower indicates more stable predictions
   - CoTTA aims to reduce variance compared to TTDA

3. **Source Domain Retention**: Performance on clear_noon (source domain) at the end of evaluation
   - Measures catastrophic forgetting
   - Reported as percentage of initial performance

### Expected Results

Based on our experiments, you should observe:

- **Static baseline**: mIoU ~0.314 (average across scenarios)
- **TTDA (Tent)**: mIoU ~0.295 (slight degradation, -6.5%)
- **CoTTA**: mIoU ~0.279 (degradation, -11.2%)

**Key finding**: Unlike the original CoTTA paper on image classification, test-time adaptation does not improve performance for semantic segmentation in CARLA. However, CoTTA does show ~22% lower variance than TTDA, indicating more stable predictions.

