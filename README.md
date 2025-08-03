# MovSem: Movement-Semantics Contrastive Learning for Trajectory Similarity - AAAI 2026 Submission

## Key Features
- **Movement-Semantics Feature Extraction**: Converts raw GPS data into meaningful movement and semantic features
- **Hierarchical Representation Learning**: Utilizes intra-patch and inter-patch attention to capture multi-level trajectory information
- **Curvature-Guided Augmentation**: A novel data augmentation strategy that improves model robustness
- **Contrastive Learning**: Employs contrastive learning principles to learn discriminative trajectory embeddings
- **Fine-tuning Support**: Supports fine-tuning to approximate existing heuristic trajectory similarity measures

## Project Structure
```
SemMovCL/
├── config.py                    # Configuration settings
├── data/                        # Dataset storage (place datasets here)
├── exp/                         # Experiment logs and snapshots
│   ├── log/
│   └── snapshots/
├── model/                       # Core model definitions
│   ├── hier_sem_encoding.py     # Hierarchical Semantic Encoding layer
│   ├── moco.py                  # MoCo implementation
│   ├── node2vec_.py             # Node2Vec implementation
│   └── semmovcl.py              # Main SemMovCL model
├── task/                        # Specific tasks
│   └── trajsimi.py              # Trajectory similarity task
├── train.py                     # Main training script
├── train_trajsimi.py            # Fine-tuning script for trajectory similarity
└── utils/                       # Utility functions and data handling
    ├── cellspace.py             # Cell space definitions
    ├── data_loader.py           # Data loading utilities
    ├── edwp.py                  # Enhanced Dynamic Time Warping
    ├── preprocessing.py         # Trajectory preprocessing functions
    ├── rdp.py                   # Ramer-Douglas-Peucker algorithm
    ├── tool_funcs.py            # General utility functions
    └── traj.py                  # Trajectory related utilities
```

## Quick Start

### Dataset Download
As an example, we provide a preprocessed Germany dataset:
- **Germany Dataset (Preprocessed)**: [Download from Google Drive](https://drive.google.com/file/d/1J_HCICiutPlJbtDihoDk6_SdB8gRdC3y/view?usp=sharing)

Extract the downloaded dataset and place it in the `SemMovCL/data/` directory.

### Installation
1. **Setup project**:
   ```bash
   cd SemMovCL
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Training and Testing
#### Pre-training SemMovCL (using Germany dataset as example)
```bash
python train.py --dataset germany
```

#### Fine-tuning for Trajectory Similarity
To fine-tune the pre-trained SemMovCL to learn to approximate existing heuristic trajectory similarity measures:
```bash
python train_trajsimi.py --dataset germany --trajsimi_measure_fn_name hausdorff
```

### Configuration
You can modify `config.py` to adjust:
- Training parameters (learning rate, batch size, epochs)
- Model configurations (embedding dimensions, attention heads)
- Dataset paths and preprocessing options
- Device settings (CPU/GPU)

## Usage Examples
### Basic Training
```bash
# Train on Germany dataset with default settings
python train.py --dataset germany

# Train with custom parameters
python train.py --dataset germany --batch_size 64 --learning_rate 0.001
```

### Fine-tuning for Different Similarity Measures
```bash
# Fine-tune for Hausdorff distance
python train_trajsimi.py --dataset germany --trajsimi_measure_fn_name hausdorff
```

## Acknowledgments
We are deeply grateful to Dr. Yanchuan Chang for providing the dataset used in this research. Our code is built upon her TrajCL framework, which we consider to be the current state-of-the-art model for trajectory similarity calculation among published works.
