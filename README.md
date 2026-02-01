# nanoGPT Hyperparameter Tuning Project

A comprehensive multi-team hyperparameter tuning study of [nanoGPT](https://github.com/karpathy/nanoGPT), an educational implementation of a GPT-style language model trained on Shakespeare character-level data. This project systematically explores how different hyperparameter configurations affect model performance across multiple team contributions.

## Overview

This project evaluates the impact of various hyperparameter combinations on nanoGPT's performance using the Shakespeare character-level dataset. Four experimental configurations of Vijay's system (Shallow-Small, Shallow-Large, Deep-Small, Deep-Large) each conducted multiple experiments with different fixed hyperparameter constraints, collectively exploring a comprehensive hyperparameter space.

## Project Structure

```
nanoGPT_repo/
├── README.md                                    # This file
├── nanoGPT_vijay_shallow_small.ipynb            # Vijay (Shallow-Small) training notebook
├── nanoGPT_vijay_shallow_large.ipynb            # Vijay (Shallow-Large) training notebook
├── nanoGPT_vijay_deep_small.ipynb               # Vijay (Deep-Small) training notebook
├── nanoGPT_vijay_deep_large.ipynb               # Vijay (Deep-Large) training notebook
├── nanoGPT_visualization.ipynb                  # Analysis & visualization notebook
├── nanoGPT_results_vijay_shallow_small/         # Vijay (Shallow-Small) results directory
│   ├── results.csv                              # Experiment results summary
│   ├── b64_L4_H4_E128_BS8_MI1000_D10_s1/        # Experiment checkpoint
│   ├── b64_L4_H4_E128_BS8_MI1000_D20_s2/        # ...
│   └── samples/                                 # Text samples from trained models
├── nanoGPT_results_vijay_shallow_large/         # Vijay (Shallow-Large) results directory
│   ├── results.csv
│   ├── [32 experiment folders]
│   └── samples/
├── nanoGPT_results_vijay_deep_small/            # Vijay (Deep-Small) results directory
│   ├── results.csv
│   ├── [32 experiment folders]
│   └── samples/
└── nanoGPT_results_vijay_deep_large/            # Vijay (Deep-Large) results directory
    ├── results.csv
    ├── [3+ experiment folders]
    └── samples/
```

## Hyperparameter Exploration Strategy

Each team member was assigned a specific block size and number of layers, then conducted a grid search over the remaining hyperparameters.

### Team Assignments

| Team Member           | Block Size | N Layers | Experiments | ID Range |
|----------------------|------------|----------|-------------|----------|
| Vijay (Shallow-Small) | 64         | 4        | 32          | 1-32     |
| Vijay (Deep-Small)    | 64         | 6        | 32          | 33-64    |
| Vijay (Shallow-Large) | 128        | 4        | 32          | 65-96    |
| Vijay (Deep-Large)    | 128        | 6        | 3+          | 97+      |

### Hyperparameter Grid

Each team explored the following grid of hyperparameters (except for their fixed block size and n_layers):

- **N Heads**: [4, 8]
- **N Embeddings**: [128, 256]
- **Batch Size**: [8, 16]
- **Max Iterations**: [1000, 2000]
- **Dropout**: [0.1, 0.2]

This produces 2 × 2 × 2 × 2 × 2 = **32 experiment combinations per team member** (where applicable).

## Experiment Configuration

Each experiment is named using the pattern:
```
b{block_size}_L{n_layers}_H{n_heads}_E{n_embeddings}_BS{batch_size}_MI{max_iters}_D{dropout}_s{seed}
```

**Example**: `b128_L4_H4_E128_BS8_MI1000_D10_s1`
- Block size: 128
- Layers: 4
- Heads: 4
- Embeddings: 128
- Batch size: 8
- Max iterations: 1000
- Dropout: 0.10
- Seed: 1

## Results Format

Each team's results are stored in a CSV file with the following columns:

| Column       | Description |
|--------------|-------------|
| Experiment   | Experiment identifier (naming convention above) |
| Train Loss   | Final training loss value |
| Val Loss     | Final validation loss value |
| Loss Gap     | Difference between validation and training loss (overfitting metric) |
| Total Params | Total number of model parameters |
| Config Path  | Path to the training configuration file |

### Example Results Entry
```csv
b128_L4_H4_E128_BS8_MI1000_D10_s1,2.2018,2.2313,0.0295,800000.0,/content/nanoGPT/b128_L4_H4_E128_BS8_MI1000_D10_s1.py
```

## Notebooks

### Individual Training Notebooks
- **nanoGPT_vijay_shallow_small.ipynb** - Vijay (Shallow-Small) training pipeline
  - Sets up Google Colab environment
  - Clones nanoGPT repository
  - Prepares Shakespeare dataset
  - Runs 32 experiments with block_size=64, n_layers=4
  - Generates text samples from trained models
  
- **nanoGPT_vijay_shallow_large.ipynb** - Vijay (Shallow-Large) training pipeline
  - Similar structure to Vijay (Shallow-Small) notebook
  - Runs experiments with block_size=128, n_layers=4
  
- **nanoGPT_vijay_deep_small.ipynb** - Vijay (Deep-Small) training pipeline
  - Similar structure
  - Runs experiments with block_size=64, n_layers=6
  
- **nanoGPT_vijay_deep_large.ipynb** - Vijay (Deep-Large) training pipeline
  - Similar structure
  - Runs experiments with block_size=128, n_layers=6

### Shared Analysis Notebook
- **nanoGPT_visualization.ipynb** - Comprehensive analysis and visualization
  - Loads results from all team members
  - Creates loss curves (training vs validation)
  - Analyzes overfitting patterns (loss gap)
  - Identifies best performing experiments
  - Generates comparison visualizations across teams
  - Ranks experiments by validation loss and generalization

## Key Findings

The results are analyzed across multiple dimensions:

1. **Loss Performance**: Identification of hyperparameter combinations that minimize training and validation loss
2. **Generalization**: Analysis of loss gap (validation - training) to identify overfitting patterns
3. **Model Complexity**: Comparison of parameter counts across different embedding dimensions
4. **Training Stability**: Evaluation of how different hyperparameters affect convergence

## Generated Artifacts

### Experiment Checkpoints
- **ckpt.pt**: Saved model checkpoint after training completion
- Located in each experiment folder: `nanoGPT_results_{member}/{experiment_name}/`
- Can be used for inference or further fine-tuning

### Text Samples
- **samples/**: Directory containing generated text samples from trained models
- Generated using the `sample.py` script from nanoGPT
- Provides qualitative assessment of model quality across configurations

## Running Experiments

Each notebook follows this workflow:

1. **Environment Setup**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !git clone https://github.com/karpathy/nanoGPT.git
   !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Data Preparation**
   - Shakespeare character-level dataset is prepared from nanoGPT's data directory

3. **Experiment Execution**
   ```python
   !python run_experiments.py {member_id}
   ```

4. **Sample Generation**
   - Text samples generated from checkpoints and saved to `samples/` directory

## Dataset

- **Source**: Shakespeare character-level dataset (included with nanoGPT)
- **Training Split**: Used for computing training loss
- **Validation Split**: Used for computing validation loss and identifying best models
- **Task**: Character-level language modeling

## Configuration Template

Experiments are configured using the following template:

```python
out_dir = "{save_dir}/{exp_name}"
dataset = "shakespeare_char"
eval_interval = 200
log_interval = 10
always_save_checkpoint = True

batch_size = {batch_size}
block_size = {block_size}
n_layer = {n_layer}
n_head = {n_head}
n_embd = {n_embd}
dropout = {dropout}

learning_rate = 3e-4
max_iters = {max_iters}
lr_decay_iters = {max_iters}

seed = {seed}
device = "cuda"  # or "cpu"

num_workers = 0
compile = False
```

## Analysis Metrics

1. **Train Loss**: Loss on training data at the end of training
2. **Val Loss**: Loss on validation data at the end of training
3. **Loss Gap**: Val Loss - Train Loss (higher values indicate overfitting)
4. **Total Params**: Number of trainable parameters in the model
5. **Model Quality**: Assessed qualitatively through generated text samples

## Dependencies

- **PyTorch**: GPU-accelerated deep learning framework
- **Python**: 3.7+
- **Jupyter**: For notebook execution
- **Pandas**: Data analysis and CSV handling
- **Matplotlib**: Visualization
- **NumPy**: Numerical computations
- **tqdm**: Progress bars
- **nanoGPT**: Educational GPT implementation (cloned in notebooks)

## How to Use This Repository

1. **Review Results**
   - Open any `results.csv` file to see experiment metrics
   - View the visualization notebook to see comparative analysis

2. **Analyze Performance**
   - Use pandas to load and filter results by hyperparameter values
   - Identify patterns between configurations and performance

3. **Generate New Samples**
   - Load any checkpoint file with `torch.load('ckpt.pt')`
   - Use nanoGPT's `sample.py` script to generate text

4. **Reproduce Experiments**
   - Run any training notebook with your own hyperparameter choices
   - Modify the configuration template as needed

## Team Members

- **Vijay (Shallow-Small)** - block_size=64, n_layers=4 experiments (32 configs)
- **Vijay (Shallow-Large)** - block_size=128, n_layers=4 experiments (32 configs)
- **Vijay (Deep-Small)** - block_size=64, n_layers=6 experiments (32 configs)
- **Vijay (Deep-Large)** - block_size=128, n_layers=6 experiments (3+ configs)

## References

- **nanoGPT**: https://github.com/karpathy/nanoGPT
- **Original Paper**: Andrej Karpathy's educational GPT implementation
- **Dataset**: Shakespeare character-level text corpus

## Notes

- All experiments were conducted on Google Colab with GPU acceleration
- Results are saved in `/content/drive/MyDrive/nanoGPT_results` during training
- Text samples provide qualitative assessment of model quality
- The loss gap metric (Val Loss - Train Loss) is key for identifying overfitting
- Higher embedding dimensions generally require more training time and memory

---

**Last Updated**: February 2026  
**Total Experiments**: 100+ (32 + 32 + 32 + 3+ across team members)
