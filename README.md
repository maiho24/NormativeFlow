# BrainNormativeRealNVP

A Python package for normative modeling of brain imaging data using conditional RealNVP ([Dinh et al., 2017](https://openreview.net/forum?id=HkpbnH9lx)).

## Overview

BrainNormativeRealNVP is a comprehensive toolkit for building and applying normative models to neuroimaging data. It implements a conditional RealNVP approach that can:
- Learn normative patterns from brain imaging data conditioned on demographic and clinical variables
- Perform robust statistical inference using multiple scoring methods
- Quantify abnormality through multiple complementary approaches:
  - Exact conditional density estimation (direct probability calculation)
  - Percentile-based scoring for individual measurements, using cumulative distribution function (CDF)
  - Monte Carlo sampling for distribution characterisation (mean, median, std, etc.)
  - Traditional and robust z-score calculations
  
## Key Features

- **Adaptive Transformation Layer** (Optional): The model includes a learnable pre-processing transformation layer that automatically handles non-Gaussian, skewed, and heavy-tailed distributions in your data.
- **Conditional Coupling Layers**: A series of coupling layers that learn complex mappings between your data and a standard normal distribution, conditioned on covariates

## Installation

Using Conda helps manage dependencies and ensures compatibility across different systems. We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to keep your environment lightweight.

1. Create and activate a new conda environment:
```bash
conda create -n brain_nvp python=3.9
conda activate brain_nvp
```

2. Clone and install the package:
```bash
git clone https://github.com/maiho24/BrainNormativeRealNVP.git
cd BrainNormativeRealNVP
pip install -e .
```

**Note**: Make sure to always activate the environment before using the package:
```bash
conda activate brain_nvp
```
## Quick Start

### Training a Model

```bash
# Show help message and available options
brain-nvp-train --help

# Direct training with specified parameters
brain-nvp-train \
    --config configs/direct_config.yaml \
    --mode direct \
    --gpu

# Hyperparameter optimisation with Optuna
brain-nvp-train \
    --config configs/optuna_config.yaml \
    --mode optuna \
    --gpu
```

### Running Inference

```bash
# Show help message and available options
brain-nvp-inference --help

# Basic density calculation and percentiles-based scoring
brain-nvp-inference \
    --model_path models/model.pkl \
    --data_dir data/ \
    --output_dir results/ \
    --exact_density \
    --percentile_scores \
    --num_samples 1000

# Z-score derived from distribution metrics
brain-nvp-inference \
    --model_path models/model.pkl \
    --data_dir data/ \
    --output_dir results/ \
    --distribution_metrics \
    --num_samples 1000 \
    --zscore_method both

# Comprehensive analysis
brain-nvp-inference \
    --model_path models/model.pkl \
    --data_dir data/ \
    --output_dir results/ \
    --exact_density \
    --distribution_metrics \
    --num_samples 1000 \
    --zscore_method both \
    --percentile_scores
```

## Data Format

### Required Files
For training, these files are required:
- `train_data.csv`: Training data features
- `train_covariates.csv`: Training data covariates

For inference, the following files should be present in the data directory:
- `test_data.csv`: Test data features
- `test_covariates.csv`: Test data covariates

### Covariate Structure
The following covariates are expected for the current implementation:
- Age (continuous)
- wb_EstimatedTotalIntraCranial_Vol (continuous)
- Sex (categorical)
- Diabetes Status (categorical)
- Smoking Status (categorical)
- Hypercholesterolemia Status (categorical)
- Obesity Status (categorical)
- Alcohol Intake Status (categorical)
- Scanner (categorical)

To apply the package to other covariates, modifications are required in the `process_covariates()` function located in `utils/data.py`.

## Training Configuration
Create a YAML configuration file with the following structure for training:

### Direct Training Configuration

```yaml
model:
  hidden_dim: "128_64_128"                      # Architecture for hidden layers of the coupling networks
  covariate_net_dim: "32_64_128"                # Architecture for covariate networks used in each coupling layer
  n_layers: 6                                   # Number of coupling layers
  learning_rate: 0.001                          # Control how large of a step to take when updating weights during gradient descent
  weight_decay: 0.00001                         # Control the complexity of the model (Optional, Default: 1e-5)
  use_transformation_layer: True                # Enable or disable transformation layer (Optional, Default: False)
  conditional_transform: True                   # Make transformation depend on covariates (Optional, Default: False)
  transform_hidden_dims: "64_128"               # Architecture for hidden layers of the covariate-conditioned transformation networks (Optional)

training:
  epochs: 300
  batch_size: 64
  early_stopping_patience: 25
  validation_split: 0.2
  extract_transform_params: True                # Extract the transformation layer's parameters

paths:
  data_dir: "/path/to/data/"
  output_dir: "/path/to/output/"
```

### Optuna Configuration for Hyperparameter Optimisation

```yaml
training:
  epochs: 200
  early_stopping_patience: 20
  validation_split: 0.2

optuna:
  study_name: "realnvp_optimisation"
  n_trials: 100
  n_jobs: 8                                     # Run n_jobs in parallel
  use_transformation_layer: True                # Enable or disable transformation layer
  search_space:
    hidden_dim:
      type: "categorical"
      choices: [
        "256_128_256",
        "128_64_129",
      ]
    covariate_net_dim:
      type: "categorical"
      choices: [
        "32_64_128",
        "16_32",
      ]    
    n_layers:
      type: "int"
      min: 4
      max: 10
    learning_rate:                              # Control how large of a step to take when updating weights during gradient descent
      type: "loguniform"
      min: 1e-4
      max: 1e-2
    weight_decay:                               # Regularisation technique that adds a penalty proportional to weight magnitude to the loss function
      type: "loguniform"
      min: 1e-3
      max: 1e-6    
    batch_size:
      type: "categorical"
      choices: [32, 64, 128]
    conditional_transform:
            choices: [True, False]              # Try both conditional and non-conditional transforms
    transform_hidden_dims:
        choices: ["64_128", "64_128_256"]       # Architecture for the hidden layers of the covariate networks used for the transformation layer
paths:
  data_dir: "/path/to/data/"
  output_dir: "/path/to/output/"
```

## Inference Options
The package provides multiple methods for analyzing brain imaging data:

1. **Exact Density Calculation**:
    - Calculate exact conditional density directly from the model (how likely an observation is given its covariates)
    - Flag: `--exact_density`
    - Output: `conditional_density.csv` (overall log probability), `feature_logprob.csv` (log probability per feature)
    - This method quantifies how "normal" or "abnormal" a brain scan is compared to the expected distribution

2. **Distribution Metrics via Monte Carlo Sampling**:
    - Generate many samples for each covariate combination and compute summary statistics
    - Flag: `--distribution_metrics`
    - Requires: `--num_samples` (e.g., 1000 samples per covariate combination)
    - Outputs:
        - `means.csv` - Average predicted values for each feature
        - `medians.csv` - Median predicted values
        - `stds.csv` - Standard deviations
        - `iqrs.csv` - Interquartile ranges (75th percentile - 25th percentile)
        - `mads.csv` - Median absolute deviations
        - `quantiles_05.csv`, `quantiles_95.csv` - 5th and 95th percentiles
        - `quantiles_25.csv`, `quantiles_75.csv` - 25th and 75th percentiles
        - `skewness.csv`, `kurtosis.csv` - Distribution shape metrics

3. **Deviation Scoring Methods**:
    - Traditional Z-scores derived through distribution metrics:
        - Normal z-scores: (observed-mean)/std - assumes normal distribution
        - Robust z-scores: (observed-median)/MAD - less sensitive to outliers
        - Flag: `--zscore_method` [normal|robust|both]
        - Output: `normal_zscore.csv` and/or `robust_zscores.csv`
    - Percentile-based scores:
        - Uses the cumulative distribution function (CDF) to determine where the observed value falls within the distribution
        - This method answers: "Where the actual value ranks among the generated values?"
        - Flag: `--percentile_scores`
        - Requires: `--num_samples` (number of samples to generate per subject)
        - Output:
            - `percentile_CDF.csv` - Percentile scores from 0-100% for each brain measurement
            - `percentile_zscores.csv` - Same information converted to z-score format

## Output Directory Structure

The package organizes all outputs in a consistent directory structure:

```
output_dir/
├── logs/                               # Logging directory
│   ├── training_YYYYMMDD_HHMMSS.log    # Training process logs
│   └── inference_YYYYMMDD_HHMMSS.log   # Inference logs
│
├── models/                             # Model directory
│   ├── training_config.yaml            # Training configuration parameters
│   ├── final_model.pkl                 # Saved trained model
│   ├── Losses_training_validation.png  # Loss plot during training
│   └── best_params.yaml                # Best hyperparameters (Optuna mode only)
│
└── results/                            # Analysis results directory
    ├── inference_config.yaml           # Inference configuration
    ├── conditional_density.csv         # Overall log probability for each sample
    ├── feature_logprob.csv             # Feature-wise log probabilities
    ├── means.csv                       # Mean predictions from Monte Carlo sampling
    ├── medians.csv                     # Median predictions
    ├── stds.csv                        # Standard deviations
    ├── iqrs.csv                        # Interquartile ranges
    ├── mads.csv                        # Median absolute deviations
    ├── quartiles_25.csv                # Percentiles 25th
    ├── quartiles_75.csv                # Percentiles 75th
    ├── quartiles_05.csv                # Percentiles 5th
    ├── quartiles_95.csv                # Percentiles 95th
    ├── normal_zscores.csv              # Traditional z-scores
    ├── robust_zscores.csv              # Robust z-scores
    ├── percentile_CDF.csv              # Percentile scores
    |__ percentile_zscores.csv          # Z-scores derived from percentiles
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Attribution

If you use this package, please cite:

```bibtex
@software{Ho_BrainNormativeRealNVP,
  author = {Ho M., Song Y., Sachdev P., Fan L., Jiang J., Wen W.},
  title = {Normative Modeling of High-Dimensional Neuroimaging Data Through Conditional Normalizing Flows},
  year = {2025},
  url = {https://github.com/maiho24/BrainNormativeRealNVP}
}
```