# ISER: Isolation-based Spherical Ensemble Representations for Anomaly Detection

A Python implementation of the ISER (Isolation-based Spherical Ensemble Representations) algorithm for unsupervised anomaly detection, based on the research paper published in PVLDB.

## Overview

ISER combines isolation principles with efficient local density estimation through hypersphere-based space partitioning while maintaining linear time and space complexity. The method constructs ensemble representations where hypersphere radii serve as proxies for local density: smaller radii indicate dense regions while larger radii correspond to sparse areas.

## Installation

### Requirements

```bash
pip install numpy scipy scikit-learn matplotlib pandas
```

### Optional Dependencies

For visualization and demo notebooks:
```bash
pip install jupyter tqdm pyod
```

## Quick Start

### Basic Usage

```python
from ISER import ISER
import numpy as np

# Generate sample data
X = np.random.randn(1000, 2)

# Initialize ISER
detector = ISER(max_samples=32, n_estimators=200, random_state=42)

# Fit the model
detector.fit(X)

# Get anomaly scores using different methods
scores_avg = detector.ISER_A(X)      # Average-based scoring
scores_sim = detector.ISER_S(X)      # Similarity-based scoring
scores_if = detector.ISER_IF(X)      # Enhanced Isolation Forest
```

## API Reference

### ISER Class

#### Constructor Parameters

- `max_samples` (int, default=16): Number of hyperspheres
- `n_estimators` (int, default=200): Number of ensemble partitionings
- `if_max_samples` (int, default=256): Samples for building Isolation Forest
- `random_state` (int, default=None): Random seed for reproducibility
- `novelty` (bool, default=False): Whether to use novelty detection mode

#### Methods

##### `fit(X)`
Fit the ISER model on training data.

**Parameters:**
- `X` (array-like): Training data of shape (n_samples, n_features)

**Returns:**
- `self`: Returns the fitted estimator

##### `transform(X)`
Transform data into ensemble representation space.

**Parameters:**
- `X` (array-like): Input data of shape (n_samples, n_features)

**Returns:**
- `array`: Ensemble representations of shape (n_samples, n_estimators)

##### `ISER_A(X)`
Compute anomaly scores using average-based scoring.

**Parameters:**
- `X` (array-like): Input data of shape (n_samples, n_features)

**Returns:**
- `array`: Anomaly scores (higher values indicate more anomalous)

##### `ISER_S(X)`
Compute anomaly scores using similarity-based scoring.

**Parameters:**
- `X` (array-like): Input data of shape (n_samples, n_features)

**Returns:**
- `array`: Anomaly scores (higher values indicate more anomalous)

##### `ISER_IF(X)`
Compute anomaly scores using enhanced Isolation Forest.

**Parameters:**
- `X` (array-like): Input data of shape (n_samples, n_features)

**Returns:**
- `array`: Anomaly scores (higher values indicate more anomalous)

## Examples

### Demo Notebook

Run the demonstration notebook to see ISER in action on different types of anomalies:

```bash
jupyter notebook demo.ipynb
```

The demo includes:
- Global anomaly detection on clustered data
- Local anomaly detection on spiral patterns  
- Dependency anomaly detection on correlated features



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Contact

For questions and support, please open an issue in the GitHub repository.
