# numerai-neutralizer

A Python library for feature neutralization and analysis of Numerai predictions. This package provides tools for neutralizing predictions against feature exposures, calculating correlations, and analyzing model performance with Numerai-specific metrics.

## Installation

Install using pip:

```bash
pip install git+https://github.com/whit3rabbit/numerai-neutralizer.git
```

Requires Python 3.8 or later.

## Quick Start

```python
from numerai_neutralizer import NumeraiNeutralizer, NumeraiModel
import pandas as pd

# Initialize neutralizer
neutralizer = NumeraiNeutralizer()

# Create predictions and feature data
predictions_df = pd.DataFrame(...)  # Your predictions
feature_df = pd.DataFrame(...)      # Your feature data

# Neutralize predictions
neutral_predictions = neutralizer.neutralize(
    predictions_df,
    feature_df,
    proportion=1.0
)
```

## Numerai Model Submission Example

Here's a complete example of how to use the library for a Numerai model submission:

```python
from numerai_neutralizer import NumeraiNeutralizer, NumeraiModel
import pandas as pd
import cloudpickle

# Assume you have:
# - your_trained_model: Your trained model (sklearn, lightgbm, etc.)
# - medium_features: List of feature names used for training
# - med_serenity_feats: List of features to neutralize against

# 1. Set up the neutralizer and model wrapper
neutralizer = NumeraiNeutralizer()
model = NumeraiModel(
    model=your_trained_model,          # Your trained model
    neutralizer=neutralizer,
    features=medium_features,          # Features used for prediction
    neutralization_features=med_serenity_feats,  # Features to neutralize against
    proportion=1.0                     # Full neutralization
)

# 2. Create prediction function for Numerai submission
def predict_neutral(live_features: pd.DataFrame) -> pd.DataFrame:
    """Generate neutralized predictions for Numerai submission.
    
    Args:
        live_features: DataFrame containing current Numerai features
        
    Returns:
        DataFrame with neutralized and ranked predictions
    """
    return model.predict(live_features)

# 3. Test the pipeline
live_features = pd.read_parquet("v5.0/live.parquet", columns=medium_features)
predictions = predict_neutral(live_features)

# 4. Save for submission
with open("feature_neutralization.pkl", "wb") as f:
    f.write(cloudpickle.dumps(predict_neutral))
```

You can also get detailed metrics during development:

```python
# Get predictions with comprehensive metrics
predictions, metrics = model.predict_with_metrics(
    X=live_features,
    meta_model=meta_model_predictions,  # Optional: for MMC calculation
    target=targets,                     # Optional: for correlation metrics
    era_col='era'                      # Optional: for era-wise analysis
)

# Access various metrics
feature_exposure = metrics['feature_exposure']
correlation = metrics['correlation']
era_metrics = metrics['era_wise']
mmc_scores = metrics['mmc']
```

## Features

- **Feature Neutralization**: Remove unwanted feature exposures from predictions
- **Correlation Metrics**: Calculate Numerai-specific correlation metrics
- **MMC Calculation**: Compute Meta-Model Contribution scores
- **Model Integration**: Wrapper for easy integration with any sklearn-compatible model
- **Data Processing**: Utilities for common Numerai data transformations
- **Performance Analysis**: Comprehensive metrics and diagnostics

## Core Components

### NumeraiNeutralizer

Main class for feature neutralization and analysis:

```python
from numerai_neutralizer import NumeraiNeutralizer

neutralizer = NumeraiNeutralizer()

# Basic neutralization
neutral_preds = neutralizer.neutralize(
    predictions,
    neutralizers,
    proportion=1.0
)

# Calculate MMC
mmc_scores = neutralizer.calculate_mmc(
    predictions,
    meta_model,
    targets
)
```

### NumeraiModel

Wrapper for models with integrated neutralization:

```python
from numerai_neutralizer import NumeraiModel
from sklearn.linear_model import LinearRegression

model = NumeraiModel(
    model=LinearRegression(),
    neutralizer=neutralizer,
    features=['feature1', 'feature2'],
    neutralization_features=['feature1'],
    proportion=1.0
)

# Get predictions with metrics
predictions, metrics = model.predict_with_metrics(
    X=feature_data,
    meta_model=meta_model,
    target=targets,
    era_col='era'
)
```

### DataProcessor

Utilities for data transformation:

```python
from numerai_neutralizer import DataProcessor

# Rank transform
ranked_data = DataProcessor.rank(data)

# Gaussianize data
gaussian_data = DataProcessor.gaussian(ranked_data)

# Standardize
standardized = DataProcessor.standardize(data)
```

## Advanced Usage

### Feature Exposure Analysis

```python
# Get predictions with full metrics
predictions, metrics = model.predict_with_metrics(
    X=feature_data,
    meta_model=meta_model,
    target=targets,
    era_col='era'
)

# Access feature exposure
feature_exposure = metrics['feature_exposure']
max_exposure = metrics['max_feature_exposure']
```

### Era-wise Analysis

```python
# Get era-wise metrics
predictions, metrics = model.predict_with_metrics(
    X=feature_data,
    target=targets,
    era_col='era'
)

# Access era metrics
era_corr_mean = metrics['era_wise']['mean']
era_corr_std = metrics['era_wise']['std']
era_sharpe = metrics['era_wise']['sharpe']
```

## Requirements

The package requires Python 3.8 or later. Dependencies will be installed automatically with the package. Key dependencies include:

- numpy >= 1.19.0
- pandas >= 1.2.0
- scipy >= 1.7.0
- scikit-learn >= 0.24.0
