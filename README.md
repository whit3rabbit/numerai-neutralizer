# numerai-neutralizer

A robust Python library designed for feature neutralization and comprehensive analysis of Numerai predictions. This package offers tools to neutralize predictions against feature exposures, calculate key correlations, and analyze model performance with Numerai-specific metrics, including Meta-Model Contribution (MMC) and era-wise analysis.

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

# Initialize components
neutralizer = NumeraiNeutralizer()
model = NumeraiModel(
    model=your_trained_model,  # Your trained model (sklearn API)
    neutralizer=neutralizer,
    features=feature_names,      # Features used in training
    neutralization_features=neutralization_feature_names,  # Features to neutralize against (optional)
    proportion=1.0             # Neutralization strength (0-1)
)

# Load data (replace with your data loading logic)
training_data = pd.read_parquet("training_data.parquet")
tournament_data = pd.read_parquet("tournament_data.parquet")

# Fit the model (if not already fitted)
model.fit(training_data[feature_names], training_data["target"])

# Generate neutralized predictions
predictions = model.predict(tournament_data)

# Get predictions with metrics (optional, for analysis)
predictions, metrics = model.predict_with_metrics(
    X=tournament_data,
    target=tournament_data.get("target"),  # Include for correlation and MMC
    meta_model=tournament_data.get("meta_model"), # Include for MMC if available
    era_col="era"  # Include for era-wise metrics if available
)
```

## Key Features

* **Feature Neutralization:** Mitigate unintended feature exposures, handling cases like zero-variance neutralizers.
* **Correlation Analysis:** Compute Numerai-specific correlation and feature exposure metrics.
* **MMC Calculation:** Calculate Meta-Model Contribution (MMC) scores, handling top/bottom scenarios and edge cases.
* **Model Integration:**  `NumeraiModel` wrapper for easy integration with any model supporting the sklearn API (`.fit` and `.predict`).
* **Data Processing:** Utilities for ranking, gaussianization, standardization, and other data transformations.
* **Performance Evaluation:** Comprehensive metrics including era-wise analysis, correlation, feature exposure, MMC, and more.
* **Robust Logging:** Structured logging with JSON output for detailed insights and debugging.
* **Performance Monitoring:** `@log_performance` decorator tracks function execution times.
* **Thorough Testing:**  Unit tests cover core functions, edge cases, and error handling.


## Core Components

### `NumeraiNeutralizer`

Performs neutralization, feature exposure calculation, and MMC calculation.

```python
neutralizer = NumeraiNeutralizer()
neutralized_predictions = neutralizer.neutralize(predictions, features, proportion=0.5)
feature_exposures = neutralizer.calculate_feature_exposure(predictions, features)
mmc_scores = neutralizer.calculate_mmc(predictions, meta_model_predictions, targets)
```

### `NumeraiModel`

Wraps your prediction model for seamless Numerai integration.

```python
model = NumeraiModel(your_model, neutralizer, features, neutralization_features, proportion)
predictions = model.predict(data)
predictions, metrics = model.predict_with_metrics(data, target, meta_model, era_col)
```

### `DataProcessor`

Provides static methods for data transformations.

```python
ranked_data = DataProcessor.rank(data)
gaussian_data = DataProcessor.gaussian(data)
standardized_data = DataProcessor.standardize(data)
variance_normalized_data = DataProcessor.variance_normalize(data)
```

## Requirements

Python 3.8 or later. Dependencies are automatically installed:

- numpy
- pandas
- scipy
- scikit-learn


## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License