import pytest
import numpy as np
import pandas as pd
from numerai_neutralizer import NumeraiNeutralizer, NumeraiModel
from sklearn.linear_model import LinearRegression
import warnings

def test_neutralization():
    """Test basic neutralization functionality."""
    neutralizer = NumeraiNeutralizer()
    np.random.seed(42)
    n_samples = 1000

    # Create feature that predictions are correlated with
    feature = pd.Series(np.random.randn(n_samples))

    # Create predictions correlated with feature
    predictions = 0.7 * feature + 0.3 * np.random.randn(n_samples)
    predictions = pd.DataFrame({
        "prediction": predictions
    })

    # Neutralize predictions
    neutral_preds = neutralizer.neutralize(
        predictions,
        feature.to_frame("feature")
    )

    # Check correlation is reduced
    original_corr = abs(predictions["prediction"].corr(feature))
    neutral_corr = abs(neutral_preds["prediction"].corr(feature))

    assert neutral_corr < original_corr
    assert neutral_corr < 0.1  # Should be close to zero

def test_calculate_feature_exposure_zero_variance():
    """Test calculate_feature_exposure with a zero variance feature."""
    neutralizer = NumeraiNeutralizer()
    np.random.seed(42)
    n_samples = 100

    # Create features with one zero variance feature
    features = pd.DataFrame(np.random.randn(n_samples, 4), columns=[f'feature_{i}' for i in range(4)])
    features['zero_variance'] = 1.0  # Add zero variance feature

    # Create random predictions
    predictions = pd.Series(np.random.randn(n_samples), name='prediction')

    # Suppress RuntimeWarnings for invalid value encountered in divide
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        exposures = neutralizer.calculate_feature_exposure(predictions, features)

    # Assertions
    assert 'zero_variance' in exposures.index, "Exposures should include zero_variance feature"
    assert exposures['zero_variance'] == 0.0, "Exposure to zero variance feature should be zero"

def test_calculate_feature_exposure_with_nan():
    """Test calculate_feature_exposure with NaN values in features."""
    neutralizer = NumeraiNeutralizer()
    np.random.seed(42)
    n_samples = 100

    # Create features with NaN values
    features = pd.DataFrame(np.random.randn(n_samples, 5), columns=[f'feature_{i}' for i in range(5)])
    features.iloc[0, 0] = np.nan  # Introduce NaN

    # Create random predictions
    predictions = pd.Series(np.random.randn(n_samples), name='prediction')

    # Expect the method to raise a ValueError due to NaN values
    with pytest.raises(ValueError, match="features contains NaN values"):
        exposures = neutralizer.calculate_feature_exposure(predictions, features)

def test_calculate_feature_exposure_misaligned_indices():
    """Test calculate_feature_exposure with misaligned indices."""
    neutralizer = NumeraiNeutralizer()
    np.random.seed(42)
    n_samples = 100

    # Create features with indices 0 to 99
    features = pd.DataFrame(np.random.randn(n_samples, 5), columns=[f'feature_{i}' for i in range(5)])

    # Create predictions with indices 100 to 199
    predictions = pd.Series(np.random.randn(n_samples), index=range(100, 200), name='prediction')

    # Expect the method to handle index alignment or raise an error
    with pytest.raises(ValueError, match="No overlapping indices between predictions and features."):
        exposures = neutralizer.calculate_feature_exposure(predictions, features)

def test_numerai_model_predict_with_metrics():
    """Test NumeraiModel's predict_with_metrics method."""
    np.random.seed(42)
    n_samples = 100

    # Create features and target
    features = pd.DataFrame(np.random.randn(n_samples, 5), columns=[f'feature_{i}' for i in range(5)])
    target = features['feature_0'] * 0.5 + np.random.randn(n_samples) * 0.5
    target = pd.Series(target, name='target')

    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(features, target)

    # Initialize NumeraiModel
    neutralizer = NumeraiNeutralizer()
    numerai_model = NumeraiModel(
        model=model,
        neutralizer=neutralizer,
        features=features.columns.tolist(),
        neutralization_features=features.columns.tolist(),
        proportion=0.5
    )

    # Generate predictions and metrics
    predictions, metrics = numerai_model.predict_with_metrics(
        X=features,
        target=target,
        era_col=None  # Not using eras in this test
    )

    # Assertions
    assert 'feature_exposure' in metrics, "Metrics should include feature_exposure"
    assert 'max_feature_exposure' in metrics, "Metrics should include max_feature_exposure"
    assert len(metrics['feature_exposure']) == 5, "Feature exposure length should match number of features"

    # Check that max_feature_exposure is reasonable
    assert metrics['max_feature_exposure'] > 0.0, "Max feature exposure should be positive"

    # Ensure predictions are not NaN
    assert not predictions.isna().any().any(), "Predictions should not contain NaN values"
