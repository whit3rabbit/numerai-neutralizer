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

def test_predict_with_metrics_basic():
    """Test basic functionality of predict_with_metrics."""
    np.random.seed(42)
    n_samples = 100

    # Create features and target
    features = pd.DataFrame(np.random.randn(n_samples, 5), columns=[f'feature_{i}' for i in range(5)])
    target = features['feature_0'] * 0.5 + np.random.randn(n_samples) * 0.5
    target = pd.Series(target, name='target')

    # Initialize model
    model = LinearRegression()
    model.fit(features, target)

    numerai_model = NumeraiModel(
        model=model,
        neutralizer=NumeraiNeutralizer(),
        features=features.columns.tolist(),
        neutralization_features=features.columns.tolist(),
        proportion=0.5
    )

    # Test basic prediction with metrics
    predictions, metrics = numerai_model.predict_with_metrics(
        X=features,
        target=target
    )

    # Basic assertions
    assert isinstance(predictions, pd.DataFrame)
    assert 'prediction' in predictions.columns
    assert isinstance(metrics, dict)
    assert 'correlation' in metrics
    assert isinstance(metrics['correlation'], float)
    assert 'diagnostics' in metrics
    assert isinstance(metrics['diagnostics']['prediction_std'], float)
    assert isinstance(metrics['diagnostics']['prediction_mean'], float)
    assert isinstance(metrics['diagnostics']['num_rows'], int)
    assert isinstance(metrics['diagnostics']['nulls'], int)

def test_predict_with_metrics_era_wise():
    """Test era-wise calculations in predict_with_metrics."""
    np.random.seed(42)
    n_samples = 200

    # Create features, target, and eras
    features = pd.DataFrame(np.random.randn(n_samples, 5), columns=[f'feature_{i}' for i in range(5)])
    target = features['feature_0'] * 0.5 + np.random.randn(n_samples) * 0.5
    target = pd.Series(target, name='target')
    eras = pd.Series(np.repeat(['era1', 'era2'], n_samples//2), name='era')

    # Initialize model
    model = LinearRegression()
    model.fit(features, target)

    numerai_model = NumeraiModel(
        model=model,
        neutralizer=NumeraiNeutralizer(),
        features=features.columns.tolist(),
        neutralization_features=features.columns.tolist(),
        proportion=0.5
    )

    # Test with era-wise calculations
    predictions, metrics = numerai_model.predict_with_metrics(
        X=features,
        target=target,
        era_col=eras
    )

    # Era-wise metric assertions
    assert 'era_wise' in metrics
    assert isinstance(metrics['era_wise'], dict)
    assert 'mean' in metrics['era_wise']
    assert 'std' in metrics['era_wise']
    assert 'sharpe' in metrics['era_wise']
    assert 'per_era' in metrics['era_wise']
    assert all(isinstance(v, float) for v in metrics['era_wise']['per_era'].values())
    assert all(isinstance(k, str) for k in metrics['era_wise']['per_era'].keys())

def test_predict_with_metrics_meta_model():
    """Test meta-model calculations in predict_with_metrics."""
    np.random.seed(42)
    n_samples = 100

    # Create features, target, and meta-model predictions
    features = pd.DataFrame(np.random.randn(n_samples, 5), columns=[f'feature_{i}' for i in range(5)])
    target = features['feature_0'] * 0.5 + np.random.randn(n_samples) * 0.5
    target = pd.Series(target, name='target')
    meta_model = pd.Series(np.random.randn(n_samples), name='meta_pred')

    # Initialize model
    model = LinearRegression()
    model.fit(features, target)

    numerai_model = NumeraiModel(
        model=model,
        neutralizer=NumeraiNeutralizer(),
        features=features.columns.tolist(),
        neutralization_features=features.columns.tolist(),
        proportion=0.5
    )

    # Test with meta-model
    predictions, metrics = numerai_model.predict_with_metrics(
        X=features,
        target=target,
        meta_model=meta_model
    )

    # Meta-model metric assertions
    assert 'mmc' in metrics
    assert isinstance(metrics['mmc'], float)

def test_predict_with_metrics_error_handling():
    """Test error handling in predict_with_metrics."""
    np.random.seed(42)
    n_samples = 100

    # Create features and model
    features = pd.DataFrame(np.random.randn(n_samples, 5), columns=[f'feature_{i}' for i in range(5)])
    model = LinearRegression()
    model.fit(features, features['feature_0'])  # Simple fit for testing

    numerai_model = NumeraiModel(
        model=model,
        neutralizer=NumeraiNeutralizer(),
        features=features.columns.tolist(),
        neutralization_features=features.columns.tolist(),
        proportion=0.5
    )

    # Test with misaligned target
    misaligned_target = pd.Series(np.random.randn(n_samples + 1), name='target')
    with pytest.raises(ValueError, match="Target length .* does not match predictions length"):
        predictions, metrics = numerai_model.predict_with_metrics(
            X=features,
            target=misaligned_target
        )

    # Test with NaN in target
    nan_target = pd.Series(np.random.randn(n_samples), name='target')
    nan_target.iloc[0] = np.nan
    with pytest.raises(ValueError, match="Missing target values after alignment"):
        predictions, metrics = numerai_model.predict_with_metrics(
            X=features,
            target=nan_target
        )

    # Test with NaN in meta_model
    nan_meta = pd.Series(np.random.randn(n_samples), name='meta_pred')
    nan_meta.iloc[0] = np.nan
    with pytest.raises(ValueError, match="Missing meta model values after alignment"):
        predictions, metrics = numerai_model.predict_with_metrics(
            X=features,
            target=pd.Series(np.random.randn(n_samples), name='target'),  # Added name
            meta_model=nan_meta
        )

def test_predict_with_metrics_feature_exposure():
    """Test feature exposure calculations in predict_with_metrics."""
    np.random.seed(42)
    n_samples = 100

    # Create features with known correlation
    base_feature = np.random.randn(n_samples)
    features = pd.DataFrame({
        'feature_0': base_feature,
        'feature_1': base_feature * 0.8 + np.random.randn(n_samples) * 0.2,
        'feature_2': np.random.randn(n_samples)
    })
    
    target = base_feature * 0.5 + np.random.randn(n_samples) * 0.5
    target = pd.Series(target, name='target')

    # Initialize model
    model = LinearRegression()
    model.fit(features, target)

    numerai_model = NumeraiModel(
        model=model,
        neutralizer=NumeraiNeutralizer(),
        features=features.columns.tolist(),
        neutralization_features=features.columns.tolist(),
        proportion=0.5
    )

    # Test feature exposure calculations
    predictions, metrics = numerai_model.predict_with_metrics(
        X=features,
        target=target
    )

    # Feature exposure assertions
    assert 'feature_exposure' in metrics
    assert isinstance(metrics['feature_exposure'], dict)
    assert all(isinstance(k, str) for k in metrics['feature_exposure'].keys())
    assert all(isinstance(v, float) for v in metrics['feature_exposure'].values())
    assert 'max_feature_exposure' in metrics
    assert isinstance(metrics['max_feature_exposure'], float)
    assert metrics['max_feature_exposure'] >= 0.0
    assert metrics['max_feature_exposure'] <= 1.0
