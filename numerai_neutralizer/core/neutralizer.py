from typing import Dict, Optional
import numpy as np
import pandas as pd

from numerai_neutralizer.core.data_processor import DataProcessor
from numerai_neutralizer.core.correlation import CorrelationCalculator
from numerai_neutralizer.metrics.feature_metrics import FeatureMetrics
from numerai_neutralizer.utils.validation import validate_data
from numerai_neutralizer.utils.logging import (
    debug, info, warning, exception, log_performance
)

class NumeraiNeutralizer:
    """Main class for feature neutralization and analysis."""
    
    def __init__(self):
        self.feature_metrics: Dict[str, FeatureMetrics] = {}
        self.correlator = CorrelationCalculator()
        info(
            "Initialized NumeraiNeutralizer",
            {'component': 'NumeraiNeutralizer', 'action': 'init'}
        )
    
    @log_performance
    def neutralize(
        self,
        df: pd.DataFrame,
        neutralizers: pd.DataFrame,
        proportion: float = 1.0
    ) -> pd.DataFrame:
        """Neutralize predictions against features."""
        try:
            info(
                "Starting neutralization",
                {
                    'action': 'neutralize',
                    'predictions_shape': df.shape,
                    'neutralizers_shape': neutralizers.shape,
                    'proportion': proportion
                }
            )
            
            # Input validation
            validate_data(df, "predictions")
            validate_data(neutralizers, "neutralizers")
            assert len(df.index) == len(neutralizers.index), "Index mismatch"
            assert (df.index == neutralizers.index).all(), "Indices don't match"
            
            # Handle zero variance columns
            df = df.copy()
            neutralizers = neutralizers.copy()
            
            # Remove zero variance columns from neutralizers
            non_zero_cols = neutralizers.std() != 0
            zero_var_count = (~non_zero_cols).sum()
            
            if zero_var_count > 0:
                warning(
                    f"Removing zero variance neutralizer columns",
                    {
                        'action': 'remove_zero_variance',
                        'removed_columns': int(zero_var_count),
                        'remaining_columns': int(non_zero_cols.sum())
                    }
                )
                neutralizers = neutralizers.loc[:, non_zero_cols]
                
            if neutralizers.empty:
                warning(
                    "No valid neutralizer columns, returning original predictions",
                    {'action': 'skip_neutralization', 'reason': 'no_valid_neutralizers'}
                )
                return df
                
            # Add constant term for bias
            neutralizer_arr = np.hstack((
                neutralizers.values,
                np.ones((len(neutralizers), 1))
            ))
            
            # Calculate neutralization
            least_squares = np.linalg.lstsq(neutralizer_arr, df.values, rcond=1e-6)[0]
            adjustments = proportion * neutralizer_arr.dot(least_squares)
            neutral = df.values - adjustments
            
            info(
                "Neutralization completed successfully",
                {
                    'action': 'neutralize_complete',
                    'original_std': float(df.std().mean()),
                    'neutralized_std': float(pd.DataFrame(neutral).std().mean()),
                    'adjustment_magnitude': float(np.abs(adjustments).mean())
                }
            )
            
            return pd.DataFrame(neutral, index=df.index, columns=df.columns)
            
        except Exception as e:
            exception(
                "Error during neutralization",
                {
                    'action': 'neutralize_error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise

    @log_performance
    def fast_neutralize(self, v: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Fast neutralization for single column case."""
        try:
            debug(
                "Starting fast neutralization",
                {
                    'action': 'fast_neutralize',
                    'vector_length': len(v) if isinstance(v, np.ndarray) else 'unknown'
                }
            )
            
            v = v.ravel()
            u = u.ravel()
            
            if len(v) != len(u):
                raise ValueError("Vectors must have same length")
            
            u_norm_squared = u @ u
            if np.isclose(u_norm_squared, 0):
                raise ValueError("Cannot neutralize against a zero vector")
            
            projection_coefficient = (v @ u) / u_norm_squared
            result = v - u * projection_coefficient
            
            debug(
                "Fast neutralization completed",
                {
                    'action': 'fast_neutralize_complete',
                    'projection_coefficient': float(projection_coefficient),
                    'result_mean': float(np.mean(result)),
                    'result_std': float(np.std(result))
                }
            )
            
            return result
            
        except Exception as e:
            exception(
                "Error in fast neutralization",
                {
                    'action': 'fast_neutralize_error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise

    @log_performance
    def calculate_feature_exposure(
        self,
        predictions: pd.Series,
        features: pd.DataFrame
    ) -> pd.Series:
        """Calculate feature exposure of predictions to features."""
        try:
            info(
                "Calculating feature exposure",
                {
                    'action': 'calculate_feature_exposure',
                    'num_features': len(features.columns),
                    'num_samples': len(predictions)
                }
            )
            
            # Validate inputs
            validate_data(predictions, "predictions")
            validate_data(features, "features")
            
            # Ensure predictions and features have aligned indices
            predictions, features = predictions.align(features, join='inner', axis=0)
            
            if len(predictions) == 0:
                raise ValueError("No overlapping indices between predictions and features.")
                
            # Standardize features
            std = features.std()
            zero_std_features = std[std == 0].index.tolist()
            
            if zero_std_features:
                warning(
                    "Found features with zero standard deviation",
                    {
                        'action': 'zero_std_warning',
                        'zero_std_features': zero_std_features
                    }
                )
                
            standardized_features = (features - features.mean()) / std.replace(0, 1)
            
            # Compute exposure
            exposures = standardized_features.apply(lambda x: predictions.corr(x))
            
            # Set exposure to zero for zero variance features
            exposures.loc[zero_std_features] = 0.0
            
            # Handle NaN exposures
            exposures = exposures.fillna(0.0)
            
            info(
                "Feature exposure calculation completed",
                {
                    'action': 'feature_exposure_complete',
                    'max_exposure': float(abs(exposures).max()),
                    'mean_exposure': float(abs(exposures).mean()),
                    'num_significant_exposures': int((abs(exposures) > 0.1).sum())
                }
            )
            
            return exposures
            
        except Exception as e:
            exception(
                "Error calculating feature exposure",
                {
                    'action': 'feature_exposure_error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise

    @log_performance
    def calculate_mmc(
        self,
        predictions: pd.DataFrame,
        meta_model: pd.Series,
        targets: pd.Series,
        top_bottom: Optional[int] = None
    ) -> pd.Series:
        """Calculate Meta-Model Contribution scores."""
        try:
            info(
                "Starting MMC calculation",
                {
                    'action': 'calculate_mmc',
                    'predictions_shape': predictions.shape,
                    'top_bottom': top_bottom
                }
            )
            
            # Strict index alignment and sorting
            common_idx = predictions.index.intersection(meta_model.index).intersection(targets.index)
            if len(common_idx) == 0:
                raise ValueError("No overlapping indices between predictions, meta model, and targets")
            
            predictions = predictions.loc[common_idx].sort_index()
            meta_model = meta_model.loc[common_idx].sort_index()
            targets = targets.loc[common_idx].sort_index()
            
            # Transform predictions using tie-kept ranking
            p = DataProcessor.gaussian(
                DataProcessor.rank(predictions, method="average")
            )
            
            # Transform meta model predictions
            m = DataProcessor.gaussian(
                DataProcessor.rank(meta_model.to_frame(), method="average")
            )[meta_model.name]
            
            # Check for infinities
            if np.isinf(p.values).any() or np.isinf(m.values).any():
                raise ValueError("Infinite values encountered after gaussian transform")
            
            # Calculate neutralized predictions
            neutral_preds = np.zeros_like(p.values)
            for i in range(p.shape[1]):
                try:
                    # Use orthogonalization via fast_neutralize
                    neutral_preds[:, i] = self.fast_neutralize(
                        p.iloc[:, i].values,
                        m.values
                    )
                except ValueError as e:
                    warning(
                        f"Skipping column {i} in MMC calculation",
                        {
                            'action': 'mmc_column_skip',
                            'column': i,
                            'error': str(e)
                        }
                    )
                    continue
            
            # Process targets
            if (targets >= 0).all() and (targets <= 1).all():
                targets = targets * 4
            targets = targets - targets.mean()
            
            # Calculate MMC
            if top_bottom:
                result = self._calculate_mmc_top_bottom(
                    predictions=predictions,
                    neutral_preds=neutral_preds,
                    targets=targets,
                    top_bottom=top_bottom
                )
            else:
                result = pd.Series(
                    (targets.values.reshape(-1, 1) * neutral_preds).mean(axis=0),
                    index=predictions.columns
                )
            
            info(
                "MMC calculation completed",
                {
                    'action': 'mmc_complete',
                    'mean_mmc': float(result.mean()),
                    'max_mmc': float(result.max()),
                    'min_mmc': float(result.min())
                }
            )
            
            return result
            
        except Exception as e:
            exception(
                "Error calculating MMC",
                {
                    'action': 'mmc_error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise

    @log_performance
    def _calculate_mmc_top_bottom(
        self,
        predictions: pd.DataFrame,
        neutral_preds: np.ndarray,
        targets: pd.Series,
        top_bottom: int
    ) -> pd.Series:
        """Helper method for MMC calculation with top/bottom filtering."""
        try:
            debug(
                "Starting top/bottom MMC calculation",
                {
                    'action': 'mmc_top_bottom',
                    'top_bottom_count': top_bottom
                }
            )
            
            mmc_values = []
            
            for i in range(predictions.shape[1]):
                pred_series = pd.Series(neutral_preds[:, i], index=predictions.index)
                
                try:
                    # Filter to top and bottom predictions
                    filtered_preds = DataProcessor.filter_top_bottom(pred_series, top_bottom)
                    
                    # Get corresponding targets
                    filtered_targets = targets.loc[filtered_preds.index]
                    
                    if len(filtered_preds) != 2 * top_bottom:
                        raise ValueError(f"Expected {2 * top_bottom} predictions after filtering")
                        
                    # Calculate MMC for this column
                    mmc = (filtered_targets * filtered_preds).mean()
                    mmc_values.append(mmc)
                    
                except ValueError as e:
                    warning(
                        f"Skipping column {i} in top/bottom MMC calculation",
                        {
                            'action': 'mmc_top_bottom_skip',
                            'column': i,
                            'error': str(e)
                        }
                    )
                    mmc_values.append(np.nan)
            
            result = pd.Series(mmc_values, index=predictions.columns)
            
            debug(
                "Top/bottom MMC calculation completed",
                {
                    'action': 'mmc_top_bottom_complete',
                    'mean_mmc': float(result.mean()),
                    'max_mmc': float(result.max()),
                    'min_mmc': float(result.min()),
                    'nan_count': int(result.isna().sum())
                }
            )
            
            return result
            
        except Exception as e:
            exception(
                "Error in top/bottom MMC calculation",
                {
                    'action': 'mmc_top_bottom_error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise