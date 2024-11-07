import numpy as np
import pandas as pd
from typing import Optional

from numerai_neutralizer.core.data_processor import DataProcessor
from numerai_neutralizer.utils.validation import validate_data
from numerai_neutralizer.utils.logging import (
    debug, info, error, log_performance
)

class CorrelationCalculator:
    """Handles correlation calculations for Numerai data."""
    
    @staticmethod
    def validate_indices(target: pd.Series, predictions: pd.Series) -> None:
        """Validate that indices match and are properly sorted."""
        try:
            debug(
                "Validating indices",
                {
                    'action': 'validate_indices',
                    'target_length': len(target),
                    'predictions_length': len(predictions)
                }
            )
            
            if not np.array_equal(predictions.index, target.index.sort_values()):
                error(
                    "Index mismatch",
                    {
                        'action': 'index_validation',
                        'target_index_sample': target.index[:5].tolist(),
                        'predictions_index_sample': predictions.index[:5].tolist()
                    }
                )
                raise AssertionError("Prediction indices must match sorted target indices")
            
            nan_preds = predictions.isna().sum()
            nan_targets = target.isna().sum()
            
            if nan_preds > 0 or nan_targets > 0:
                error(
                    "NaN values detected",
                    {
                        'action': 'nan_validation',
                        'nan_predictions': int(nan_preds),
                        'nan_targets': int(nan_targets)
                    }
                )
                if nan_preds > 0:
                    raise AssertionError("Predictions contain NaN values")
                if nan_targets > 0:
                    raise AssertionError("Targets contain NaN values")
                    
            debug("Index validation successful", {'action': 'validate_complete'})
            
        except Exception as e:
            error(
                "Error in index validation",
                {
                    'action': 'validate_error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise

    @staticmethod
    @log_performance
    def pearson(
        target: pd.Series,
        predictions: pd.Series,
        top_bottom: Optional[int] = None
    ) -> float:
        """Calculate Pearson correlation, optionally on top/bottom subset."""
        try:
            info(
                "Starting Pearson correlation calculation",
                {
                    'action': 'pearson_start',
                    'top_bottom': top_bottom,
                    'data_length': len(predictions)
                }
            )
            
            validate_data(target, "target")
            validate_data(predictions, "predictions")
            
            if top_bottom:
                debug(
                    "Filtering to top/bottom subset",
                    {
                        'action': 'top_bottom_filter',
                        'n': top_bottom,
                        'original_length': len(predictions)
                    }
                )
                predictions = DataProcessor.filter_top_bottom(predictions, top_bottom)
                target = target.loc[predictions.index]
            
            CorrelationCalculator.validate_indices(target, predictions)
            correlation = target.corr(predictions, method="pearson")
            
            info(
                "Pearson correlation calculation complete",
                {
                    'action': 'pearson_complete',
                    'correlation': float(correlation),
                    'final_length': len(predictions)
                }
            )
            
            return correlation
            
        except Exception as e:
            error(
                "Error calculating Pearson correlation",
                {
                    'action': 'pearson_error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise

    @staticmethod
    @log_performance
    def numerai_correlation(
        predictions: pd.DataFrame,
        targets: pd.Series,
        max_filtered_ratio: float = 0.2,
        top_bottom: Optional[int] = None
    ) -> pd.Series:
        """Calculate canonical Numerai correlation."""
        try:
            info(
                "Starting Numerai correlation calculation",
                {
                    'action': 'numerai_correlation_start',
                    'predictions_shape': predictions.shape,
                    'targets_length': len(targets),
                    'max_filtered_ratio': max_filtered_ratio,
                    'top_bottom': top_bottom
                }
            )
            
            validate_data(predictions, "predictions")
            validate_data(targets, "targets")
            
            targets = targets - targets.mean()
            
            # Filter and align indices
            common_idx = predictions.index.intersection(targets.index)
            filtered_ratio = 1 - len(common_idx) / len(predictions)
            
            debug(
                "Index alignment stats",
                {
                    'action': 'index_alignment',
                    'common_indices': len(common_idx),
                    'filtered_ratio': float(filtered_ratio),
                    'original_length': len(predictions)
                }
            )
            
            if filtered_ratio > max_filtered_ratio:
                error(
                    f"Too many indices filtered: {filtered_ratio:.2%}",
                    {
                        'action': 'filtered_ratio_error',
                        'filtered_ratio': float(filtered_ratio),
                        'max_allowed': max_filtered_ratio
                    }
                )
                raise ValueError(f"Too many indices filtered: {filtered_ratio:.2%}")
                
            predictions = predictions.loc[common_idx]
            targets = targets.loc[common_idx]
            
            # Transform data
            debug("Applying transformations to predictions")
            predictions = DataProcessor.power(
                DataProcessor.gaussian(
                    DataProcessor.rank(predictions)
                ), 1.5
            )
            
            debug("Applying transformations to targets")
            targets = DataProcessor.power(targets.to_frame(), 1.5)[targets.name]
            
            # Calculate correlations
            info("Calculating per-column correlations")
            correlations = predictions.apply(
                lambda p: CorrelationCalculator.pearson(targets, p, top_bottom)
            )
            
            info(
                "Numerai correlation calculation complete",
                {
                    'action': 'numerai_correlation_complete',
                    'mean_correlation': float(correlations.mean()),
                    'min_correlation': float(correlations.min()),
                    'max_correlation': float(correlations.max()),
                    'std_correlation': float(correlations.std())
                }
            )
            
            return correlations
            
        except Exception as e:
            error(
                "Error calculating Numerai correlation",
                {
                    'action': 'numerai_correlation_error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise