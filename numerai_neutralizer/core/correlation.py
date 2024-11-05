import numpy as np
import pandas as pd
from typing import Optional
from numerai_neutralizer.core.data_processor import DataProcessor
from numerai_neutralizer.utils.validation import validate_data

class CorrelationCalculator:
    """Handles correlation calculations for Numerai data."""
    
    @staticmethod
    def validate_indices(target: pd.Series, predictions: pd.Series) -> None:
        """Validate that indices match and are properly sorted."""
        assert np.array_equal(predictions.index, target.index.sort_values()), \
            "Prediction indices must match sorted target indices"
        assert not predictions.isna().any(), "Predictions contain NaN values"
        assert not target.isna().any(), "Targets contain NaN values"

    @staticmethod
    def pearson(
        target: pd.Series,
        predictions: pd.Series,
        top_bottom: Optional[int] = None
    ) -> float:
        """Calculate Pearson correlation, optionally on top/bottom subset."""
        validate_data(target, "target")
        validate_data(predictions, "predictions")
        
        if top_bottom:
            predictions = DataProcessor.filter_top_bottom(predictions, top_bottom)
            target = target.loc[predictions.index]
        
        CorrelationCalculator.validate_indices(target, predictions)
        return target.corr(predictions, method="pearson")

    @staticmethod
    def numerai_correlation(
        predictions: pd.DataFrame,
        targets: pd.Series,
        max_filtered_ratio: float = 0.2,
        top_bottom: Optional[int] = None
    ) -> pd.Series:
        """Calculate canonical Numerai correlation.
        
        Applies rank -> gaussian -> power transformation before correlation.
        """
        validate_data(predictions, "predictions")
        validate_data(targets, "targets")
        
        targets = targets - targets.mean()
        
        # Filter and align indices
        common_idx = predictions.index.intersection(targets.index)
        filtered_ratio = 1 - len(common_idx) / len(predictions)
        
        if filtered_ratio > max_filtered_ratio:
            raise ValueError(f"Too many indices filtered: {filtered_ratio:.2%}")
            
        predictions = predictions.loc[common_idx]
        targets = targets.loc[common_idx]
        
        # Transform data
        predictions = DataProcessor.power(
            DataProcessor.gaussian(
                DataProcessor.rank(predictions)
            ), 1.5
        )
        targets = DataProcessor.power(targets.to_frame(), 1.5)[targets.name]
        
        # Calculate correlations
        return predictions.apply(
            lambda p: CorrelationCalculator.pearson(targets, p, top_bottom)
        )