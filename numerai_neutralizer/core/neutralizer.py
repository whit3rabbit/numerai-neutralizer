import numpy as np
import pandas as pd
from typing import Dict, Optional

from numerai_neutralizer.core.data_processor import DataProcessor
from numerai_neutralizer.core.correlation import CorrelationCalculator
from numerai_neutralizer.metrics.feature_metrics import FeatureMetrics
from numerai_neutralizer.utils.logging import logger
from numerai_neutralizer.utils.validation import validate_data


class NumeraiNeutralizer:
    """Main class for feature neutralization and analysis."""
    
    def __init__(self):
        self.feature_metrics: Dict[str, FeatureMetrics] = {}
        self.correlator = CorrelationCalculator()
    
    def neutralize(
        self,
        df: pd.DataFrame,
        neutralizers: pd.DataFrame,
        proportion: float = 1.0
    ) -> pd.DataFrame:
        """Neutralize predictions against features."""
        try:
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
            if not non_zero_cols.all():
                logger.warning(f"Removing {(~non_zero_cols).sum()} zero variance neutralizer columns")
                neutralizers = neutralizers.loc[:, non_zero_cols]
                
            if neutralizers.empty:
                logger.warning("No valid neutralizer columns, returning original predictions")
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
            
            return pd.DataFrame(neutral, index=df.index, columns=df.columns)
            
        except Exception as e:
            logger.error(f"Error during neutralization: {str(e)}")
            raise

    def fast_neutralize(self, v: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Fast neutralization for single column case."""
        try:
            v = v.ravel()
            u = u.ravel()
            
            if len(v) != len(u):
                raise ValueError("Vectors must have same length")
            
            u_norm_squared = u @ u
            if np.isclose(u_norm_squared, 0):
                raise ValueError("Cannot neutralize against a zero vector")
            
            projection_coefficient = (v @ u) / u_norm_squared
            return v - u * projection_coefficient
            
        except Exception as e:
            logger.error(f"Error in fast neutralization: {str(e)}")
            raise

    def calculate_feature_exposure(
        self,
        predictions: pd.Series,
        features: pd.DataFrame
    ) -> pd.Series:
        """Calculate feature exposure of predictions to features.

        Args:
            predictions: A Series of predictions.
            features: A DataFrame of features.

        Returns:
            A Series containing the exposure of the predictions to each feature.
        """
        # Validate inputs
        validate_data(predictions, "predictions")
        validate_data(features, "features")

        # Ensure predictions and features have aligned indices
        predictions, features = predictions.align(features, join='inner', axis=0)

        # Check if any indices remain after alignment
        if len(predictions) == 0:
            raise ValueError("No overlapping indices between predictions and features.")

        # Standardize features
        std = features.std()
        zero_std_features = std[std == 0].index.tolist()
        standardized_features = (features - features.mean()) / std.replace(0, 1)

        # Compute exposure
        exposures = standardized_features.apply(lambda x: predictions.corr(x))

        # Set exposure to zero for zero variance features
        exposures.loc[zero_std_features] = 0.0

        # Handle NaN exposures due to constant features or NaN in data
        exposures = exposures.fillna(0.0)

        return exposures

    def calculate_mmc(
        self,
        predictions: pd.DataFrame,
        meta_model: pd.Series,
        targets: pd.Series,
        top_bottom: Optional[int] = None
    ) -> pd.Series:
        """Calculate Meta-Model Contribution scores."""
        try:
            # Strict index alignment
            common_idx = predictions.index.intersection(meta_model.index).intersection(targets.index)
            if len(common_idx) == 0:
                raise ValueError("No overlapping indices between predictions, meta model, and targets")
                
            predictions = predictions.loc[common_idx]
            meta_model = meta_model.loc[common_idx]
            targets = targets.loc[common_idx]
            
            # Transform data
            p = DataProcessor.gaussian(DataProcessor.rank(predictions))
            m = DataProcessor.gaussian(
                DataProcessor.rank(meta_model.to_frame())
            )[meta_model.name]
            
            # Check for infinities
            if np.isinf(p.values).any() or np.isinf(m.values).any():
                raise ValueError("Infinite values encountered after gaussian transform")
            
            # Calculate neutralized predictions
            neutral_preds = np.zeros_like(p.values)
            for i in range(p.shape[1]):
                try:
                    neutral_preds[:, i] = self.fast_neutralize(
                        p.iloc[:, i].values, 
                        m
                    )
                except ValueError as e:
                    logger.warning(f"Skipping column {i} due to neutralization error: {str(e)}")
                    continue
                    
            # Process targets
            if (targets >= 0).all() and (targets <= 1).all():
                targets = targets * 4
            targets = targets - targets.mean()
            
            # Calculate MMC
            if top_bottom:
                return self._calculate_mmc_top_bottom(
                    predictions, neutral_preds, targets, top_bottom
                )
            else:
                mmc = (targets.values.reshape(-1, 1) * neutral_preds).mean(axis=0)
                return pd.Series(mmc, index=predictions.columns)
                
        except Exception as e:
            logger.error(f"Error calculating MMC: {str(e)}")
            raise

    def _calculate_mmc_top_bottom(
        self,
        predictions: pd.DataFrame,
        neutral_preds: np.ndarray,
        targets: pd.Series,
        top_bottom: int
    ) -> pd.Series:
        """Helper method for MMC calculation with top/bottom filtering."""
        mmc_values = []
        for i in range(predictions.shape[1]):
            col_preds = pd.Series(neutral_preds[:, i], index=predictions.index)
            try:
                filtered_preds = DataProcessor.filter_top_bottom(col_preds, top_bottom)
                col_targets = targets.loc[filtered_preds.index]
                mmc = (col_targets * filtered_preds).mean()
            except ValueError as e:
                logger.warning(f"Skipping column {i} in MMC calculation: {str(e)}")
                mmc = 0
            mmc_values.append(mmc)
                
        return pd.Series(mmc_values, index=predictions.columns)