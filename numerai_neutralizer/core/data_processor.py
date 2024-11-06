import numpy as np
import pandas as pd
from scipy import stats

from numerai_neutralizer.utils.logging import logger
from numerai_neutralizer.utils.validation import validate_data

class DataProcessor:
    """Handles data preprocessing and transformations for Numerai data."""
    
    @staticmethod
    def filter_top_bottom(series: pd.Series, n: int) -> pd.Series:
        """Filter to top and bottom n values of a Series."""
        if 2*n > len(series):
            raise ValueError(f"2*n ({2*n}) must be <= series length ({len(series)})")
            
        sorted_idx = np.argsort(series.values)
        bottom_idx = sorted_idx[:n]
        top_idx = sorted_idx[-n:]
        selected_idx = np.concatenate([bottom_idx, top_idx])
        return series.iloc[selected_idx].sort_index()

    @staticmethod
    def rank(df: pd.DataFrame, method: str = "average") -> pd.DataFrame:
        """Rank transform with proper scaling to avoid numerical issues.
        
        Args:
            df: DataFrame or Series to rank
            method: Ranking method (default: "average")
            
        Returns:
            Ranked data scaled to [epsilon, 1-epsilon]
            
        Note:
            Automatically sorts index if needed for consistent results
        """
        validate_data(df)
        epsilon = 1e-8
        
        def safe_rank(s: pd.Series) -> pd.Series:
            if len(s) <= 1:
                return pd.Series(0.5, index=s.index)
                
            if s.std() == 0:
                return pd.Series(0.5, index=s.index)
            
            # Sort index if needed
            if not s.index.is_monotonic_increasing:
                s = s.sort_index()
                
            ranks = s.rank(method=method)
            return (ranks - 1) / (len(ranks) - 1)
            
        if isinstance(df, pd.Series):
            df = df.to_frame()
            
        # Check if any column needs sorting
        if not df.index.is_monotonic_increasing:
            logger.debug("Automatically sorting non-monotonic index in rank transform")
            df = df.sort_index()
            
        ranked = df.apply(safe_rank)
        return ranked.clip(epsilon, 1 - epsilon)

    @staticmethod
    def gaussian(df: pd.DataFrame) -> pd.DataFrame:
        """Gaussianize data using inverse normal CDF."""
        validate_data(df)
        result = df.apply(lambda s: stats.norm.ppf(s))
        
        if np.isinf(result.values).any():
            raise ValueError("Infinite values encountered in gaussian transformation")
            
        return result

    @staticmethod
    def power(df: pd.DataFrame, p: float) -> pd.DataFrame:
        """Raise data to power while preserving sign and handling edge cases."""
        validate_data(df)
        
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            result = np.sign(df) * np.abs(df) ** p
            
            # Handle infinities
            result = result.replace([np.inf, -np.inf], np.nan)
            
            # Verify correlation with original data
            non_zero_std = df.std() != 0
            if non_zero_std.any():
                correlations = result.loc[:, non_zero_std].corrwith(df.loc[:, non_zero_std])
                assert (correlations >= 0.9).all(), "Power transform decorrelated data"
            
            return result

    @staticmethod
    def standardize(df: pd.DataFrame) -> pd.DataFrame:
        """Scale to mean=0, std=1."""
        validate_data(df)
        return (df - df.mean()) / df.std()
    
    @staticmethod
    def variance_normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Scale to std=1."""
        validate_data(df)
        std = df.std()
        if (std == 0).any():
            logger.warning("Zero standard deviation columns found")
        return df / std.replace(0, 1)  # Replace zero std with 1 to avoid division by zero