import numpy as np
import pandas as pd
from scipy import stats

from numerai_neutralizer.utils.logging import (
    debug, info, warning, error, log_performance
)
from numerai_neutralizer.utils.validation import validate_data

class DataProcessor:
    """Handles data preprocessing and transformations for Numerai data."""
    
    @staticmethod
    @log_performance
    def filter_top_bottom(series: pd.Series, n: int) -> pd.Series:
        """Filter to top and bottom n values of a Series."""
        try:
            debug(
                "Starting top/bottom filtering",
                {
                    'action': 'filter_top_bottom',
                    'series_length': len(series),
                    'n': n
                }
            )
            
            if 2*n > len(series):
                raise ValueError(f"2*n ({2*n}) must be <= series length ({len(series)})")
                
            sorted_idx = np.argsort(series.values)
            bottom_idx = sorted_idx[:n]
            top_idx = sorted_idx[-n:]
            selected_idx = np.concatenate([bottom_idx, top_idx])
            result = series.iloc[selected_idx].sort_index()
            
            debug(
                "Top/bottom filtering complete",
                {
                    'action': 'filter_complete',
                    'output_length': len(result),
                    'min_value': float(result.min()),
                    'max_value': float(result.max())
                }
            )
            
            return result
        except Exception as e:
            error(
                "Error in top/bottom filtering",
                {
                    'action': 'filter_error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise

    @staticmethod
    @log_performance
    def rank(df: pd.DataFrame, method: str = "average") -> pd.DataFrame:
        """Rank transform with proper scaling to avoid numerical issues."""
        try:
            validate_data(df)
            epsilon = 1e-8
            
            info(
                "Starting rank transformation",
                {
                    'action': 'rank',
                    'method': method,
                    'input_shape': df.shape,
                    'epsilon': epsilon
                }
            )
            
            def safe_rank(s: pd.Series) -> pd.Series:
                if len(s) <= 1:
                    debug("Single value series, returning 0.5", {'length': len(s)})
                    return pd.Series(0.5, index=s.index)
                    
                if s.std() == 0:
                    debug("Zero variance series, returning 0.5", {'name': s.name})
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
                debug(
                    "Automatically sorting non-monotonic index in rank transform",
                    {'action': 'sort_index'}
                )
                df = df.sort_index()
                
            ranked = df.apply(safe_rank)
            result = ranked.clip(epsilon, 1 - epsilon)
            
            info(
                "Rank transformation complete",
                {
                    'action': 'rank_complete',
                    'output_shape': result.shape,
                    'min_value': float(result.min().min()),
                    'max_value': float(result.max().max())
                }
            )
            
            return result
            
        except Exception as e:
            error(
                "Error in rank transformation",
                {
                    'action': 'rank_error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise

    @staticmethod
    @log_performance
    def gaussian(df: pd.DataFrame) -> pd.DataFrame:
        """Gaussianize data using inverse normal CDF."""
        try:
            validate_data(df)
            
            info(
                "Starting Gaussian transformation",
                {
                    'action': 'gaussian',
                    'input_shape': df.shape
                }
            )
            
            result = df.apply(lambda s: stats.norm.ppf(s))
            
            if np.isinf(result.values).any():
                raise ValueError("Infinite values encountered in gaussian transformation")
                
            info(
                "Gaussian transformation complete",
                {
                    'action': 'gaussian_complete',
                    'output_shape': result.shape,
                    'mean': float(result.mean().mean()),
                    'std': float(result.std().mean())
                }
            )
            
            return result
            
        except Exception as e:
            error(
                "Error in Gaussian transformation",
                {
                    'action': 'gaussian_error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise

    @staticmethod
    @log_performance
    def power(df: pd.DataFrame, p: float) -> pd.DataFrame:
        """Raise data to power while preserving sign and handling edge cases."""
        try:
            validate_data(df)
            
            info(
                "Starting power transformation",
                {
                    'action': 'power',
                    'power': p,
                    'input_shape': df.shape
                }
            )
            
            with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
                result = np.sign(df) * np.abs(df) ** p
                
                # Handle infinities
                inf_count = np.isinf(result.values).sum()
                if inf_count > 0:
                    debug(
                        "Replacing infinite values with NaN",
                        {
                            'action': 'handle_inf',
                            'inf_count': int(inf_count)
                        }
                    )
                    result = result.replace([np.inf, -np.inf], np.nan)
                
                # Verify correlation with original data
                non_zero_std = df.std() != 0
                if non_zero_std.any():
                    correlations = result.loc[:, non_zero_std].corrwith(df.loc[:, non_zero_std])
                    if not (correlations >= 0.9).all():
                        raise ValueError("Power transform decorrelated data")
                    
                    debug(
                        "Correlation verification complete",
                        {
                            'action': 'correlation_check',
                            'min_correlation': float(correlations.min()),
                            'mean_correlation': float(correlations.mean())
                        }
                    )
                
                info(
                    "Power transformation complete",
                    {
                        'action': 'power_complete',
                        'output_shape': result.shape,
                        'nan_count': int(result.isna().sum().sum())
                    }
                )
                
                return result
                
        except Exception as e:
            error(
                "Error in power transformation",
                {
                    'action': 'power_error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise

    @staticmethod
    @log_performance
    def standardize(df: pd.DataFrame) -> pd.DataFrame:
        """Scale to mean=0, std=1."""
        try:
            validate_data(df)
            
            info(
                "Starting standardization",
                {
                    'action': 'standardize',
                    'input_shape': df.shape
                }
            )
            
            result = (df - df.mean()) / df.std()
            
            info(
                "Standardization complete",
                {
                    'action': 'standardize_complete',
                    'output_shape': result.shape,
                    'mean': float(result.mean().mean()),
                    'std': float(result.std().mean())
                }
            )
            
            return result
            
        except Exception as e:
            error(
                "Error in standardization",
                {
                    'action': 'standardize_error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise
    
    @staticmethod
    @log_performance
    def variance_normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Scale to std=1."""
        try:
            validate_data(df)
            
            info(
                "Starting variance normalization",
                {
                    'action': 'variance_normalize',
                    'input_shape': df.shape
                }
            )
            
            std = df.std()
            zero_std_cols = (std == 0)
            if zero_std_cols.any():
                warning(
                    "Zero standard deviation columns found",
                    {
                        'action': 'zero_std_warning',
                        'zero_std_count': int(zero_std_cols.sum()),
                        'columns': zero_std_cols[zero_std_cols].index.tolist()
                    }
                )
            
            result = df / std.replace(0, 1)  # Replace zero std with 1
            
            info(
                "Variance normalization complete",
                {
                    'action': 'variance_normalize_complete',
                    'output_shape': result.shape,
                    'mean': float(result.mean().mean()),
                    'std': float(result.std().mean())
                }
            )
            
            return result
            
        except Exception as e:
            error(
                "Error in variance normalization",
                {
                    'action': 'variance_normalize_error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise