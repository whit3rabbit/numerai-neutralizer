import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from numerai_neutralizer.utils.logging import (
    log_performance, debug, info, warning, error, exception
)
from numerai_neutralizer.utils.validation import validate_data
from numerai_neutralizer.core.neutralizer import NumeraiNeutralizer

class NumeraiModel:
    """Wrapper for models with feature neutralization.
    
    This class wraps any model with a predict method and adds Numerai-specific
    functionality like feature neutralization and performance metrics.
    
    Attributes:
        model: Any model with a predict method
        neutralizer: NumeraiNeutralizer instance for feature neutralization
        features: List of feature names to use for predictions
        neutralization_features: Optional list of features to neutralize against
        proportion: Strength of neutralization (0-1)
    """
    
    def __init__(
        self,
        model: Any,
        neutralizer: NumeraiNeutralizer,
        features: List[str],
        neutralization_features: Optional[List[str]] = None,
        proportion: float = 1.0
    ):
        """Initialize NumeraiModel."""
        try:
            info(
                "Initializing NumeraiModel",
                {
                    'action': 'init',
                    'num_features': len(features),
                    'num_neutralization_features': len(neutralization_features) if neutralization_features else len(features),
                    'proportion': proportion,
                    'model_type': type(model).__name__
                }
            )
            
            if not hasattr(model, 'predict'):
                raise ValueError("Model must have predict method")
                
            if not features:
                raise ValueError("Features list cannot be empty")
                
            if not 0 <= proportion <= 1:
                raise ValueError("Proportion must be between 0 and 1")
                
            self.model = model
            self.neutralizer = neutralizer
            self.features = features
            self.neutralization_features = neutralization_features or features
            self.proportion = proportion
            
            info(
                "NumeraiModel initialized successfully",
                {
                    'action': 'init_complete',
                    'features': features[:5] + ['...'] if len(features) > 5 else features,
                    'neutralization_features': (neutralization_features[:5] + ['...'] 
                                             if neutralization_features and len(neutralization_features) > 5 
                                             else neutralization_features)
                }
            )
            
        except Exception as e:
            exception(
                "Error initializing NumeraiModel",
                {
                    'action': 'init_error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise
    
    @log_performance
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate neutralized predictions."""
        try:
            info(
                "Starting prediction generation",
                {
                    'action': 'predict_start',
                    'input_shape': X.shape,
                    'num_features': len(self.features),
                    'neutralization_enabled': bool(self.neutralization_features)
                }
            )
            
            validate_data(X, "feature data")
            
            # Validate features
            missing_features = set(self.features) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
                
            # Generate predictions
            debug("Generating raw predictions")
            predictions = pd.DataFrame(
                self.model.predict(X[self.features]),
                index=X.index,
                columns=["prediction"]
            )
            
            info(
                "Raw predictions generated",
                {
                    'action': 'raw_predictions',
                    'predictions_shape': predictions.shape,
                    'predictions_stats': {
                        'mean': float(predictions.mean().iloc[0]),
                        'std': float(predictions.std().iloc[0]),
                        'min': float(predictions.min().iloc[0]),
                        'max': float(predictions.max().iloc[0])
                    }
                }
            )
            
            # Apply neutralization if features specified
            if self.neutralization_features:
                missing_neutral = set(self.neutralization_features) - set(X.columns)
                if missing_neutral:
                    raise ValueError(f"Missing neutralization features: {missing_neutral}")
                    
                info(
                    "Applying feature neutralization",
                    {
                        'action': 'neutralization',
                        'num_neutralization_features': len(self.neutralization_features),
                        'proportion': self.proportion
                    }
                )
                
                predictions = self.neutralizer.neutralize(
                    predictions,
                    X[self.neutralization_features],
                    self.proportion
                )
                
                info(
                    "Neutralization completed",
                    {
                        'action': 'neutralization_complete',
                        'neutralized_stats': {
                            'mean': float(predictions.mean().iloc[0]),
                            'std': float(predictions.std().iloc[0])
                        }
                    }
                )
                
            # Rank transform predictions
            info("Ranking predictions...")
            ranked_preds = predictions.rank(pct=True)
            
            # Validate final predictions
            if ranked_preds.isna().any().any():
                raise ValueError("NaN values in final predictions")
                
            info(
                "Prediction generation completed",
                {
                    'action': 'predict_complete',
                    'final_shape': ranked_preds.shape,
                    'final_stats': {
                        'mean': float(ranked_preds.mean().iloc[0]),
                        'std': float(ranked_preds.std().iloc[0])
                    }
                }
            )
            
            return ranked_preds
            
        except Exception as e:
            error(
                "Error generating predictions",
                {
                    'action': 'predict_error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise

    @log_performance
    def predict_with_metrics(
        self,
        X: pd.DataFrame,
        meta_model: Optional[pd.Series] = None,
        target: Optional[pd.Series] = None,
        era_col: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate predictions with comprehensive performance metrics."""
        try:
            info(
                "Starting prediction with metrics",
                {
                    'action': 'predict_metrics_start',
                    'input_shape': X.shape,
                    'has_meta_model': meta_model is not None,
                    'has_target': target is not None,
                    'has_era': era_col is not None
                }
            )
            
            # Generate predictions
            predictions = self.predict(X)
            metrics: Dict[str, Any] = {}
            
            if target is not None:
                info("Processing target data and calculating metrics")
                
                # Ensure target has a name
                if target.name is None:
                    target = target.copy()
                    target.name = 'target'
                    
                # Validate dimensions
                if len(target) != len(predictions):
                    raise ValueError(f"Target length ({len(target)}) does not match predictions length ({len(predictions)})")
                    
                # Align target with predictions
                target = target.reindex(predictions.index).sort_index()
                if target.isna().any():
                    raise ValueError("Missing target values after alignment")
                    
                # Process meta_model if provided
                if meta_model is not None:
                    info("Processing meta model predictions")
                    if len(meta_model) != len(predictions):
                        raise ValueError("Meta model predictions length mismatch")
                        
                    meta_model = meta_model.reindex(predictions.index).sort_index()
                    if meta_model.isna().any():
                        raise ValueError("Missing meta model values after alignment")
                
                # Calculate correlations
                info("Calculating correlation metrics")
                corrs = self.neutralizer.correlator.numerai_correlation(
                    predictions=predictions,
                    targets=target
                )
                metrics["correlation"] = float(corrs.iloc[0])
                
                info(
                    "Correlation calculation complete",
                    {'correlation': metrics["correlation"]}
                )
                
                # Calculate MMC if meta_model provided
                if meta_model is not None:
                    info("Calculating MMC scores")
                    mmc_scores = self.neutralizer.calculate_mmc(
                        predictions=predictions,
                        meta_model=meta_model,
                        targets=target
                    )
                    metrics["mmc"] = float(mmc_scores.mean())
                    metrics["mmc_per_column"] = mmc_scores.to_dict()
                    
                    info(
                        "MMC calculation complete",
                        {
                            'mmc_mean': metrics["mmc"],
                            'mmc_std': float(mmc_scores.std())
                        }
                    )
                
                # Calculate era-wise metrics
                if era_col is not None:
                    info("Calculating era-wise metrics")
                    era_scores = {}
                    era_col = era_col.reindex(predictions.index).sort_index()
                    
                    for era in era_col.unique():
                        mask = era_col == era
                        era_preds = predictions[mask]
                        era_target = target[mask]
                        
                        if len(era_preds) > 0:
                            era_corrs = self.neutralizer.correlator.numerai_correlation(
                                predictions=era_preds,
                                targets=era_target
                            )
                            era_scores[str(era)] = float(era_corrs.iloc[0])
                    
                    era_values = np.array(list(era_scores.values()))
                    metrics["era_wise"] = {
                        "mean": float(np.mean(era_values)),
                        "std": float(np.std(era_values)),
                        "sharpe": float(np.mean(era_values) / np.std(era_values)) if np.std(era_values) != 0 else 0,
                        "per_era": era_scores
                    }
                    
                    info(
                        "Era-wise calculations complete",
                        {
                            'num_eras': len(era_scores),
                            'era_stats': {
                                'mean': metrics["era_wise"]["mean"],
                                'std': metrics["era_wise"]["std"],
                                'sharpe': metrics["era_wise"]["sharpe"]
                            }
                        }
                    )
                
                # Calculate feature exposure
                info("Calculating feature exposure")
                metrics["feature_exposure"] = self.neutralizer.calculate_feature_exposure(
                    predictions['prediction'],
                    X[self.neutralization_features]
                ).to_dict()
                
                metrics["max_feature_exposure"] = float(max(abs(v) for v in metrics["feature_exposure"].values()))
                
                info(
                    "Feature exposure calculation complete",
                    {
                        'max_exposure': metrics["max_feature_exposure"],
                        'num_features': len(metrics["feature_exposure"])
                    }
                )
                
                # Add diagnostics
                metrics["diagnostics"] = {
                    "prediction_std": float(predictions['prediction'].std()),
                    "prediction_mean": float(predictions['prediction'].mean()),
                    "num_rows": int(len(predictions)),
                    "nulls": int(predictions.isna().sum().sum())
                }
                
                info(
                    "All metrics calculated successfully",
                    {'diagnostics': metrics["diagnostics"]}
                )
                
            return predictions, metrics
            
        except Exception as e:
            error(
                "Error in predict_with_metrics",
                {
                    'action': 'predict_metrics_error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            raise

    @log_performance
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance if model supports it."""
        try:
            info(
                "Getting feature importance",
                {
                    'action': 'feature_importance',
                    'model_type': type(self.model).__name__
                }
            )
            
            if hasattr(self.model, 'feature_importances_'):
                importance = pd.Series(
                    self.model.feature_importances_,
                    index=self.features
                ).sort_values(ascending=False)
                
                info(
                    "Feature importance calculation complete",
                    {
                        'method': 'feature_importances_',
                        'top_features': importance.head().to_dict()
                    }
                )
                return importance
                
            elif hasattr(self.model, 'coef_'):
                importance = pd.Series(
                    abs(self.model.coef_),
                    index=self.features
                ).sort_values(ascending=False)
                
                info(
                    "Feature importance calculation complete",
                    {
                        'method': 'coef_',
                        'top_features': importance.head().to_dict()
                    }
                )
                return importance
                
            else:
                warning(
                    "Model doesn't support feature importance",
                    {
                        'reason': 'unsupported_model'
                    }
                )
                return None
                
        except Exception as e:
            error(
                "Error getting feature importance",
                {
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            return None