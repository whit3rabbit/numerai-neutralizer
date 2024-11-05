import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from numerai_neutralizer.utils.logging import logger
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
        """Initialize NumeraiModel.
        
        Args:
            model: Any model with a predict method
            neutralizer: NumeraiNeutralizer instance
            features: List of feature names to use
            neutralization_features: Optional list of features to neutralize against
            proportion: Strength of neutralization (0-1)
            
        Raises:
            ValueError: If proportion not in [0,1] or features list is empty
        """
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
        
        logger.info(
            f"Initialized NumeraiModel with {len(features)} features and "
            f"{len(neutralization_features) if neutralization_features else len(features)} "
            "neutralization features"
        )
        
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate neutralized predictions.
        
        Args:
            X: Feature data
            
        Returns:
            DataFrame with neutralized and ranked predictions
            
        Raises:
            ValueError: If required features are missing
        """
        try:
            logger.info("Generating predictions...")
            validate_data(X, "feature data")
            
            # Validate features
            missing_features = set(self.features) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
                
            # Generate predictions
            predictions = pd.DataFrame(
                self.model.predict(X[self.features]),
                index=X.index,
                columns=["prediction"]
            )
            
            # Apply neutralization if features specified
            if self.neutralization_features:
                missing_neutral = set(self.neutralization_features) - set(X.columns)
                if missing_neutral:
                    raise ValueError(f"Missing neutralization features: {missing_neutral}")
                    
                logger.info("Applying feature neutralization...")
                predictions = self.neutralizer.neutralize(
                    predictions,
                    X[self.neutralization_features],
                    self.proportion
                )
                
            # Rank transform predictions
            logger.info("Ranking predictions...")
            ranked_preds = predictions.rank(pct=True)
            
            # Validate final predictions
            if ranked_preds.isna().any().any():
                raise ValueError("NaN values in final predictions")
                
            return ranked_preds
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise

    def predict_with_metrics(
        self,
        X: pd.DataFrame,
        meta_model: Optional[pd.Series] = None,
        target: Optional[pd.Series] = None,
        era_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate predictions with comprehensive performance metrics.
        
        Args:
            X: Feature data
            meta_model: Optional meta model predictions for MMC calculation
            target: Optional target values for correlation metrics
            era_col: Optional era column name for era-wise metrics
            
        Returns:
            Tuple of (predictions DataFrame, metrics dictionary)
            
        Raises:
            ValueError: If data alignment fails or inputs are invalid
        """
        try:
            # Generate predictions
            predictions = self.predict(X)
            metrics: Dict[str, Any] = {}
            
            if target is not None:
                # Align target with predictions
                target = target.reindex(predictions.index)
                if target.isna().any():
                    raise ValueError("Missing target values after alignment")
                
                logger.info("Calculating correlation metrics...")
                
                # Calculate overall correlation
                metrics["correlation"] = self.neutralizer.correlator.numerai_correlation(
                    predictions, target
                )
                
                # Calculate era-wise metrics if era_col provided
                if era_col is not None and era_col in X.columns:
                    era_metrics = (
                        X.groupby(era_col)
                        .apply(lambda x: self.neutralizer.correlator.numerai_correlation(
                            predictions.loc[x.index], 
                            target.loc[x.index]
                        ))
                    )
                    metrics["era_wise"] = {
                        "mean": era_metrics.mean(),
                        "std": era_metrics.std(),
                        "sharpe": era_metrics.mean() / era_metrics.std() if era_metrics.std() != 0 else 0,
                        "per_era": era_metrics.to_dict()
                    }
                
                # Calculate MMC if meta_model provided
                if meta_model is not None:
                    logger.info("Calculating MMC...")
                    meta_model = meta_model.reindex(predictions.index)
                    if meta_model.isna().any():
                        raise ValueError("Missing meta model values after alignment")
                        
                    metrics["mmc"] = self.neutralizer.calculate_mmc(
                        predictions, meta_model, target
                    )
                    
                # Calculate feature exposure
                logger.info("Calculating feature exposure...")
                metrics["feature_exposure"] = self.neutralizer.calculate_feature_exposure(
                    predictions["prediction"],
                    X[self.neutralization_features]
                )
                
                metrics["max_feature_exposure"] = metrics["feature_exposure"].max()
                
                # Add diagnostics
                metrics["diagnostics"] = {
                    "prediction_std": predictions.std().to_dict(),
                    "prediction_mean": predictions.mean().to_dict(),
                    "num_rows": len(predictions),
                    "nulls": predictions.isna().sum().to_dict()
                }
                
            return predictions, metrics
            
        except Exception as e:
            logger.error(f"Error in predict_with_metrics: {str(e)}")
            raise
            
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance if model supports it.
        
        Returns:
            Series of feature importance scores if available
        """
        try:
            if hasattr(self.model, 'feature_importances_'):
                return pd.Series(
                    self.model.feature_importances_,
                    index=self.features
                ).sort_values(ascending=False)
            elif hasattr(self.model, 'coef_'):
                return pd.Series(
                    abs(self.model.coef_),
                    index=self.features
                ).sort_values(ascending=False)
            else:
                logger.warning("Model doesn't support feature importance")
                return None
                
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return None