from dataclasses import dataclass

@dataclass
class FeatureMetrics:
    """Stores comprehensive metrics for feature analysis."""
    
    mean: float
    std: float
    sharpe: float
    max_drawdown: float
    delta: float
    max_feature_exposure: float
    
    def __post_init__(self):
        """Validate metric values after initialization."""
        assert isinstance(self.mean, float), "mean must be float"
        assert isinstance(self.std, float), "std must be float"
        assert self.std >= 0, "std must be non-negative"
        
    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "mean": self.mean,
            "std": self.std,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "delta": self.delta,
            "max_feature_exposure": self.max_feature_exposure
        }