from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional, Tuple
from sklearn.pipeline import Pipeline

@dataclass
class HyperParameterTrial:
    """Container for a single hyperparameter tuning trial."""
    parameters: Dict[str, Any]
    cv_results: Dict[str, float]
    
@dataclass
class FoldResults:
    """Store evaluation metrics for a single fold."""
    auc: float
    logloss: float
    auprc: float


@dataclass
class TrainingResults:
    """Store comprehensive evaluation metrics and metadata from model training."""
    prediction_time: Tuple[int, int]
    training_info: Dict[str, Any] = field(default_factory=dict)
    calibration_info: Dict[str, Any] = field(default_factory=dict)
    test_results: Dict[str, float] = field(default_factory=dict)
    balance_info: Dict[str, Union[bool, int, float]] = field(default_factory=dict)


@dataclass
class TrainedClassifier:
    """Container for trained model artifacts and their associated information."""
    training_results: TrainingResults
    pipeline: Optional[Pipeline] = None
    calibrated_pipeline: Optional[Pipeline] = None