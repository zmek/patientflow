from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional, Tuple
from sklearn.pipeline import Pipeline


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
    valid_logloss: float
    feature_names: Union[List[str], List[float]]
    feature_importances: List[float]
    metadata: Dict[str, Any]
    balance_info: Dict[str, Union[bool, int, float]] = field(default_factory=dict)


@dataclass
class TrainedModel:
    """Container for trained model artifacts and their associated metrics."""

    metrics: TrainingResults
    pipeline: Optional[Pipeline] = None
    calibrated_pipeline: Optional[Pipeline] = None
