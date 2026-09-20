"""Evaluation: metrics, paired statistics, gates, run records and the driver."""

from drososense.evaluation.gates import GateEvaluation, GateEvaluator, GateExpressionError
from drososense.evaluation.metrics import (
    EMPTY_CLASS_POLICY,
    classification_metrics,
    regression_metrics,
)
from drososense.evaluation.results import RunRecord, aggregate_records, load_records, write_record
from drososense.evaluation.stats import (
    InsufficientDataError,
    PairedResult,
    PairedSpec,
    fold_cluster_bootstrap,
    holm_correction,
    paired_test,
)

__all__ = [
    "EMPTY_CLASS_POLICY",
    "GateEvaluation",
    "GateEvaluator",
    "GateExpressionError",
    "InsufficientDataError",
    "PairedResult",
    "PairedSpec",
    "RunRecord",
    "aggregate_records",
    "classification_metrics",
    "fold_cluster_bootstrap",
    "holm_correction",
    "load_records",
    "paired_test",
    "regression_metrics",
    "write_record",
]
