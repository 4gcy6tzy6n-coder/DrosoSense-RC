"""Evaluation: metrics, run records, and the benchmark driver."""

from drososense.evaluation.metrics import classification_metrics, regression_metrics
from drososense.evaluation.results import RunRecord, aggregate_records, load_records, write_record

__all__ = [
    "RunRecord",
    "aggregate_records",
    "classification_metrics",
    "load_records",
    "regression_metrics",
    "write_record",
]
