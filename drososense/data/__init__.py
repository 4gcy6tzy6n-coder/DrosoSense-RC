"""Data layer: schemas, loading, specimen-level splitting, scaling, windowing.

The invariant enforced throughout this package is:

    a specimen never appears in more than one of train / validation / test,
    and a window never contains timesteps from two specimens.

``drososense.data.leakage`` contains active audits for both.
"""

from drososense.data.leakage import LeakageError, audit_fold, audit_scaler, audit_windows
from drososense.data.scaling import Standardizer
from drososense.data.schema import Dataset, DatasetSchema, SpecimenSource
from drososense.data.splits import Fold, make_folds
from drososense.data.windowing import WindowSet, make_windows

__all__ = [
    "Dataset",
    "DatasetSchema",
    "Fold",
    "LeakageError",
    "SpecimenSource",
    "Standardizer",
    "WindowSet",
    "audit_fold",
    "audit_scaler",
    "audit_windows",
    "make_folds",
    "make_windows",
]
