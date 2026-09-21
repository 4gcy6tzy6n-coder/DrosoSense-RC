"""Data layer: schemas, loading, specimen-level splitting, scaling, windowing.

The invariants enforced throughout this package are:

    a specimen never appears in more than one of train / validation / test,
    and a window never contains timesteps from two specimens or from two
    acquisition sessions.

``drososense.data.leakage`` contains active audits for all of them.
"""

from drososense.data.acquisition import extract_remote_archive_members
from drososense.data.leakage import LeakageError, audit_fold, audit_scaler, audit_windows
from drososense.data.remote_zip import RemoteZip, RemoteZipEntry
from drososense.data.scaling import Standardizer
from drososense.data.schema import Dataset, DatasetSchema, SpecimenSource
from drososense.data.splits import Fold, make_folds
from drososense.data.windowing import WindowSet, make_windows

__all__ = [
    "Dataset",
    "DatasetSchema",
    "Fold",
    "LeakageError",
    "RemoteZip",
    "RemoteZipEntry",
    "SpecimenSource",
    "Standardizer",
    "WindowSet",
    "audit_fold",
    "audit_scaler",
    "audit_windows",
    "extract_remote_archive_members",
    "make_folds",
    "make_windows",
]
