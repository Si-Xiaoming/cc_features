"""
cc_features
===========
High-performance point cloud PCA feature extraction.
100 % mathematically compatible with CloudCompare / CCCoreLib.

Quick start
-----------
>>> import numpy as np
>>> import cc_features
>>>
>>> pts = np.random.rand(500_000, 3).astype(np.float64)
>>>
>>> result = cc_features.compute_features(pts, [
...     ("Verticality",      0.2),
...     ("Omnivariance",     0.2),   # same radius → one KD-Tree query + PCA
...     ("SurfaceVariation", 0.2),
...     ("Planarity",        0.5),
... ])
>>>
>>> result["Verticality_0.2m"]   # np.ndarray(500_000,), NaN for edge points
"""

from ._cc_features import (   # noqa: F401
    FeatureType,
    TaskConfig,
    FeatureExtractor,
    compute_features,
)

__version__ = "1.0.0"
__all__ = [
    "FeatureType",
    "FeatureExtractor",
    "TaskConfig",
    "compute_features",
]
