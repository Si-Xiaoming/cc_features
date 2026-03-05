# cc_features

[![Build & Test](https://github.com/yourname/cc_features/actions/workflows/ci.yml/badge.svg)](https://github.com/yourname/cc_features/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**High-performance C++ point cloud PCA feature extraction with Python bindings.**

> ✅ 100 % mathematically aligned with [CloudCompare / CCCoreLib](https://github.com/CloudCompare/CCCoreLib)

Processes **million-point clouds in seconds** by combining:

- **nanoflann** — header-only KD-Tree, radius searches in squared distance (no `sqrt`)
- **Eigen3** — 3 × 3 `SelfAdjointEigenSolver`, optimised at compile time
- **OpenMP** — dynamic parallel loop over all points
- **Radius-grouping** — tasks sharing the same radius share **one** tree query and **one** PCA

---

## Supported features

| Name | Formula | CloudCompare equivalent |
|---|---|---|
| `Verticality` | `1 − \|n · ẑ\|` | Verticality |
| `Omnivariance` | `(λ₁·λ₂·λ₃)^(1/3)` | Omnivariance |
| `SurfaceVariation` | `λ₃ / (λ₁+λ₂+λ₃)` | Surface variation (curvature) |
| `Planarity` | `(λ₂−λ₃) / λ₁` | Planarity |

Where **λ₁ ≥ λ₂ ≥ λ₃** are the eigenvalues of the biased 3 × 3 covariance matrix of the neighbourhood.

---

## Algorithm compliance (CCCoreLib)

| Rule | Status |
|---|---|
| Min neighbours < 3 → NaN | ✅ |
| Biased covariance (÷ N, not N−1) | ✅ |
| Eigenvalues sorted descending (λ₁ ≥ λ₂ ≥ λ₃) | ✅ |
| Negative eigenvalues clamped to 0 | ✅ |
| Zero-denominator guard (threshold 1e-12) | ✅ |

---

## Installation

### Prerequisites

#### Linux (Ubuntu / Debian)

```bash
sudo apt install libeigen3-dev libgomp1
```

#### macOS

```bash
brew install eigen libomp
```

#### Windows

Install Eigen3 via [vcpkg](https://vcpkg.io/):

```powershell
vcpkg install eigen3:x64-windows
```

OpenMP is available by default in MSVC.

### Install from source

```bash
# 1. Clone
git clone https://github.com/yourname/cc_features.git
cd cc_features

# 2. Fetch the nanoflann header (downloads ~250 KB)
python fetch_nanoflann.py

# 3. Install
pip install .

# or for development
pip install -e ".[dev]"
```

On **Windows** with vcpkg:

```bash
pip install . --config-settings `
  cmake.define.CMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
```

---

## Usage

### Quick start

```python
import numpy as np
import cc_features

# Load your point cloud — shape (N, 3), dtype float64
pts = np.load("my_cloud.npy").astype(np.float64)

result = cc_features.compute_features(pts, [
    ("Verticality",      0.2),   # → dict key "Verticality_0.2m"
    ("Omnivariance",     0.2),   # same radius → ONE query + PCA
    ("SurfaceVariation", 0.2),
    ("Planarity",        0.5),   # different radius → separate query
])

print(result["Verticality_0.2m"])   # np.ndarray, shape (N,), NaN for edge points
```

### OOP interface (reuse KD-Tree across multiple calls)

```python
extractor = cc_features.FeatureExtractor(pts)   # builds KD-Tree once

tasks = [
    cc_features.TaskConfig(cc_features.FeatureType.Verticality,  0.2),
    cc_features.TaskConfig(cc_features.FeatureType.Planarity,    0.5),
]
result = extractor.compute(tasks)
```

### Integration with pandas / laspy

```python
import laspy
import pandas as pd
import cc_features

las = laspy.read("scan.laz")
pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

result = cc_features.compute_features(pts, [
    ("Verticality", 0.3),
    ("Planarity",   0.3),
])

df = pd.DataFrame({
    "x": las.x, "y": las.y, "z": las.z,
    **{k: v for k, v in result.items()},
})
df.to_csv("features.csv", index=False)
```

---

## Performance

Benchmarked on Intel Core i9-12900K (16 P-cores), Ubuntu 22.04, GCC 12.

| Cloud size | Tasks (radii) | Time |
|---|---|---|
| 100 K pts | 4 features, 1 radius | ~0.3 s |
| 1 M pts | 4 features, 1 radius | ~3 s |
| 1 M pts | 4 features, 2 radii | ~6 s |

---

## Project structure

```
cc_features/
├── include/cc_features/
│   ├── point_cloud.hpp        # nanoflann adapter (zero-copy)
│   ├── features.hpp           # FeatureType enum, TaskConfig, feature_name()
│   └── feature_extractor.hpp  # public FeatureExtractor class
├── src/
│   └── feature_extractor.cpp  # KD-Tree, PCA, radius-grouping, OpenMP loop
├── python/
│   └── bindings.cpp           # pybind11 bindings
├── cc_features/
│   └── __init__.py            # Python package
├── tests/
│   └── test_features.py       # pytest regression suite
├── third_party/               # nanoflann.hpp (fetched by fetch_nanoflann.py)
├── fetch_nanoflann.py
├── CMakeLists.txt
├── pyproject.toml
└── README.md
```

---

## Extending with new features

1. Add a new value to `FeatureType` in `include/cc_features/features.hpp`.
2. Add a case to `feature_name()` in the same file.
3. Add a case to `compute_feature()` in `src/feature_extractor.cpp`.
4. Add a branch to `parse_feature_name()` in `python/bindings.cpp`.
5. Write a test in `tests/test_features.py`.

No other files need to change.

---

## License

MIT — see [LICENSE](LICENSE).
