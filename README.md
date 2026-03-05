# cc_features

A C++ library for computing point cloud geometric features.

This library was created to meet a specific project need for calculating geometric features (such as planarity, verticality, curvature, etc.). The computational logic is modeled after the [CloudCompare (CCCoreLib)](https://github.com/CloudCompare/CCCoreLib) implementation.

## Key Features
- **Logic Compliance**: Aligns as much as possible with CloudCompare's algorithm details.
- **Project Oriented**: Designed for easy integration with Python, utilizing OpenMP for acceleration.
- **Simple & Efficient**: Built on Eigen and nanoflann, aiming for solid performance in practical use cases.

---

## Supported Features

| Name | Formula | CloudCompare equivalent |
|---|---|---|
| `Verticality` | `1 − |n · ẑ|` | Verticality |
| `Omnivariance` | `(λ₁·λ₂·λ₃)^(1/3)` | Omnivariance |
| `SurfaceVariation` | `λ₃ / (λ₁+λ₂+λ₃)` | Surface variation (curvature) |
| `Planarity` | `(λ₂−λ₃) / λ₁` | Planarity |

Where **λ₁ ≥ λ₂ ≥ λ₃** are the eigenvalues of the biased 3 × 3 covariance matrix of the neighborhood.

---

## Implementation Details

To ensure consistency with CloudCompare, the following implementation details were considered:

- **Minimum Neighbors**: Returns NaN if the neighborhood has fewer than 3 points.
- **Covariance Matrix**: Uses biased covariance (divided by N, instead of N-1), matching CC's behavior.
- **Eigenvalue Handling**: Sorted in descending order ($\lambda_1 \ge \lambda_2 \ge \lambda_3$) with negative values clamped to zero.
- **Numerical Stability**: Guard thresholds (1e-12) are used for potential zero-denominator cases.

---

## Installation

### Prerequisites
Install Eigen3:

#### Linux (Ubuntu / Debian)
```bash
conda install main::eigen
```

#### Windows
Install Eigen3 via [vcpkg](https://vcpkg.io/):

```powershell
vcpkg install eigen3:x64-windows
```

OpenMP is available by default in MSVC.

### Install from Source

```bash
# 1. Clone
git clone https://github.com/Si-Xiaoming/cc_feature
cd cc_features

# 2. Install
pip install .
```

On **Windows** with vcpkg:

```bash
pip install . --config-settings `
  cmake.define.CMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
```

---

## Usage Examples

### Quick Start

```python
import numpy as np
import cc_features

# Load your point cloud — shape (N, 3), dtype float64
pts = np.load("my_cloud.npy").astype(np.float64)

result = cc_features.compute_features(pts, [
    ("Verticality",      0.2),   # → dict key "Verticality_0.2"
    ("Omnivariance",     0.2),   
    ("SurfaceVariation", 0.2),
    ("Planarity",        0.5),   
])

print(result["Verticality_0.2"])   # np.ndarray, shape (N,)
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

## How to Add New Features

If you need to add custom geometric features, follow these steps:

1. Add a new value to the `FeatureType` enum in `include/cc_features/features.hpp`.
2. Add the corresponding name mapping in `feature_name()`.
3. Implement the calculation logic in `compute_feature()` within `src/feature_extractor.cpp`.
4. Add a parsing branch in `parse_feature_name()` in `python/bindings.cpp`.
5. Add a regression test in `tests/test_features.py`.

---

## License

MIT — see [LICENSE](LICENSE).
