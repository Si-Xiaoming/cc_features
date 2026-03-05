#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sstream>
#include <string>

// 辅助函数：将浮点数转为干净的字符串（去除多余的 0）
std::string formatRadius(double radius) {
    std::ostringstream oss;
    oss << radius;
    return oss.str();
}

#include "cc_features/feature_extractor.hpp"
#include "cc_features/features.hpp"

#include <stdexcept>

#include <vector>

namespace py = pybind11;
using namespace cc_features;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Validate numpy array and return (ptr, n_points).
static std::pair<const double*, std::size_t>
validate_points(const py::array_t<double,
                py::array::c_style | py::array::forcecast>& arr)
{
    auto buf = arr.request();
    std::size_t n = 0;

    if (buf.ndim == 2) {
        if (buf.shape[1] != 3)
            throw std::invalid_argument(
                "points must have shape (N, 3), got shape ("
                + std::to_string(buf.shape[0]) + ", "
                + std::to_string(buf.shape[1]) + ")");
        n = static_cast<std::size_t>(buf.shape[0]);
    } else if (buf.ndim == 1) {
        if (buf.shape[0] % 3 != 0)
            throw std::invalid_argument(
                "Flat points array length must be a multiple of 3.");
        n = static_cast<std::size_t>(buf.shape[0]) / 3;
    } else {
        throw std::invalid_argument(
            "points must be a 1-D (N*3,) or 2-D (N, 3) float64 array.");
    }

    return {static_cast<const double*>(buf.ptr), n};
}

/// Parse feature name string → FeatureType enum.
static FeatureType parse_feature_name(const std::string& name)
{
    if (name == "Verticality")      return FeatureType::Verticality;
    if (name == "Omnivariance")     return FeatureType::Omnivariance;
    if (name == "SurfaceVariation") return FeatureType::SurfaceVariation;
    if (name == "Planarity")        return FeatureType::Planarity;
    throw std::invalid_argument(
        "Unknown feature name: '" + name +
        "'. Supported: Verticality, Omnivariance, SurfaceVariation, Planarity");
}

/// Convert raw results + tasks into a Python dict {col_name: np.ndarray}.
static py::dict pack_results(
    const std::vector<std::vector<double>>& raw,
    const std::vector<TaskConfig>&          tasks,
    std::size_t                             n_pts)
{
    py::dict out;
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        // Column name: "Verticality_0.2m"
        std::string key = feature_name(tasks[i].feature)
                          + "_" + std::to_string(tasks[i].radius);

        py::array_t<double> arr(static_cast<py::ssize_t>(n_pts));
        auto ab = arr.request();
        std::copy(raw[i].begin(), raw[i].end(),
                  static_cast<double*>(ab.ptr));
        out[key.c_str()] = arr;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Module
// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(_cc_features, m) {
    m.doc() =
        "High-performance point cloud PCA feature extraction.\n"
        "100 % mathematically compatible with CloudCompare / CCCoreLib.";

    // ── FeatureType enum ──────────────────────────────────────────────────────
    py::enum_<FeatureType>(m, "FeatureType",
        "PCA-derived geometric feature types.")
        .value("Verticality",      FeatureType::Verticality,
               "1 - |normal · ẑ|")
        .value("Omnivariance",     FeatureType::Omnivariance,
               "(λ1·λ2·λ3)^(1/3)")
        .value("SurfaceVariation", FeatureType::SurfaceVariation,
               "λ3 / (λ1+λ2+λ3)")
        .value("Planarity",        FeatureType::Planarity,
               "(λ2−λ3) / λ1")
        .export_values();

    // ── TaskConfig ────────────────────────────────────────────────────────────
    py::class_<TaskConfig>(m, "TaskConfig",
        "One computation task: a (FeatureType, radius) pair.")
        .def(py::init<FeatureType, double>(),
             py::arg("feature"), py::arg("radius"))
        .def_readwrite("feature", &TaskConfig::feature)
        .def_readwrite("radius",  &TaskConfig::radius)
        .def("__repr__", [](const TaskConfig& t) {
            return "TaskConfig(feature=" + feature_name(t.feature)
                   + ", radius=" + std::to_string(t.radius) + ")";
        });

    // ── FeatureExtractor ──────────────────────────────────────────────────────
    py::class_<FeatureExtractor>(m, "FeatureExtractor",
        R"doc(
Builds a KD-Tree once, then computes any combination of PCA features.

Parameters
----------
points : np.ndarray
    Shape (N, 3) or flat (N*3,), dtype float64.
    The array must remain alive for the lifetime of the extractor.
        )doc")
        .def(py::init([](py::array_t<double,
                          py::array::c_style | py::array::forcecast> pts) {
                 auto [raw, n] = validate_points(pts);
                 return new FeatureExtractor(raw, n);
             }),
             py::arg("points"),
             py::keep_alive<1, 2>())   // keep numpy array alive with extractor
        .def("compute",
             [](const FeatureExtractor& self,
                const std::vector<TaskConfig>& tasks) -> py::dict {
                 auto raw = self.compute(tasks);
                 return pack_results(raw, tasks, self.point_count());
             },
             py::arg("tasks"),
             R"doc(
Compute all requested features.

Parameters
----------
tasks : list[TaskConfig]

Returns
-------
dict[str, np.ndarray]
    Keys are "<FeatureName>_<radius>m", e.g. "Verticality_0.2m".
    Values are float64 arrays of length N; NaN for invalid points.
             )doc")
        .def("point_count", &FeatureExtractor::point_count,
             "Number of points in the KD-Tree.");

    // ── compute_features() convenience function ───────────────────────────────
    m.def("compute_features",
        [](py::array_t<double,
            py::array::c_style | py::array::forcecast> points,
           const std::vector<std::pair<std::string, double>>& task_specs)
              -> py::dict
        {
            auto [raw, n] = validate_points(points);

            std::vector<TaskConfig> tasks;
            tasks.reserve(task_specs.size());
            for (auto& [name, radius] : task_specs)
                tasks.emplace_back(parse_feature_name(name), radius);

            FeatureExtractor extractor(raw, n);
            auto results = extractor.compute(tasks);
            return pack_results(results, tasks, n);
        },
        py::arg("points"),
        py::arg("tasks"),
        R"doc(
Compute multiple PCA-based features in a single pass (convenience wrapper).

Parameters
----------
points : np.ndarray
    Shape (N, 3) or (N*3,), dtype float64.
tasks : list[tuple[str, float]]
    Each tuple is (feature_name, radius_in_metres).
    Supported names: Verticality, Omnivariance, SurfaceVariation, Planarity.
    Tasks sharing the same radius share one KD-Tree query and one PCA.

Returns
-------
dict[str, np.ndarray]
    Keys: "<FeatureName>_<radius>m".  Values: float64 arrays length N.
    NaN is written for points with fewer than 3 neighbours.

Examples
--------
>>> import numpy as np
>>> import cc_features
>>> pts = np.random.rand(1_000_000, 3).astype(np.float64)
>>> result = cc_features.compute_features(pts, [
...     ("Verticality",      0.2),
...     ("Omnivariance",     0.2),   # same radius → shared query + PCA
...     ("SurfaceVariation", 0.2),
...     ("Planarity",        0.5),
... ])
>>> result["Verticality_0.2m"].shape
(1000000,)
        )doc");
}
