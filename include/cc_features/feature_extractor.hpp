#pragma once
#include "point_cloud.hpp"
#include "features.hpp"
#include <vector>
#include <cstddef>

namespace cc_features {

/**
 * High-performance PCA-based geometric feature extractor.
 *
 * ## Algorithm guarantees (100 % CCCoreLib-compatible)
 *  - Neighbour count < 3  →  NaN  (same threshold as CC)
 *  - Biased covariance matrix  (divide by N, not N−1)
 *  - Eigenvalues sorted descending:  λ1 ≥ λ2 ≥ λ3
 *  - Negative eigenvalues clamped to 0 (float-point guard)
 *  - Zero-denominator guard  (threshold 1e-12)
 *
 * ## Performance
 *  - KD-Tree built once at construction, reused for every task.
 *  - Tasks are grouped by radius internally: one radius search +
 *    one Eigen SelfAdjointEigenSolver per (point, radius group).
 *  - OpenMP dynamic scheduling over points.
 *  - No heap allocation inside the hot loop.
 */
class FeatureExtractor {
public:
    /**
     * @param points   Flat interleaved XYZ array, length = n_points * 3.
     *                 The caller owns this memory; it must remain valid
     *                 for the lifetime of the extractor.
     * @param n_points Number of points.
     */
    explicit FeatureExtractor(const double* points, std::size_t n_points);
    ~FeatureExtractor();

    // Non-copyable (owns the KD-Tree)
    FeatureExtractor(const FeatureExtractor&)            = delete;
    FeatureExtractor& operator=(const FeatureExtractor&) = delete;

    /**
     * Compute all requested features in one pass.
     *
     * @param tasks  List of (FeatureType, radius) pairs.
     *               Tasks sharing the same radius are grouped automatically.
     *
     * @return  One std::vector<double> per task (same order as input).
     *          Each vector has length == n_points.
     *          NaN is written for points where computation is impossible.
     */
    std::vector<std::vector<double>> compute(
        const std::vector<TaskConfig>& tasks) const;

    std::size_t point_count() const noexcept { return n_points_; }

private:
    struct Impl;
    Impl*       impl_;
    std::size_t n_points_;
};

} // namespace cc_features
