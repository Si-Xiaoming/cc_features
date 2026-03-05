#include "cc_features/feature_extractor.hpp"
#include "cc_features/point_cloud.hpp"

// ── Third-party (header-only) ─────────────────────────────────────────────────
#include <nanoflann.hpp>
#include <Eigen/Dense>

// ── Standard library ──────────────────────────────────────────────────────────
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace cc_features {

// ─────────────────────────────────────────────────────────────────────────────
// KD-Tree type
//
// MSVC FIX: The 4th template parameter is the index type used both internally
// and in the ResultItem returned by radiusSearch.  MSVC defaults it to
// uint32_t, which causes a type-mismatch error when the matches vector is
// declared as vector<ResultItem<size_t,double>>.  Pinning to std::size_t
// keeps the types consistent on Linux, Windows and macOS.
// ─────────────────────────────────────────────────────────────────────────────
using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud>,
    PointCloud,
    3,           /*dims*/
    std::size_t  /*index type — must match ResultItem<size_t,…> below*/
>;

// ─────────────────────────────────────────────────────────────────────────────
// PImpl
// ─────────────────────────────────────────────────────────────────────────────
struct FeatureExtractor::Impl {
    PointCloud cloud;
    KDTree     tree;

    Impl(const double* pts, std::size_t n)
        : cloud(pts, n),
          tree(3, cloud,
               nanoflann::KDTreeSingleIndexAdaptorParams(/*leaf_max_size=*/16))
    {
        tree.buildIndex();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Construction / destruction
// ─────────────────────────────────────────────────────────────────────────────
FeatureExtractor::FeatureExtractor(const double* points, std::size_t n_points)
    : impl_(new Impl(points, n_points)), n_points_(n_points)
{}

FeatureExtractor::~FeatureExtractor() { delete impl_; }

// ─────────────────────────────────────────────────────────────────────────────
// PCA helpers — strictly mirroring CCCoreLib/src/Neighbourhood.cpp
// ─────────────────────────────────────────────────────────────────────────────

// CCCoreLib uses std::numeric_limits<double>::epsilon() (≈2.22e-16) as its
// zero-denominator guard, NOT a custom 1e-12.  We match it exactly.
static const double CC_EPS = std::numeric_limits<double>::epsilon();

/**
 * Explicit 3-element insertion sort: sort (e[], evec columns) together
 * so that e[0] ≥ e[1] ≥ e[2]  (DESCENDING).
 *
 * This replicates CCCoreLib's Jacobi<double>::SortEigenValuesAndVectors().
 * Eigen3's SelfAdjointEigenSolver returns eigenvalues ASCENDING and is
 * numerically identical to Jacobi for well-conditioned matrices, but on
 * nearly-degenerate neighbourhoods (λ2 ≈ λ3) the two solvers may produce
 * eigenvectors in different subspace orientations.  The explicit sort after
 * the solve guarantees the same λ1 ≥ λ2 ≥ λ3 ordering as CC regardless.
 */
static inline void sort_descending(double e[3], Eigen::Matrix3d& evec)
{
    // 3-element insertion sort on e[], swapping evec columns in tandem
    for (int i = 1; i < 3; ++i) {
        double   key_e = e[i];
        Eigen::Vector3d key_v = evec.col(i);
        int j = i - 1;
        while (j >= 0 && e[j] < key_e) {
            e[j + 1]         = e[j];
            evec.col(j + 1)  = evec.col(j);
            --j;
        }
        e[j + 1]        = key_e;
        evec.col(j + 1) = key_v;
    }
}

// eigenvalues-only variant (no eigenvector bookkeeping needed)
static inline void sort_descending_vals(double e[3])
{
    for (int i = 1; i < 3; ++i) {
        double key = e[i];
        int j = i - 1;
        while (j >= 0 && e[j] < key) { e[j+1] = e[j]; --j; }
        e[j+1] = key;
    }
}

/**
 * Compute biased 3×3 covariance (÷N) and its eigendecomposition.
 *
 * ── CCCoreLib compliance (Neighbourhood.cpp — exact source-level match) ─────
 *
 *  ROOT CAUSE 1 (search radius) — fixed in the caller via float32 cast of r².
 *
 *  ROOT CAUSE 2 (small-N amplification) — unavoidable; a single boundary point
 *  flip in N=3..5 neighbourhoods can swing Planarity/Verticality by up to ~1.
 *  No formula change can eliminate this — it is an inherent property of the
 *  degenerate PCA subspace when N is tiny.
 *
 *  ROOT CAUSE 3 (float32 centroid truncation) — fixed here:
 *    CC computeGravityCenter():
 *      CCVector3d Psum = Σ P_i   (points are PointCoordinateType = float32)
 *      G = static_cast<PointCoordinateType>(Psum / N)   ← truncated to float32!
 *    CC computeCovarianceMatrix():
 *      CCVector3 dP = *P_i - *G   ← float32 − float32 = float32 difference!
 *      mXX += (double)dP.x * dP.x  ← product promoted to double, but dP is f32
 *    We replicate this exactly: cast coords to float32, centroid to float32,
 *    compute float32 differences, then accumulate products in double.
 *
 *  Other invariants:
 *  ✅ Neighbour count < 3             → return false (caller writes NaN)
 *  ✅ Biased covariance  ÷ N          (not N−1)
 *  ✅ NO eigenvalue clamping          CC never does max(0, λ)
 *  ✅ Descending sort  λ1 ≥ λ2 ≥ λ3  via explicit insertion sort
 *  ✅ Zero-guard = std::numeric_limits<double>::epsilon()  (≈2.22e-16)
 *  ✅ Omnivariance uses pow(..., 1.0/3.0)  (not cbrt)
 */
static bool run_pca(const double*              all_pts,
                    const std::vector<size_t>& indices,
                    double                     e[3],
                    double*                    normal)    // nullable
{
    const int N = static_cast<int>(indices.size());
    if (N < 3) return false;

    // ── ROOT CAUSE 3 FIX: replicate CC's float32 centroid truncation ─────────
    //
    // CC stores every point as PointCoordinateType = float (32-bit).
    // computeGravityCenter accumulates in double, then casts the result back
    // to float32:
    //   G = static_cast<PointCoordinateType>(Psum.x / count)
    //
    // Then computeCovarianceMatrix computes differences as float32:
    //   CCVector3 dP = *P_i - *G          // float32 − float32
    //   mXX += static_cast<double>(dP.x) * dP.x   // product in double
    //
    // We must reproduce all three steps — cast to f32, truncated centroid,
    // f32 differences — to get the same covariance matrix as CC.

    // Step A: accumulate centroid in double (CC uses CCVector3d for the sum)
    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
    for (int i = 0; i < N; ++i) {
        const std::size_t idx = indices[static_cast<std::size_t>(i)];
        // CC reads float32 coords; simulating the float32→double promotion
        sum_x += static_cast<double>(static_cast<float>(all_pts[idx * 3 + 0]));
        sum_y += static_cast<double>(static_cast<float>(all_pts[idx * 3 + 1]));
        sum_z += static_cast<double>(static_cast<float>(all_pts[idx * 3 + 2]));
    }
    const double inv_N = 1.0 / static_cast<double>(N);

    // Step B: truncate centroid to float32 (CC's setGravityCenter cast)
    const float gx_f32 = static_cast<float>(sum_x * inv_N);
    const float gy_f32 = static_cast<float>(sum_y * inv_N);
    const float gz_f32 = static_cast<float>(sum_z * inv_N);

    // Step C: float32 differences, products accumulated in double
    // Mirrors: CCVector3 dP = *P_i - *G  (float32 subtraction)
    //          mXX += (double)dP.x * dP.x
    double cov00=0, cov01=0, cov02=0,
                    cov11=0, cov12=0,
                              cov22=0;
    for (int i = 0; i < N; ++i) {
        const std::size_t idx = indices[static_cast<std::size_t>(i)];
        // cast point coords to float32, then compute float32 differences
        const float px = static_cast<float>(all_pts[idx * 3 + 0]);
        const float py = static_cast<float>(all_pts[idx * 3 + 1]);
        const float pz = static_cast<float>(all_pts[idx * 3 + 2]);
        const double dx = static_cast<double>(px - gx_f32);   // float32 diff
        const double dy = static_cast<double>(py - gy_f32);
        const double dz = static_cast<double>(pz - gz_f32);
        cov00 += dx*dx;  cov01 += dx*dy;  cov02 += dx*dz;
                         cov11 += dy*dy;  cov12 += dy*dz;
                                          cov22 += dz*dz;
    }
    cov00*=inv_N; cov01*=inv_N; cov02*=inv_N;
    cov11*=inv_N; cov12*=inv_N; cov22*=inv_N;

    // ── Eigen3 SelfAdjointEigenSolver  (always compute eigenvectors so we ────
    //    can apply the explicit descending sort on both vals AND vecs)         
    Eigen::Matrix3d M;
    M << cov00, cov01, cov02,
         cov01, cov11, cov12,
         cov02, cov12, cov22;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(M,
        Eigen::ComputeEigenvectors);
    if (solver.info() != Eigen::Success) return false;

    // Eigen returns ASCENDING order; copy into mutable arrays for the sort.
    const auto& ev = solver.eigenvalues();
    Eigen::Matrix3d evec = solver.eigenvectors();   // mutable copy

    // ── FIX: NO clamping — CCCoreLib never does max(0, λ) ────────────────────
    e[0] = ev(2);   // tentatively ascending→descending (will be re-sorted)
    e[1] = ev(1);
    e[2] = ev(0);
    // Mirror the column swap in evec to match the initial reversal
    {
        Eigen::Vector3d tmp = evec.col(0);
        evec.col(0) = evec.col(2);
        evec.col(2) = tmp;
    }

    // Explicit insertion sort — mirrors Jacobi::SortEigenValuesAndVectors
    sort_descending(e, evec);

    if (normal != nullptr) {
        // Normal = eigenvector of SMALLEST eigenvalue → column 2 after sort
        normal[0] = evec(0, 2);
        normal[1] = evec(1, 2);
        normal[2] = evec(2, 2);
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature formulas  — exact port of CCCoreLib Neighbourhood::computeFeature()
// ─────────────────────────────────────────────────────────────────────────────
static inline double compute_feature(FeatureType  ft,
                                     const double e[3],
                                     const double normal[3])
{
    switch (ft) {
        case FeatureType::Verticality: {
            // CC: value = 1.0 - std::abs(Z.dot(e3))
            return 1.0 - std::abs(normal[2]);
        }
        case FeatureType::Omnivariance: {
            // CC: value = pow(l1 * l2 * l3, 1.0/3.0)
            // ⚠️  pow() returns NaN for negative products (unlike cbrt).
            //     This matches CC's behaviour on degenerate neighbourhoods
            //     where floating-point noise makes λ3 < 0 (no clamping).
            return std::pow(e[0] * e[1] * e[2], 1.0 / 3.0);
        }
        case FeatureType::SurfaceVariation: {
            // CC: if (std::abs(sum) > epsilon)  value = l3 / sum
            const double sum = e[0] + e[1] + e[2];
            return (std::abs(sum) > CC_EPS) ? (e[2] / sum) : 0.0;
        }
        case FeatureType::Planarity: {
            // CC: if (std::abs(l1) > epsilon)  value = (l2 - l3) / l1
            return (std::abs(e[0]) > CC_EPS) ? ((e[1] - e[2]) / e[0]) : 0.0;
        }
        default:
            return std::numeric_limits<double>::quiet_NaN();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public compute()
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::vector<double>>
FeatureExtractor::compute(const std::vector<TaskConfig>& tasks) const
{
    const std::size_t npts   = n_points_;
    const std::size_t ntasks = tasks.size();
    const double      NAN_VAL = std::numeric_limits<double>::quiet_NaN();

    // Allocate output — all NaN by default
    std::vector<std::vector<double>> results(ntasks,
        std::vector<double>(npts, NAN_VAL));

    if (ntasks == 0 || npts == 0) return results;

    // ── Group tasks by radius ─────────────────────────────────────────────────
    // Ordered map so we always process radii in ascending order (cache-friendly).
    // Value: list of (task_index, FeatureType)
    using GroupEntry = std::pair<std::size_t, FeatureType>;
    std::map<double, std::vector<GroupEntry>> groups;
    for (std::size_t ti = 0; ti < ntasks; ++ti)
        groups[tasks[ti].radius].emplace_back(ti, tasks[ti].feature);

    // Pre-compute which radius groups need eigenvectors (Verticality only)
    std::map<double, bool> needs_evec_map;
    for (auto& [r, grp] : groups) {
        bool need = false;
        for (auto& [ti, ft] : grp)
            if (ft == FeatureType::Verticality) { need = true; break; }
        needs_evec_map[r] = need;
    }

    const double* pts = impl_->cloud.data;

    // ── Parallel loop over points ─────────────────────────────────────────────
    // Each thread reuses its own scratch vectors (declared inside the loop
    // so the compiler keeps them on the stack / TLS).
    //
    // MSVC FIX: OpenMP requires the loop variable to have a *signed* integral
    // type (ISO OpenMP §2.9.2).  MSVC enforces this strictly and rejects
    // std::size_t (unsigned).  We use std::ptrdiff_t and cast back to
    // std::size_t for indexing — zero overhead, correct on all platforms.
    const std::ptrdiff_t npts_signed = static_cast<std::ptrdiff_t>(npts);

#pragma omp parallel for schedule(dynamic, 256) default(none) \
        shared(results, groups, needs_evec_map, pts, npts_signed, NAN_VAL)
    for (std::ptrdiff_t spi = 0; spi < npts_signed; ++spi) {
        const std::size_t pi = static_cast<std::size_t>(spi);

        const double query[3] = {
            pts[pi * 3 + 0],
            pts[pi * 3 + 1],
            pts[pi * 3 + 2]
        };

        // Thread-local scratch — reused across radius groups for this point
        std::vector<nanoflann::ResultItem<std::size_t, double>> matches;
        std::vector<std::size_t> indices;

        for (auto& [radius, group] : groups) {

            // ── One radius search per (point, radius) ─────────────────────────
            // CC FIX: CCCoreLib stores point coordinates as PointCoordinateType
            // = float (32-bit by default).  Its spherical search computes:
            //   squareRadius = static_cast<double>(radius) * radius
            // where 'radius' is already a float32 value.
            // We must cast to float first so the r^2 threshold matches CC exactly,
            // otherwise the ~6e-9 m shell of boundary points is handled differently,
            // which for tiny neighbourhoods (N≈3) can flip features by up to ~0.4.
            const float  radius_f32 = static_cast<float>(radius);
            const double r2_cc      = static_cast<double>(radius_f32)
                                      * static_cast<double>(radius_f32);

            matches.clear();
            nanoflann::SearchParameters params;
            params.sorted = false;   // order irrelevant for covariance
            impl_->tree.radiusSearch(query, r2_cc, matches, params);

            // Copy hit indices into flat vector
            indices.resize(matches.size());
            for (std::size_t k = 0; k < matches.size(); ++k)
                indices[k] = matches[k].first;

            // ── One PCA per (point, radius) ───────────────────────────────────
            double e[3];
            double normal[3] = {0.0, 0.0, 1.0};   // fallback (unused if ok==false)
            const bool need_ev = needs_evec_map.at(radius);

            const bool ok = run_pca(pts, indices, e, need_ev ? normal : nullptr);
            if (!ok) continue;   // leave NaN in results

            // ── Write all features for this (point, radius) ───────────────────
            for (auto& [ti, ft] : group)
                results[ti][pi] = compute_feature(ft, e, normal);
        }
    }

    return results;
}

} // namespace cc_features
