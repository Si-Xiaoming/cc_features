#pragma once
#include <cstddef>

namespace cc_features {

/**
 * Zero-copy nanoflann adapter over a raw interleaved double* XYZ buffer.
 * Layout:  [x0, y0, z0,  x1, y1, z1,  ...,  xN-1, yN-1, zN-1]
 */
struct PointCloud {
    const double* data;
    std::size_t   count;

    PointCloud() : data(nullptr), count(0) {}
    PointCloud(const double* pts, std::size_t n) : data(pts), count(n) {}

    /* ---- nanoflann required interface ---- */
    inline std::size_t kdtree_get_point_count() const { return count; }

    inline double kdtree_get_pt(const std::size_t idx,
                                const std::size_t dim) const {
        return data[idx * 3 + dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

} // namespace cc_features
