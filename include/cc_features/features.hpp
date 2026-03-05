#pragma once
#include <string>

namespace cc_features {

/**
 * All supported PCA-derived geometric features.
 * Add new values here to extend without changing the public API.
 */
enum class FeatureType : int {
    Verticality      = 0,  // 1 - |normal · ẑ|
    Omnivariance     = 1,  // (λ1·λ2·λ3)^(1/3)
    SurfaceVariation = 2,  // λ3 / (λ1+λ2+λ3)   [Chang curvature]
    Planarity        = 3,  // (λ2−λ3) / λ1
    // ── Future extensions ──────────────────────────────────────────
    // Linearity    = 4,  // (λ1−λ2) / λ1
    // Anisotropy   = 5,  // (λ1−λ3) / λ1
    // Sphericity   = 6,  // λ3 / λ1
    // Eigenentropy = 7,  // -Σ λi·ln(λi)
};

/** One computation task: a (feature, radius) pair. */
struct TaskConfig {
    FeatureType feature;
    double      radius;

    TaskConfig(FeatureType f, double r) : feature(f), radius(r) {}
};

/** Human-readable name used for column labels / dict keys. */
inline std::string feature_name(FeatureType f) {
    switch (f) {
        case FeatureType::Verticality:      return "Verticality";
        case FeatureType::Omnivariance:     return "Omnivariance";
        case FeatureType::SurfaceVariation: return "SurfaceVariation";
        case FeatureType::Planarity:        return "Planarity";
        default:                            return "Unknown";
    }
}

} // namespace cc_features
