#include "cc_features/feature_extractor.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace cc_features;

int main() {
    std::cout << "--- cc_features C++ Debug Session ---" << std::endl;

    // 1. Create a dummy point cloud (a small 2x2x2 cube corner + some random points)
    std::vector<double> points = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        0.5, 0.5, 0.5,
        0.1, 0.1, 0.1,
        0.9, 0.9, 0.9,
        0.2, 0.8, 0.2
    };
    std::size_t n_points = points.size() / 3;

    try {
        // 2. Initialize FeatureExtractor
        std::cout << "[Info] Building KD-Tree for " << n_points << " points..." << std::endl;
        FeatureExtractor extractor(points.data(), n_points);

        // 3. Define tasks
        std::vector<TaskConfig> tasks = {
            {FeatureType::Verticality,      0.5},
            {FeatureType::Planarity,        0.5},
            {FeatureType::SurfaceVariation, 0.5}
        };

        // 4. Compute
        std::cout << "[Info] Computing features (radius=0.5)..." << std::endl;
        auto results = extractor.compute(tasks);

        // 5. Print results for the first few points
        std::cout << "\nResults (First 5 points):" << std::endl;
        std::cout << std::setw(10) << "PointIdx" 
                  << std::setw(15) << "Verticality" 
                  << std::setw(15) << "Planarity" 
                  << std::setw(15) << "SurfVar" << std::endl;
        
        for (std::size_t i = 0; i < std::min(n_points, (std::size_t)5); ++i) {
            std::cout << std::setw(10) << i 
                      << std::setw(15) << results[0][i] 
                      << std::setw(15) << results[1][i] 
                      << std::setw(15) << results[2][i] << std::endl;
        }

        std::cout << "\n--- Debug Session Finished Successfully ---" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[Error] Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
