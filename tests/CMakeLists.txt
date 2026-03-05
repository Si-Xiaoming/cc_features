# Add a standalone executable for C++ debugging
add_executable(cc_features_debug main_debug.cpp)

# Link against the core library
target_link_libraries(cc_features_debug PRIVATE cc_features_core)

# Inherit compile options (like AVX2, Wall) if needed, 
# although target_link_libraries(PUBLIC) handles includes/OpenMP.
# We explicitly set standard just in case.
set_target_properties(cc_features_debug PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Output to the project root or bin for easy access (optional)
# set_target_properties(cc_features_debug PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
