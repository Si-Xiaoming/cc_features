cmake_minimum_required(VERSION 3.18)
project(cc_features VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "[cc_features] Build type: ${CMAKE_BUILD_TYPE}")


option(CC_FEATURES_BUILD_PYTHON  "Build pybind11 Python extension" ON)
option(CC_FEATURES_BUILD_TESTS   "Build unit tests"                ON)

# find eigen and nanoflann
find_package(Eigen3 5.0.1 REQUIRED NO_MODULE)
message(STATUS "[cc_features] Eigen3 found: ${EIGEN3_INCLUDE_DIR}")


if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nanoflann.hpp")
    message(STATUS "[cc_features] nanoflann: using bundled third_party/nanoflann.hpp")
    add_library(nanoflann_iface INTERFACE)
    target_include_directories(nanoflann_iface INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
    add_library(nanoflann::nanoflann ALIAS nanoflann_iface)
else()
    find_package(nanoflann REQUIRED)
    message(STATUS "[cc_features] nanoflann: found via CMake package")
endif()


find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "[cc_features] OpenMP enabled")
else()
    message(WARNING "[cc_features] OpenMP NOT found — single-threaded build")
endif()

# #######################################################
# kernal of cc_features
# #######################################################
add_library(cc_features_core STATIC
    src/feature_extractor.cpp
)

target_include_directories(cc_features_core PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(cc_features_core PUBLIC
    Eigen3::Eigen
    nanoflann::nanoflann
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(cc_features_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Compiler-specific optimisation flags
target_compile_options(cc_features_core PRIVATE
    $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:
        -O3 -march=native -ffast-math>
    $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:
        /O2 /arch:AVX2 /fp:fast>
    # Strict aliasing & warnings in debug
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# #######################################################
# python extension
# #######################################################
if(CC_FEATURES_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    message(STATUS "[cc_features] pybind11 ${pybind11_VERSION} found")

    pybind11_add_module(_cc_features python/bindings.cpp)
    target_link_libraries(_cc_features PRIVATE cc_features_core)

    target_compile_options(_cc_features PRIVATE
        $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:
            -O3 -march=native>
        $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:
            /O2 /arch:AVX2>
    )

    # Install the .so / .pyd next to the Python package __init__.py
    install(TARGETS _cc_features
        LIBRARY DESTINATION cc_features
        RUNTIME DESTINATION cc_features   # Windows .pyd
    )
endif()

# #######################################################
# tests
# #######################################################
if(CC_FEATURES_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
