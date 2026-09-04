cmake_minimum_required(VERSION 3.21)
project(HypercubeWTF VERSION 1.0.3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Single-config generators (Ninja/Make): empty CMAKE_BUILD_TYPE disables
# $<CONFIG:…> flags below. Multi-config (VS, Xcode) leaves this alone.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ——— HypercubeCNN (vendored core; see third_party/HypercubeCNN/VENDORED.md) ———
add_subdirectory(third_party/HypercubeCNN)

# ——— Shared compile flags (core + demos) ———
add_library(wtf_compile_flags INTERFACE)
target_compile_options(wtf_compile_flags INTERFACE
    $<$<CONFIG:Release>:-O3 -march=x86-64-v2 -ffast-math>
    $<$<CONFIG:Debug>:-O0 -g>
    -Wall -Wextra -Wno-unknown-pragmas
)
if(MINGW)
    target_link_options(wtf_compile_flags INTERFACE
        -static-libgcc
        -static-libstdc++
    )
endif()

# ——— Core library ———
add_library(HypercubeWTFCore STATIC
    Reservoir.cpp
    Readout.cpp
    WTF.cpp
)

target_include_directories(HypercubeWTFCore
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)

target_link_libraries(HypercubeWTFCore
    PUBLIC  HypercubeCNNCore
    PRIVATE wtf_compile_flags
)

# ——— Smoke + demos (only when this repo is the top-level project) ———
# Layout: examples/<demo>/…  shared headers: examples/common/
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    function(wtf_add_exe name)
        cmake_parse_arguments(WTF_EX "" "" "SOURCES;INCLUDES" ${ARGN})
        add_executable(${name} ${WTF_EX_SOURCES})
        target_link_libraries(${name} PRIVATE HypercubeWTFCore wtf_compile_flags)
        target_include_directories(${name} PRIVATE
            ${PROJECT_SOURCE_DIR}/examples/common
            ${WTF_EX_INCLUDES}
        )
        if(MINGW)
            # Debug MinGW + static libs can hit duplicate symbols on some hosts.
            target_link_options(${name} PRIVATE
                $<$<CONFIG:Debug>:-Wl,--allow-multiple-definition>)
        endif()
    endfunction()

    wtf_add_exe(wtf_smoke SOURCES tests/wtf_smoke.cpp)
    wtf_add_exe(wtf_synth SOURCES examples/synth/wtf_synth.cpp)
    wtf_add_exe(wtf_mnist
        SOURCES examples/mnist/wtf_mnist.cpp
        INCLUDES ${PROJECT_SOURCE_DIR}/examples/mnist)

    wtf_add_exe(wtf_raman
        SOURCES
            examples/RamanBaselineExtraction/wtf_raman.cpp
            examples/RamanBaselineExtraction/BaselineExtractor.cpp
            examples/RamanBaselineExtraction/RamanDataset.cpp
        INCLUDES ${PROJECT_SOURCE_DIR}/examples/RamanBaselineExtraction)
    wtf_add_exe(wtf_raman_extract
        SOURCES
            examples/RamanBaselineExtraction/wtf_raman_extract.cpp
            examples/RamanBaselineExtraction/BaselineExtractor.cpp
            examples/RamanBaselineExtraction/RamanDataset.cpp
            examples/RamanBaselineExtraction/RamanExtract.cpp
        INCLUDES ${PROJECT_SOURCE_DIR}/examples/RamanBaselineExtraction)
endif()
