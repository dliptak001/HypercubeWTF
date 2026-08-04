cmake_minimum_required(VERSION 3.21)
project(HypercubeWTF VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ——— HypercubeCNN (vendored) ———
add_subdirectory(third_party/HypercubeCNN)

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

target_link_libraries(HypercubeWTFCore PUBLIC HypercubeCNNCore)

target_compile_options(HypercubeWTFCore PRIVATE
    $<$<CONFIG:Release>:-O3 -march=x86-64-v2 -ffast-math>
    $<$<CONFIG:Debug>:-O0 -g>
    -Wall -Wextra -Wno-unknown-pragmas
)

if(MINGW)
    target_link_options(HypercubeWTFCore PRIVATE -static-libgcc -static-libstdc++)
endif()

# ——— Smoke executable (construct WTF, exit 0) ———
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    add_executable(wtf_smoke tests/wtf_smoke.cpp)
    target_link_libraries(wtf_smoke PRIVATE HypercubeWTFCore)
    target_compile_options(wtf_smoke PRIVATE
        $<$<CONFIG:Release>:-O3 -march=x86-64-v2 -ffast-math>
        $<$<CONFIG:Debug>:-O0 -g>
        -Wall -Wextra -Wno-unknown-pragmas
    )
    if(MINGW)
        target_link_options(wtf_smoke PRIVATE -static-libgcc -static-libstdc++)
        target_link_options(wtf_smoke PRIVATE
            $<$<CONFIG:Debug>:-Wl,--allow-multiple-definition>)
    endif()
endif()
