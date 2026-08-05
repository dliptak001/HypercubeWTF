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

# ——— Smoke + Phase 4 demos (top-level project only) ———
# Layout: examples/<demo>/…  shared headers: examples/common/
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    function(wtf_add_exe name)
        cmake_parse_arguments(WTF_EX "" "" "SOURCES;INCLUDES" ${ARGN})
        add_executable(${name} ${WTF_EX_SOURCES})
        target_link_libraries(${name} PRIVATE HypercubeWTFCore)
        target_include_directories(${name} PRIVATE
            ${PROJECT_SOURCE_DIR}/examples/common
            ${WTF_EX_INCLUDES}
        )
        target_compile_options(${name} PRIVATE
            $<$<CONFIG:Release>:-O3 -march=x86-64-v2 -ffast-math>
            $<$<CONFIG:Debug>:-O0 -g>
            -Wall -Wextra -Wno-unknown-pragmas
        )
        if(MINGW)
            target_link_options(${name} PRIVATE -static-libgcc -static-libstdc++)
            target_link_options(${name} PRIVATE
                $<$<CONFIG:Debug>:-Wl,--allow-multiple-definition>)
        endif()
        if(WIN32)
            # done_beep.h → PlaySound (system aliases); Beep() is unreliable.
            target_link_libraries(${name} PRIVATE winmm)
        endif()
    endfunction()

    wtf_add_exe(wtf_smoke SOURCES tests/wtf_smoke.cpp)
    wtf_add_exe(wtf_synth SOURCES examples/synth/wtf_synth.cpp)
    wtf_add_exe(wtf_mnist
        SOURCES examples/mnist/wtf_mnist.cpp
        INCLUDES ${PROJECT_SOURCE_DIR}/examples/mnist)

    # Deploy binary for local tooling (e.g. C:\HypercubeWTF\wtf_mnist.exe).
    # Runs after every successful wtf_mnist link (Release and Debug build dirs).
    add_custom_command(TARGET wtf_mnist POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "C:/HypercubeWTF"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:wtf_mnist>
                "C:/HypercubeWTF/$<TARGET_FILE_NAME:wtf_mnist>"
        COMMENT "Copy wtf_mnist → C:/HypercubeWTF"
    )
endif()
