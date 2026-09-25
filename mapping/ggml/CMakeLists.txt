cmake_minimum_required(VERSION 3.16)
project(tensor-mapping-ggml-exporter CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# The GGML source tree is a build parameter
# ---------------------------------------------------------------------------
# Not a hardcoded path, and never fetched: the version the exporter was run
# against has to be the version that was reviewed. Point this at the ggml/
# directory of a local llama.cpp checkout (DESIGN.md 3).
if(NOT GGML_SOURCE_DIR AND DEFINED ENV{GGML_SOURCE_DIR})
    set(GGML_SOURCE_DIR "$ENV{GGML_SOURCE_DIR}")
endif()
if(NOT GGML_SOURCE_DIR)
    message(FATAL_ERROR
        "GGML_SOURCE_DIR is not set.\n"
        "Pass the ggml/ directory of a local llama.cpp checkout, for example:\n"
        "  cmake -S . -B build -G \"Visual Studio 17 2022\" -A x64 \\\n"
        "        -DGGML_SOURCE_DIR=<llama.cpp>/ggml -DGGML_CUDA=OFF\n"
        "or export GGML_SOURCE_DIR and configure without the option.")
endif()
get_filename_component(GGML_SOURCE_DIR "${GGML_SOURCE_DIR}" ABSOLUTE)
if(NOT EXISTS "${GGML_SOURCE_DIR}/include/ggml.h"
   OR NOT EXISTS "${GGML_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "GGML_SOURCE_DIR=${GGML_SOURCE_DIR} does not look like a ggml source tree.\n"
        "Expected both ${GGML_SOURCE_DIR}/include/ggml.h and "
        "${GGML_SOURCE_DIR}/CMakeLists.txt.")
endif()

# ---------------------------------------------------------------------------
# CPU-only by default
# ---------------------------------------------------------------------------
# The exporter is a front end; it needs the graph-building API, not a backend.
# ACCEPTANCE.md 7 asks that core verification be possible without CUDA, so CUDA
# stays off unless someone asks for it. GGML_CUDA already defaults to OFF
# upstream; this makes the choice explicit and independent of that default.
option(GGML_EXPORTER_WITH_CUDA "build the ggml backend with CUDA enabled" OFF)
if(NOT GGML_EXPORTER_WITH_CUDA)
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
endif()

# An out-of-tree subproject never sets GGML_STANDALONE (ggml/CMakeLists.txt
# compares CMAKE_SOURCE_DIR with CMAKE_CURRENT_SOURCE_DIR, which differ here),
# so ggml's own examples and tests default to off. Stated anyway: they are not
# wanted, and the export must not depend on them.
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Adds ggml as a subproject of this build. The upstream tree is read only --
# every generated file lands under this project's binary directory.
add_subdirectory("${GGML_SOURCE_DIR}" ggml-build)

# ---------------------------------------------------------------------------
# The exporter
# ---------------------------------------------------------------------------
add_executable(ggml-export-workload
    src/export_workload.cpp
    src/graphs.cpp)

target_include_directories(ggml-export-workload PRIVATE src)
target_link_libraries(ggml-export-workload PRIVATE ggml)

if(MSVC)
    # The sources are deliberately ASCII (see the header comment in
    # export_workload.cpp). /utf-8 makes that explicit instead of relying on the
    # machine's code page.
    target_compile_options(ggml-export-workload PRIVATE /utf-8)
endif()

# bin/Release/ggml-export-workload.exe with a multi-config generator, matching
# the layout of the llama.cpp builds already on this machine.
set_target_properties(ggml-export-workload PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
