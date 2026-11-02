cmake_minimum_required(VERSION 3.24)
project(execution_transform_batch_probe LANGUAGES CXX)
set(GGML_SOURCE_DIR "" CACHE PATH "Existing GGML source root")
set(GGML_BUILD_DIR "" CACHE PATH "Existing GGML build root")
find_path(PROBE_INCLUDE ggml.h PATHS "${GGML_SOURCE_DIR}/include" NO_DEFAULT_PATH REQUIRED)
set(libpaths "${GGML_BUILD_DIR}/ggml/src" "${GGML_BUILD_DIR}/ggml/src/ggml-cuda"
    "${GGML_BUILD_DIR}/ggml/src/ggml-cpu" "${GGML_BUILD_DIR}/src"
    "${GGML_BUILD_DIR}/src/ggml-cuda" "${GGML_BUILD_DIR}/src/ggml-cpu")
foreach(part base cpu cuda)
  find_library(PROBE_${part} NAMES ggml-${part} PATHS ${libpaths} PATH_SUFFIXES Release REQUIRED NO_DEFAULT_PATH)
endforeach()
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
set(batch_source "${CMAKE_CURRENT_BINARY_DIR}/compute_batch.cpp")
set(batch_manifest "${CMAKE_CURRENT_BINARY_DIR}/source-manifest.json")
add_custom_command(OUTPUT "${batch_source}" "${batch_manifest}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_probe.py"
          --output "${batch_source}" --manifest "${batch_manifest}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generate_probe.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/../execution_transforms_dual/compute_context.cpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/../matched_mechanisms/compute.cpp"
  VERBATIM)
add_executable(execution-transform-batch-probe "${batch_source}")
target_compile_features(execution-transform-batch-probe PRIVATE cxx_std_17)
target_include_directories(execution-transform-batch-probe PRIVATE "${PROBE_INCLUDE}")
target_compile_definitions(execution-transform-batch-probe PRIVATE GGML_SHARED GGML_BACKEND_SHARED MATCHED_HAVE_CUDA)
target_link_libraries(execution-transform-batch-probe PRIVATE "${PROBE_base}" "${PROBE_cpu}" "${PROBE_cuda}" CUDA::cudart Threads::Threads)
# Multi-config generators place the executable under Release/Debug. Keep the
# source identity and generated source adjacent to the actual target file so
# the reusable measurement runner sees the same layout on every generator.
set_property(TARGET execution-transform-batch-probe APPEND PROPERTY LINK_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/record_build.py")
add_custom_command(TARGET execution-transform-batch-probe POST_BUILD
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/record_build.py"
          --binary "$<TARGET_FILE:execution-transform-batch-probe>"
          --generated-source "${batch_source}"
          --source-manifest "${batch_manifest}"
          --cmake-source-dir "${CMAKE_CURRENT_SOURCE_DIR}"
          --cmake-build-dir "${CMAKE_CURRENT_BINARY_DIR}"
          --config "$<CONFIG>"
          --cmake-command "${CMAKE_COMMAND}"
          --cmake-generator "${CMAKE_GENERATOR}"
          --compiler "${CMAKE_CXX_COMPILER}"
          --compiler-id "${CMAKE_CXX_COMPILER_ID}"
          --compiler-version "${CMAKE_CXX_COMPILER_VERSION}"
          --cmake-post-build
  VERBATIM)
