cmake_minimum_required(VERSION 3.24)
project(chain_calibration_probe LANGUAGES CXX)
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
# The historical source is immutable. Only its terminal entry point is renamed
# in a generated include; all mathematical graphs and reference helpers remain.
set(PINNED_CONTEXT "${CMAKE_CURRENT_SOURCE_DIR}/../execution_transforms_dual/compute_context.cpp")
set(PINNED_MATCHED "${CMAKE_CURRENT_SOURCE_DIR}/../matched_mechanisms/compute.cpp")
file(SHA256 "${PINNED_CONTEXT}" CONTEXT_SHA)
file(SHA256 "${PINNED_MATCHED}" MATCHED_SHA)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/compute.cpp" PROBE_SHA)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" CMAKE_SHA)
if(NOT CONTEXT_SHA STREQUAL "37ed5525074d2653050daa433a7b938a6f7f751a1afdeaac17a8c21cc2173cd4"
    OR NOT MATCHED_SHA STREQUAL "2556bed47f9f45ffef7884f5f58b452ec69a5c036b98a1e137afc9c55085c327")
  message(FATAL_ERROR "Historical context/reference source differs from the pinned protocol")
endif()
file(READ "${PINNED_CONTEXT}" context)
string(REPLACE "int main(int argc,char**argv)" "static int chain_context_legacy_main(int argc,char**argv)" renamed "${context}")
if(context STREQUAL renamed)
  message(FATAL_ERROR "Historical context entry point did not match")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/context_generated.inc" "${renamed}")
file(SHA256 "${CMAKE_CURRENT_BINARY_DIR}/context_generated.inc" GENERATED_SHA)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/source-binding.json"
  "{\"context_sha256\":\"${CONTEXT_SHA}\",\"matched_sha256\":\"${MATCHED_SHA}\",\"probe_sha256\":\"${PROBE_SHA}\",\"cmake_sha256\":\"${CMAKE_SHA}\",\"generated_include_sha256\":\"${GENERATED_SHA}\",\"transformation\":\"rename_terminal_main_only\"}\n")
add_executable(chain-calibration-probe compute.cpp)
target_compile_features(chain-calibration-probe PRIVATE cxx_std_17)
target_include_directories(chain-calibration-probe PRIVATE "${PROBE_INCLUDE}"
    "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../execution_transforms_dual")
target_compile_definitions(chain-calibration-probe PRIVATE GGML_SHARED GGML_BACKEND_SHARED MATCHED_HAVE_CUDA
    CHAIN_CONTEXT_SHA="${CONTEXT_SHA}" CHAIN_MATCHED_SHA="${MATCHED_SHA}" CHAIN_PROBE_SHA="${PROBE_SHA}")
target_link_libraries(chain-calibration-probe PRIVATE "${PROBE_base}" "${PROBE_cpu}" "${PROBE_cuda}" CUDA::cudart Threads::Threads)
