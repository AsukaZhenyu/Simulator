cmake_minimum_required(VERSION 3.22)
project(single_layer_wait_diagnosis LANGUAGES CXX)
set(LLAMA_ROOT "" CACHE PATH "Existing llama.cpp source root")
set(GGML_BUILD_DIR "" CACHE PATH "Existing llama.cpp build root")
set(RUNTIME_DIR "" CACHE PATH "Explicit GGML runtime directory")
set(NVTX_INCLUDE_DIR "" CACHE PATH "Existing local include root containing nvtx3/nvToolsExt.h")
if(NOT EXISTS "${LLAMA_ROOT}/ggml/include/ggml.h" OR NOT IS_DIRECTORY "${RUNTIME_DIR}" OR NOT EXISTS "${NVTX_INCLUDE_DIR}/nvtx3/nvToolsExt.h")
  message(FATAL_ERROR "Explicit existing LLAMA_ROOT, RUNTIME_DIR, NVTX_INCLUDE_DIR required; no downloads")
endif()
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
set(libpaths "${GGML_BUILD_DIR}/ggml/src" "${GGML_BUILD_DIR}/ggml/src/ggml-cpu" "${GGML_BUILD_DIR}/ggml/src/ggml-cuda")
set(libraries)
foreach(component base cpu cuda)
  find_library(GGML_${component} NAMES ggml-${component} PATHS ${libpaths} PATH_SUFFIXES Release RelWithDebInfo NO_DEFAULT_PATH REQUIRED)
  list(APPEND libraries "${GGML_${component}}")
endforeach()
file(GLOB headers CONFIGURE_DEPENDS "${LLAMA_ROOT}/ggml/include/*.h")
file(GLOB_RECURSE nvtx_headers CONFIGURE_DEPENDS "${NVTX_INCLUDE_DIR}/nvtx3/*.h" "${NVTX_INCLUDE_DIR}/nvtx3/*.hpp")
file(GLOB runtimes CONFIGURE_DEPENDS "${RUNTIME_DIR}/*.dll" "${RUNTIME_DIR}/*.so" "${RUNTIME_DIR}/*.dylib" "${RUNTIME_DIR}/*runtime*.json")
if(NOT runtimes)
  message(FATAL_ERROR "RUNTIME_DIR contains no runtime assets")
endif()
set(bound_args)
foreach(path IN LISTS headers nvtx_headers runtimes libraries)
  list(APPEND bound_args --bound-file "${path}")
endforeach()
list(APPEND bound_args --bound-file "${LLAMA_ROOT}/vendor/nlohmann/json.hpp" --bound-file "${CMAKE_CXX_COMPILER}" --bound-file "${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt")
set(helper "${CMAKE_CURRENT_BINARY_DIR}/highlevel_generated.inc")
set(single "${CMAKE_CURRENT_BINARY_DIR}/single_layer_generated.inc")
set(pair "${CMAKE_CURRENT_BINARY_DIR}/pair_generated.inc")
set(source_manifest "${CMAKE_CURRENT_BINARY_DIR}/source-manifest.json")
add_custom_command(OUTPUT "${helper}" "${single}" "${pair}" "${source_manifest}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_probe.py" --output-dir "${CMAKE_CURRENT_BINARY_DIR}" ${bound_args}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generate_probe.py" "${CMAKE_CURRENT_SOURCE_DIR}/record_build.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/probe.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/../highlevel_accuracy/highlevel_probe.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../single_layer_calibration/probe.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../single_layer_context_pair/paired.cpp"
    "${LLAMA_ROOT}/vendor/nlohmann/json.hpp" ${headers} ${nvtx_headers} ${libraries} ${runtimes}
  VERBATIM)
add_executable(wait-diagnosis-probe probe.cpp "${helper}" "${single}" "${pair}")
target_compile_features(wait-diagnosis-probe PRIVATE cxx_std_17)
target_compile_definitions(wait-diagnosis-probe PRIVATE GGML_SHARED GGML_BACKEND_SHARED HIGHLEVEL_CUDA=1)
target_include_directories(wait-diagnosis-probe PRIVATE "${LLAMA_ROOT}/ggml/include" "${LLAMA_ROOT}/vendor" "${NVTX_INCLUDE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(wait-diagnosis-probe PRIVATE ${libraries})
if(MSVC)
  target_compile_options(wait-diagnosis-probe PRIVATE /W4 /EHsc)
endif()
add_custom_command(TARGET wait-diagnosis-probe POST_BUILD
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/record_build.py"
    --binary "$<TARGET_FILE:wait-diagnosis-probe>" --source-manifest "${source_manifest}"
    --compiler "${CMAKE_CXX_COMPILER}" --compiler-version "${CMAKE_CXX_COMPILER_VERSION}"
    --cmake-cache "${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt" --runtime-dir "${RUNTIME_DIR}" --nvtx-include-dir "${NVTX_INCLUDE_DIR}"
  VERBATIM)
