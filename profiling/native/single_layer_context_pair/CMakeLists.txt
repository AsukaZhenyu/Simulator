cmake_minimum_required(VERSION 3.22)
project(single_layer_context_pair LANGUAGES CXX)
set(LLAMA_ROOT "" CACHE PATH "Existing llama.cpp source root")
set(GGML_BUILD_DIR "" CACHE PATH "Existing llama.cpp build root")
set(RUNTIME_DIR "" CACHE PATH "Explicit GGML runtime directory")
if(NOT EXISTS "${LLAMA_ROOT}/ggml/include/ggml.h" OR NOT IS_DIRECTORY "${RUNTIME_DIR}")
  message(FATAL_ERROR "Explicit LLAMA_ROOT and RUNTIME_DIR are required")
endif()
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
set(libpaths "${GGML_BUILD_DIR}/ggml/src" "${GGML_BUILD_DIR}/ggml/src/ggml-cpu" "${GGML_BUILD_DIR}/ggml/src/ggml-cuda")
set(libraries)
foreach(component base cpu cuda)
  find_library(GGML_${component} NAMES ggml-${component} PATHS ${libpaths} PATH_SUFFIXES Release RelWithDebInfo NO_DEFAULT_PATH REQUIRED)
  list(APPEND libraries "${GGML_${component}}")
endforeach()
file(GLOB headers CONFIGURE_DEPENDS "${LLAMA_ROOT}/ggml/include/*.h")
file(GLOB runtimes CONFIGURE_DEPENDS "${RUNTIME_DIR}/*.dll" "${RUNTIME_DIR}/*.so" "${RUNTIME_DIR}/*.dylib" "${RUNTIME_DIR}/*runtime*.json")
if(NOT runtimes)
  message(FATAL_ERROR "RUNTIME_DIR contains no runtime assets")
endif()
set(bound_args)
foreach(path IN LISTS headers runtimes libraries)
  list(APPEND bound_args --bound-file "${path}")
endforeach()
list(APPEND bound_args --bound-file "${LLAMA_ROOT}/vendor/nlohmann/json.hpp" --bound-file "${CMAKE_CXX_COMPILER}" --bound-file "${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt")
set(helper "${CMAKE_CURRENT_BINARY_DIR}/highlevel_generated.inc")
set(single "${CMAKE_CURRENT_BINARY_DIR}/single_layer_generated.inc")
set(source_manifest "${CMAKE_CURRENT_BINARY_DIR}/source-manifest.json")
add_custom_command(OUTPUT "${helper}" "${single}" "${source_manifest}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_probe.py" --output-dir "${CMAKE_CURRENT_BINARY_DIR}" ${bound_args}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generate_probe.py" "${CMAKE_CURRENT_SOURCE_DIR}/record_build.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/paired.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/../highlevel_accuracy/highlevel_probe.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../single_layer_calibration/probe.cpp"
    "${LLAMA_ROOT}/vendor/nlohmann/json.hpp" ${headers} ${libraries} ${runtimes}
  VERBATIM)
add_executable(context-pair-probe paired.cpp "${helper}" "${single}")
target_compile_features(context-pair-probe PRIVATE cxx_std_17)
target_compile_definitions(context-pair-probe PRIVATE GGML_SHARED GGML_BACKEND_SHARED HIGHLEVEL_CUDA=1)
target_include_directories(context-pair-probe PRIVATE "${LLAMA_ROOT}/ggml/include" "${LLAMA_ROOT}/vendor" "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(context-pair-probe PRIVATE ${libraries})
if(MSVC)
  target_compile_options(context-pair-probe PRIVATE /W4 /EHsc)
endif()
add_custom_command(TARGET context-pair-probe POST_BUILD
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/record_build.py"
    --binary "$<TARGET_FILE:context-pair-probe>" --source-manifest "${source_manifest}"
    --compiler "${CMAKE_CXX_COMPILER}" --compiler-version "${CMAKE_CXX_COMPILER_VERSION}"
    --cmake-cache "${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt" --runtime-dir "${RUNTIME_DIR}"
  VERBATIM)
