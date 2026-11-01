cmake_minimum_required(VERSION 3.22)
project(steady_costs LANGUAGES CXX)
if(NOT WIN32)
  message(FATAL_ERROR "This explicit QPC and pinned event ABI probe currently requires Windows")
endif()
set(LLAMA_ROOT "" CACHE PATH "Existing pinned llama.cpp source root")
set(GGML_BUILD_DIR "" CACHE PATH "Existing llama.cpp build root")
set(RUNTIME_DIR "" CACHE PATH "Explicit strict F32 GGML runtime directory")
set(CUDA_ROOT "" CACHE PATH "Existing CUDA toolkit root")
if(NOT EXISTS "${LLAMA_ROOT}/ggml/include/ggml.h" OR NOT IS_DIRECTORY "${RUNTIME_DIR}" OR
   NOT EXISTS "${CUDA_ROOT}/include/cuda_runtime_api.h")
  message(FATAL_ERROR "Explicit existing LLAMA_ROOT, RUNTIME_DIR and CUDA_ROOT required; no downloads")
endif()
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
set(libpaths "${GGML_BUILD_DIR}/ggml/src" "${GGML_BUILD_DIR}/ggml/src/ggml-cpu" "${GGML_BUILD_DIR}/ggml/src/ggml-cuda")
set(libraries)
foreach(component base cpu cuda)
  find_library(GGML_${component} NAMES ggml-${component} PATHS ${libpaths} PATH_SUFFIXES Release RelWithDebInfo NO_DEFAULT_PATH REQUIRED)
  list(APPEND libraries "${GGML_${component}}")
endforeach()
find_library(CUDART_LIBRARY NAMES cudart PATHS "${CUDA_ROOT}/lib/x64" NO_DEFAULT_PATH REQUIRED)
list(APPEND libraries "${CUDART_LIBRARY}")
file(GLOB headers CONFIGURE_DEPENDS "${LLAMA_ROOT}/ggml/include/*.h")
file(GLOB runtimes CONFIGURE_DEPENDS "${RUNTIME_DIR}/*.dll" "${RUNTIME_DIR}/*runtime*.json")
set(bound_args)
foreach(path IN LISTS headers runtimes libraries)
  list(APPEND bound_args --bound-file "${path}")
endforeach()
foreach(name CMakeLists.txt generate_probe.py record_build.py probe.cpp components.hpp)
  list(APPEND bound_args --bound-file "${CMAKE_CURRENT_SOURCE_DIR}/${name}")
endforeach()
list(APPEND bound_args --bound-file "${CMAKE_CXX_COMPILER}" --bound-file "${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt"
  --bound-file "${LLAMA_ROOT}/vendor/nlohmann/json.hpp")
set(generated)
foreach(name highlevel_generated single_layer_generated timing_events numeric_helpers state_response_generated)
  list(APPEND generated "${CMAKE_CURRENT_BINARY_DIR}/${name}.inc")
endforeach()
set(source_manifest "${CMAKE_CURRENT_BINARY_DIR}/source-manifest.json")
add_custom_command(OUTPUT ${generated} "${source_manifest}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_probe.py"
    --output-dir "${CMAKE_CURRENT_BINARY_DIR}" --llama-root "${LLAMA_ROOT}" --cuda-root "${CUDA_ROOT}" ${bound_args}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/probe.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/components.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/generate_probe.py" "${CMAKE_CURRENT_SOURCE_DIR}/record_build.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" "${CMAKE_CURRENT_SOURCE_DIR}/../state_response/probe.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../state_response/generate_probe.py" "${CMAKE_CURRENT_SOURCE_DIR}/../state_response/record_build.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../highlevel_accuracy/highlevel_probe.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../single_layer_calibration/probe.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../state_calibration/compute.cpp" ${headers} ${runtimes} ${libraries}
  VERBATIM COMMAND_EXPAND_LISTS)
add_executable(steady-costs-probe probe.cpp ${generated})
target_compile_features(steady-costs-probe PRIVATE cxx_std_17)
target_compile_definitions(steady-costs-probe PRIVATE GGML_SHARED GGML_BACKEND_SHARED HIGHLEVEL_CUDA=1)
target_include_directories(steady-costs-probe PRIVATE "${LLAMA_ROOT}/ggml/include" "${LLAMA_ROOT}/ggml/src"
  "${LLAMA_ROOT}/vendor" "${CUDA_ROOT}/include" "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(steady-costs-probe PRIVATE ${libraries} bcrypt)
if(MSVC)
  target_compile_options(steady-costs-probe PRIVATE /W4 /EHsc)
endif()
add_custom_command(TARGET steady-costs-probe POST_BUILD
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/record_build.py"
    --binary "$<TARGET_FILE:steady-costs-probe>" --source-manifest "${source_manifest}"
    --compiler "${CMAKE_CXX_COMPILER}" --compiler-version "${CMAKE_CXX_COMPILER_VERSION}"
    --cmake-cache "${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt" --runtime-dir "${RUNTIME_DIR}"
  VERBATIM)
