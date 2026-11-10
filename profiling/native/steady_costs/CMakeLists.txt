cmake_minimum_required(VERSION 3.22)
project(steady_costs LANGUAGES CXX)
# The probe is one source for every host: `../portable/platform.hpp` supplies the
# Win32 names it calls on POSIX, and the generated Executor fragments already
# branch on `_WIN32`. Only the toolkit and library layouts differ, below.
#
#   cuda   links ggml-cuda and CUDART, and observes CUDA events
#   metal  links ggml-metal and takes `cuda_runtime_api.h` / `ggml-cuda.h` from
#          `../backend/metal/include`, because the fragments that include them
#          are hash-pinned and cannot be edited. Event timing is refused there.
set(PROBE_BACKEND "cuda" CACHE STRING "Accelerator this probe links: cuda or metal")
set_property(CACHE PROBE_BACKEND PROPERTY STRINGS cuda metal)
if(NOT PROBE_BACKEND STREQUAL "cuda" AND NOT PROBE_BACKEND STREQUAL "metal")
  message(FATAL_ERROR "PROBE_BACKEND is '${PROBE_BACKEND}'; the probes build for cuda or metal")
endif()
set(LLAMA_ROOT "" CACHE PATH "Existing pinned llama.cpp source root")
set(GGML_BUILD_DIR "" CACHE PATH "Existing llama.cpp build root")
set(RUNTIME_DIR "" CACHE PATH "Explicit strict F32 GGML runtime directory")
set(CUDA_ROOT "" CACHE PATH "Existing CUDA toolkit root (cuda), or the metal compat root (metal)")
if(PROBE_BACKEND STREQUAL "metal" AND NOT CUDA_ROOT)
  set(CUDA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../backend/metal")
endif()
if(NOT EXISTS "${LLAMA_ROOT}/ggml/include/ggml.h" OR NOT IS_DIRECTORY "${RUNTIME_DIR}" OR
   NOT EXISTS "${CUDA_ROOT}/include/cuda_runtime_api.h")
  message(FATAL_ERROR "Explicit existing LLAMA_ROOT, RUNTIME_DIR and CUDA_ROOT required; no downloads")
endif()
if(PROBE_BACKEND STREQUAL "metal" AND NOT EXISTS "${LLAMA_ROOT}/ggml/include/ggml-metal.h")
  message(FATAL_ERROR "A metal build needs ggml-metal.h from an existing GGML_METAL=ON tree: ${LLAMA_ROOT}")
endif()
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
# Windows keeps import libraries under `ggml/src/<config>/`; a Linux shared build
# puts `libggml-*.so` in `<build>/bin`; a macOS build puts `libggml-metal.dylib`
# there too. All layouts are searched, so one build recipe works on every host.
set(accelerator_component cuda)
if(PROBE_BACKEND STREQUAL "metal")
  set(accelerator_component metal)
endif()
set(libpaths "${GGML_BUILD_DIR}/ggml/src" "${GGML_BUILD_DIR}/ggml/src/ggml-cpu"
             "${GGML_BUILD_DIR}/ggml/src/${accelerator_component}" "${GGML_BUILD_DIR}/bin")
set(libraries)
foreach(component base cpu ${accelerator_component})
  find_library(GGML_${component} NAMES ggml-${component} PATHS ${libpaths} PATH_SUFFIXES Release RelWithDebInfo NO_DEFAULT_PATH REQUIRED)
  list(APPEND libraries "${GGML_${component}}")
endforeach()
if(PROBE_BACKEND STREQUAL "cuda")
  find_library(CUDART_LIBRARY NAMES cudart PATHS
    "${CUDA_ROOT}/lib/x64" "${CUDA_ROOT}/lib64"
    "${CUDA_ROOT}/targets/aarch64-linux/lib" "${CUDA_ROOT}/targets/x86_64-linux/lib"
    NO_DEFAULT_PATH REQUIRED)
  list(APPEND libraries "${CUDART_LIBRARY}")
endif()
file(GLOB headers CONFIGURE_DEPENDS "${LLAMA_ROOT}/ggml/include/*.h")
# The runtime directory holds the GGML libraries the probe loads at launch, so the
# glob has to name this platform's spelling of them, plus the strict-F32 receipt.
if(WIN32)
  set(RUNTIME_LIBRARY_GLOB "${RUNTIME_DIR}/*.dll")
elseif(APPLE)
  set(RUNTIME_LIBRARY_GLOB "${RUNTIME_DIR}/*.dylib")
else()
  set(RUNTIME_LIBRARY_GLOB "${RUNTIME_DIR}/*.so*")
endif()
file(GLOB runtimes CONFIGURE_DEPENDS ${RUNTIME_LIBRARY_GLOB} "${RUNTIME_DIR}/*runtime*.json")
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
set(probe_definitions GGML_SHARED GGML_BACKEND_SHARED HIGHLEVEL_CUDA=1)
set(probe_include_dirs "${LLAMA_ROOT}/ggml/include" "${LLAMA_ROOT}/ggml/src" "${LLAMA_ROOT}/vendor" "${CMAKE_CURRENT_BINARY_DIR}")
if(PROBE_BACKEND STREQUAL "cuda")
  list(APPEND probe_include_dirs "${CUDA_ROOT}/include")
  list(APPEND probe_definitions PROBE_BACKEND_CUDA=1)
else()
  # Ahead of every other include directory: these two headers are what this
  # build's `cuda_runtime_api.h` and `ggml-cuda.h` resolve to.
  list(INSERT probe_include_dirs 0 "${CUDA_ROOT}/include")
  list(APPEND probe_include_dirs "${LLAMA_ROOT}/ggml/src/ggml-metal")
  list(APPEND probe_definitions PROBE_BACKEND_METAL=1)
endif()
target_compile_definitions(steady-costs-probe PRIVATE ${probe_definitions})
target_include_directories(steady-costs-probe PRIVATE ${probe_include_dirs})
# No `bcrypt`: the digest comes from `../portable/sha256.hpp`, which is one source
# on every host, so the binary no longer depends on a Windows-only library.
target_link_libraries(steady-costs-probe PRIVATE ${libraries})
if(UNIX AND NOT APPLE)
  target_link_libraries(steady-costs-probe PRIVATE pthread dl)
endif()
if(MSVC)
  target_compile_options(steady-costs-probe PRIVATE /W4 /EHsc)
else()
  target_compile_options(steady-costs-probe PRIVATE -Wall -Wextra)
endif()
add_custom_command(TARGET steady-costs-probe POST_BUILD
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/record_build.py"
    --binary "$<TARGET_FILE:steady-costs-probe>" --source-manifest "${source_manifest}"
    --compiler "${CMAKE_CXX_COMPILER}" --compiler-version "${CMAKE_CXX_COMPILER_VERSION}"
    --cmake-cache "${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt" --runtime-dir "${RUNTIME_DIR}"
    --backend "${PROBE_BACKEND}" --architecture "${CMAKE_SYSTEM_PROCESSOR}"
  VERBATIM)
