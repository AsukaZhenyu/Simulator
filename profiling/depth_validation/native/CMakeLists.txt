cmake_minimum_required(VERSION 3.22)
project(depth_validation LANGUAGES CXX)

# Which accelerator this probe is built for. The dense column compares a GPU
# placement against a CPU one, and both CUDA and Metal are GPU placements: what
# differs is the link line and the three questions the probe asks about device
# memory, not the measurement.
#
#   cuda   links ggml-cuda, CUDART and CUPTI (the toolkit's, even though the
#          portable modes never call into CUPTI at run time)
#   metal  links ggml-metal and puts `../../native/backend/metal` ahead of the
#          toolkit headers, so the hash-pinned fragments compile unchanged
#
# Any other value is refused here rather than producing a probe that measures
# something unnamed.
set(PROBE_BACKEND "cuda" CACHE STRING "Accelerator this probe links: cuda or metal")
set_property(CACHE PROBE_BACKEND PROPERTY STRINGS cuda metal)
if(NOT PROBE_BACKEND STREQUAL "cuda" AND NOT PROBE_BACKEND STREQUAL "metal")
 message(FATAL_ERROR "PROBE_BACKEND is '${PROBE_BACKEND}'; the probes build for cuda or metal")
endif()

set(LLAMA_ROOT "" CACHE PATH "Explicit existing llama.cpp source")
set(GGML_BUILD_DIR "" CACHE PATH "Existing GGML import libraries")
set(STEADY_GENERATED_DIR "" CACHE PATH "Pinned existing generated Executor files")
set(CUDA_ROOT "" CACHE PATH "Installed CUDA toolkit (cuda builds), or the metal compat root (metal builds)")
set(CUPTI_ROOT "" CACHE PATH "Installed CUPTI toolkit (cuda builds only)")
if(PROBE_BACKEND STREQUAL "metal" AND NOT CUDA_ROOT)
 # A metal build still has to answer the CUDA names the pinned fragments use.
 # `native/backend/metal/include` holds those two headers and nothing else, so
 # pointing CUDA_ROOT at its parent is the same shape a toolkit has:
 # `<root>/include/cuda_runtime_api.h`. Nothing is downloaded or faked twice.
 set(CUDA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../native/backend/metal")
endif()
set(required_inputs "${LLAMA_ROOT}/ggml/include/ggml.h" "${STEADY_GENERATED_DIR}/state_response_generated.inc"
  "${CUDA_ROOT}/include/cuda_runtime_api.h")
if(PROBE_BACKEND STREQUAL "cuda")
 list(APPEND required_inputs "${CUPTI_ROOT}/include/cupti.h")
else()
 # A Metal build needs the backend's own header from the pinned tree, which is
 # only there when that tree was configured with GGML_METAL=ON.
 list(APPEND required_inputs "${LLAMA_ROOT}/ggml/include/ggml-metal.h")
endif()
foreach(path ${required_inputs})
 if(NOT EXISTS "${path}")
  message(FATAL_ERROR "Explicit file missing: ${path}")
 endif()
endforeach()
if(PROBE_BACKEND STREQUAL "cuda")
 set(accelerator_component cuda)
else()
 set(accelerator_component metal)
endif()
set(libraries)
# Windows keeps import libraries under `ggml/src/<config>/`; a Linux shared build
# puts `libggml-*.so` in `<build>/bin`; a macOS framework build puts
# `libggml-metal.dylib` in `<build>/bin` as well. All layouts are searched, so
# one build recipe works on every host.
foreach(component base cpu ${accelerator_component})
 find_library(GGML_${component} NAMES ggml-${component} PATHS "${GGML_BUILD_DIR}/ggml/src" "${GGML_BUILD_DIR}/ggml/src/ggml-cpu" "${GGML_BUILD_DIR}/ggml/src/ggml-cuda" "${GGML_BUILD_DIR}/ggml/src/ggml-metal" "${GGML_BUILD_DIR}/bin" PATH_SUFFIXES Release RelWithDebInfo NO_DEFAULT_PATH REQUIRED)
 list(APPEND libraries "${GGML_${component}}")
endforeach()
set(probe_definitions GGML_SHARED GGML_BACKEND_SHARED HIGHLEVEL_CUDA=1)
set(probe_include_dirs "${LLAMA_ROOT}/ggml/include" "${LLAMA_ROOT}/ggml/src" "${LLAMA_ROOT}/vendor" "${STEADY_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../native")
if(PROBE_BACKEND STREQUAL "cuda")
 # A toolkit puts its runtime in lib/x64 on Windows, lib64 on a Linux x86_64
 # install, and targets/<arch>-linux/lib on aarch64 (Jetson, and DGX Spark's
 # Grace/Blackwell part). CUPTI follows the same layout.
 if(WIN32)
  find_library(CUDART_LIBRARY NAMES cudart PATHS "${CUDA_ROOT}/lib/x64" NO_DEFAULT_PATH REQUIRED)
  find_library(CUPTI_LIBRARY NAMES cupti PATHS "${CUPTI_ROOT}/lib64" NO_DEFAULT_PATH REQUIRED)
 else()
  find_library(CUDART_LIBRARY NAMES cudart PATHS
   "${CUDA_ROOT}/lib64" "${CUDA_ROOT}/targets/aarch64-linux/lib"
   "${CUDA_ROOT}/targets/x86_64-linux/lib" NO_DEFAULT_PATH REQUIRED)
  find_library(CUPTI_LIBRARY NAMES cupti PATHS
   "${CUPTI_ROOT}/lib64" "${CUPTI_ROOT}/targets/aarch64-linux/lib"
   "${CUPTI_ROOT}/targets/x86_64-linux/lib" NO_DEFAULT_PATH REQUIRED)
 endif()
 list(APPEND libraries "${CUDART_LIBRARY}" "${CUPTI_LIBRARY}")
 list(APPEND probe_include_dirs "${CUDA_ROOT}/include" "${CUPTI_ROOT}/include")
 list(APPEND probe_definitions PROBE_BACKEND_CUDA=1)
else()
 # First on the list: these two headers replace `cuda_runtime_api.h` and
 # `ggml-cuda.h` for this build, and the fragments that include them are
 # hash-pinned, so the substitution happens at the include path.
 list(INSERT probe_include_dirs 0 "${CUDA_ROOT}/include")
 list(APPEND probe_include_dirs "${LLAMA_ROOT}/ggml/src/ggml-metal")
 list(APPEND probe_definitions PROBE_BACKEND_METAL=1)
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/depth_probe.cpp")
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/depth_probe.cpp" HISTORY_SOURCE_SHA256)
add_executable(depth-probe depth_probe.cpp)
target_compile_features(depth-probe PRIVATE cxx_std_17)
target_compile_definitions(depth-probe PRIVATE ${probe_definitions} HISTORY_SOURCE_SHA256="${HISTORY_SOURCE_SHA256}")
target_include_directories(depth-probe PRIVATE ${probe_include_dirs})
# `bcrypt` is gone: SHA-256 is now in-tree (native/portable/sha256.hpp), so the
# digest a Linux receipt records is the same function that produced the Windows
# one. `psapi` remains because <psapi.h> is still the Windows half of
# native/portable/platform.hpp.
if(WIN32)
 target_link_libraries(depth-probe PRIVATE ${libraries} psapi)
elseif(APPLE)
 target_link_libraries(depth-probe PRIVATE ${libraries})
else()
 target_link_libraries(depth-probe PRIVATE ${libraries} pthread dl)
endif()
if(MSVC)
 target_compile_options(depth-probe PRIVATE /W4 /EHsc)
else()
 target_compile_options(depth-probe PRIVATE -Wall)
endif()
