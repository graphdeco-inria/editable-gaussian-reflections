cmake_minimum_required(VERSION 3.18)
project (gausstracer LANGUAGES CXX CUDA)

# Set default build type to 'Release'
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# Add path to our CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# === Compiler Standards ===
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)  # 8.6 = RTX A6000; (7.5 = RTX 8000)

# === Dependency Discovery ===
find_package(PythonLibs REQUIRED)
# We use this to be able to link against CUDA::cudart and/or CUDA::cuda_driver
find_package(CUDAToolkit REQUIRED)
# Make sure we have access to Optix8
find_package(OptiX8 REQUIRED)
# Make sure we have access to Torch
find_package(Torch REQUIRED)

# === Environment Variables ===
set(CONFIG $ENV{CONFIG})
if(NOT CONFIG)
    set(CONFIG -1)
endif()

# === Source Files ===
set(SOURCES
    csrc/gausstracer.cpp
    csrc/populate_bvh.cu
)

# === Main Shared Library Target ===
add_library(gausstracer SHARED ${SOURCES})

# === OptiX PTX Target ===
add_library(optix_program OBJECT csrc/shaders.cu)
set_target_properties(optix_program PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_PTX_COMPILATION ON
)

# === Include Directories ===
target_include_directories(gausstracer PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${OptiX8_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_include_directories(optix_program PRIVATE
    ${OptiX8_INCLUDE_DIRS}
)

# === Definitions ===
target_compile_definitions(gausstracer PRIVATE
    CONFIG=${CONFIG}
    ENABLE_OPTIX
    CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
    CMAKE_CURRENT_LIST_DIR="${CMAKE_CURRENT_LIST_DIR}"
    $<$<BOOL:$ENV{EXTRA_H_AVAILABLE}>:EXTRA_H_AVAILABLE=1>
    OPTIX_SAMPLE_NAME_DEFINE=gausstracer
    OPTIX_SAMPLE_DIR_DEFINE=gausstracer
)

target_compile_definitions(optix_program PRIVATE
    CONFIG=${CONFIG}
)

# === Compile Options ===
target_compile_options(gausstracer PRIVATE
    -Wno-everything
    -Wno-macro-redefined
    $<$<COMPILE_LANGUAGE:CUDA>:-ffast-math -ftz=true -prec-div=false -prec-sqrt=false>
)

target_compile_options(optix_program PRIVATE
    -Wno-everything
)

# === Link Libraries ===
target_link_libraries(gausstracer PRIVATE
    ${TORCH_LIBRARIES}
    ${PYTHON_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${OptiX_LIBRARIES}
)

target_link_libraries(optix_program PRIVATE
    ${TORCH_LIBRARIES}  # TEMPORARY HACK
)

# === Custom: Generate Python Config File ===
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/raytracer_config.py
    COMMAND ${CMAKE_COMMAND} -E rm -f ${CMAKE_BINARY_DIR}/raytracer_config.py
    COMMAND ${CMAKE_CXX_COMPILER} -DCONFIG=${CONFIG} ${CMAKE_SOURCE_DIR}/csrc/format_config_to_python.cpp -o format_config_to_python
    COMMAND ${CMAKE_BINARY_DIR}/format_config_to_python > ${CMAKE_BINARY_DIR}/raytracer_config.py
    DEPENDS ${CMAKE_SOURCE_DIR}/csrc/format_config_to_python.cpp
)

add_custom_target(generate-config
    COMMAND ${CMAKE_COMMAND} -E echo "Forcing rebuild of raytracer_config.py"
    DEPENDS ${CMAKE_BINARY_DIR}/raytracer_config.py
)

add_dependencies(gausstracer generate-config)

# === Custom: Preprocessor Flags Dump ===
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/flags_output.txt
    COMMAND ${CMAKE_COMMAND} -E echo "Running gcc command..."
    COMMAND gcc -E -dM ${CMAKE_SOURCE_DIR}/csrc/flags.h | grep -v '^#define _' | grep -v '^#define [a-z]' | sed s'/#define //' | sed s'/ /=/' | sed s'/f$//' | sed -s 's/=$/=True/' > ${CMAKE_BINARY_DIR}/flags_output.txt
    DEPENDS ${CMAKE_SOURCE_DIR}/csrc/flags.h
    COMMENT "Generating flags_output.txt"
)

# === Copy Header File for Build ===
file(COPY csrc/flags.h DESTINATION ${CMAKE_BINARY_DIR})
