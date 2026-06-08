cmake_minimum_required(VERSION 3.24)

project(hpc_project LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MST_BUILD_TESTS "Build dependency-free MST tests" ON)
option(MST_ENABLE_RENDERING "Compile graph visualization support into apps" ON)
set(MST_CUDA_HOST_MEMORY_DEFAULT "pinned" CACHE STRING
    "Default CUDA host edge memory mode: pageable, pinned, or zero_copy")
set_property(CACHE MST_CUDA_HOST_MEMORY_DEFAULT PROPERTY STRINGS
             pageable pinned zero_copy mapped_zero_copy)
if(NOT MST_CUDA_HOST_MEMORY_DEFAULT MATCHES "^(pageable|pinned|zero_copy|mapped_zero_copy)$")
  message(FATAL_ERROR
          "MST_CUDA_HOST_MEMORY_DEFAULT must be pageable, pinned, zero_copy, or mapped_zero_copy")
endif()

add_library(mst_core INTERFACE)
target_include_directories(
  mst_core
  INTERFACE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)
target_compile_features(mst_core INTERFACE cxx_std_20)
target_compile_definitions(
  mst_core
  INTERFACE
    MST_ENABLE_RENDERING=$<BOOL:${MST_ENABLE_RENDERING}>
    MST_DEFAULT_CUDA_HOST_MEMORY="${MST_CUDA_HOST_MEMORY_DEFAULT}"
)

if(MST_BUILD_TESTS)
  enable_testing()
  add_executable(core_tests tests/core_tests.cpp)
  target_link_libraries(core_tests PRIVATE mst_core)
  add_test(NAME core_tests COMMAND core_tests)
endif()

if(APPLE)
  set(MST_OPENMP_PREFIX_HINTS)
  if(DEFINED ENV{HOMEBREW_PREFIX})
    list(APPEND MST_OPENMP_PREFIX_HINTS "$ENV{HOMEBREW_PREFIX}/opt/libomp")
  endif()
  list(APPEND MST_OPENMP_PREFIX_HINTS "/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")
  foreach(openmp_prefix IN LISTS MST_OPENMP_PREFIX_HINTS)
    if(IS_DIRECTORY "${openmp_prefix}")
      list(PREPEND CMAKE_PREFIX_PATH "${openmp_prefix}")
    endif()
  endforeach()
endif()

find_package(OpenMP COMPONENTS CXX QUIET)
if(OpenMP_CXX_FOUND)
  set(MST_OPENMP_STATUS "found; building openmp_app")
  add_subdirectory(openmp)
else()
  set(MST_OPENMP_STATUS "not found; skipping openmp_app")
endif()

find_package(MPI COMPONENTS CXX QUIET)
if(MPI_CXX_FOUND)
  set(MST_MPI_STATUS "found; building mpi_app")
  add_subdirectory(mpi)
else()
  set(MST_MPI_STATUS "not found; skipping mpi_app")
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(MST_CUDA_STATUS "found; building cuda_app")
  add_subdirectory(cuda)
else()
  set(MST_CUDA_STATUS "not found; skipping cuda_app")
endif()

message(STATUS "MST backend detection:")
message(STATUS "  OpenMP: ${MST_OPENMP_STATUS}")
message(STATUS "  MPI:    ${MST_MPI_STATUS}")
message(STATUS "  CUDA:   ${MST_CUDA_STATUS}")
