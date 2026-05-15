cmake_minimum_required(VERSION 3.24)

project(hpc_project LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MST_BUILD_TESTS "Build dependency-free MST tests" ON)

add_library(mst_core INTERFACE)
target_include_directories(
  mst_core
  INTERFACE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)
target_compile_features(mst_core INTERFACE cxx_std_20)

if(APPLE AND NOT DEFINED OpenMP_ROOT)
  set(OpenMP_ROOT "/opt/homebrew/opt/libomp" CACHE PATH "OpenMP root prefix")
endif()

if(APPLE AND NOT DEFINED MPI_CXX_COMPILER)
  set(MPI_CXX_COMPILER "/opt/homebrew/bin/mpicxx" CACHE FILEPATH "MPI C++ compiler wrapper")
endif()

if(MST_BUILD_TESTS)
  enable_testing()
  add_executable(core_tests tests/core_tests.cpp)
  target_link_libraries(core_tests PRIVATE mst_core)
  add_test(NAME core_tests COMMAND core_tests)
endif()

find_package(OpenMP COMPONENTS CXX QUIET)
if(OpenMP_CXX_FOUND)
  add_subdirectory(openmp)
else()
  message(STATUS "OpenMP backend skipped: OpenMP CXX package not found")
endif()

find_package(MPI COMPONENTS CXX QUIET)
if(MPI_CXX_FOUND)
  add_subdirectory(mpi)
else()
  message(STATUS "MPI backend skipped: MPI CXX package not found")
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  add_subdirectory(cuda)
else()
  message(STATUS "CUDA backend skipped: CUDA compiler not found")
endif()
