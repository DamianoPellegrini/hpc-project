cmake_minimum_required(VERSION 3.24)

project(hpc_project LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(mst_common INTERFACE)
target_include_directories(mst_common INTERFACE ${PROJECT_SOURCE_DIR})

if(APPLE AND NOT DEFINED MPI_CXX_COMPILER)
  set(MPI_CXX_COMPILER "/opt/homebrew/bin/mpicxx" CACHE FILEPATH "MPI C++ compiler wrapper")
endif()

if(APPLE AND NOT DEFINED OpenMP_ROOT)
  set(OpenMP_ROOT "/opt/homebrew/opt/libomp" CACHE PATH "OpenMP root prefix")
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_subdirectory(mpi)
add_subdirectory(openmp)
