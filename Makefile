# HPC fallback Makefile.
#
# CMake/Ninja remain the primary build path for local development. This file is
# intentionally hand-written for Slurm environments where CMake and Ninja are
# unavailable.

.DEFAULT_GOAL := help

BUILD_DIR ?= build

CXX ?= g++
MPICXX ?= mpicxx
NVCC ?= nvcc
NVCC_CCBIN ?=

CXXSTD ?= -std=c++20
CPPFLAGS ?= -I. -Iinclude
CXXFLAGS ?= -O3 -Wall -Wextra
LDFLAGS ?=
LDLIBS ?=

OPENMP_FLAGS ?= -fopenmp
MPI_CXXFLAGS ?=
MPI_LDLIBS ?=

CUDA_CPPFLAGS ?= -I. -Iinclude
CUDAFLAGS ?= -O3 -std=c++20
CUDA_ARCH ?=
CUDA_LDFLAGS ?=

ifneq ($(strip $(NVCC_CCBIN)),)
NVCC_CCBIN_FLAG := -ccbin $(NVCC_CCBIN)
endif

MST_HEADERS := $(wildcard include/mst/*/*.hpp cuda/*.cuh)

OPENMP_SRC := openmp/main.cpp
MPI_SRC := mpi/main.cpp
CUDA_SRC := cuda/main.cu
TEST_SRC := tests/core_tests.cpp

OPENMP_BIN := $(BUILD_DIR)/openmp/openmp_app
MPI_BIN := $(BUILD_DIR)/mpi/mpi_app
CUDA_BIN := $(BUILD_DIR)/cuda/cuda_app
TEST_BIN := $(BUILD_DIR)/tests/core_tests

.PHONY: help all openmp openmp_app mpi mpi_app cuda cuda_app test core_tests clean

help:
	@printf '%s\n' 'HPC fallback targets, no CMake/Ninja required:'
	@printf '%s\n' '  make openmp CXX=g++'
	@printf '%s\n' '  make mpi MPICXX=mpicxx'
	@printf '%s\n' '  make cuda NVCC=nvcc NVCC_CCBIN=g++'
	@printf '%s\n' '  make test CXX=g++'
	@printf '%s\n' ''
	@printf '%s\n' 'Use CMake presets for local development when CMake/Ninja are available.'

all: openmp mpi cuda

openmp: $(OPENMP_BIN)
openmp_app: openmp

mpi: $(MPI_BIN)
mpi_app: mpi

cuda: $(CUDA_BIN)
cuda_app: cuda

core_tests: $(TEST_BIN)

test: $(TEST_BIN)
	$<

$(OPENMP_BIN): $(OPENMP_SRC) $(MST_HEADERS) | $(BUILD_DIR)/openmp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CXXSTD) $(OPENMP_FLAGS) $< -o $@ $(LDFLAGS) $(OPENMP_FLAGS) $(LDLIBS)

$(MPI_BIN): $(MPI_SRC) $(MST_HEADERS) | $(BUILD_DIR)/mpi
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(CXXSTD) $(MPI_CXXFLAGS) $< -o $@ $(LDFLAGS) $(MPI_LDLIBS) $(LDLIBS)

$(CUDA_BIN): $(CUDA_SRC) $(MST_HEADERS) | $(BUILD_DIR)/cuda
	$(NVCC) $(CUDA_CPPFLAGS) $(CUDAFLAGS) $(CUDA_ARCH) $(NVCC_CCBIN_FLAG) $< -o $@ $(CUDA_LDFLAGS)

$(TEST_BIN): $(TEST_SRC) $(MST_HEADERS) | $(BUILD_DIR)/tests
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CXXSTD) $< -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD_DIR)/openmp $(BUILD_DIR)/mpi $(BUILD_DIR)/cuda $(BUILD_DIR)/tests:
	mkdir -p $@

clean:
	$(RM) $(OPENMP_BIN) $(MPI_BIN) $(CUDA_BIN) $(TEST_BIN)
