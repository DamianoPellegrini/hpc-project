.PHONY: all tests openmp mpi cuda clean help

UNAME_S := $(shell uname -s)

CXX ?= g++
MPICXX ?= mpicxx
NVCC ?= nvcc
NVCC_CCBIN ?= $(CXX)
BUILD_DIR ?= build
CUDA_ARCH ?=
TESTS_DIR := $(BUILD_DIR)/tests
OPENMP_DIR := $(BUILD_DIR)/openmp
MPI_DIR := $(BUILD_DIR)/mpi
CUDA_DIR := $(BUILD_DIR)/cuda

COMMON_CPPFLAGS := -I. -Iinclude
COMMON_CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -pedantic
CUDA_CXXFLAGS ?= -std=c++20 -O2
COMMON_HEADERS := $(shell find include -type f \( -name '*.hpp' -o -name '*.h' \))

OPENMP_CXXFLAGS ?= -fopenmp
OPENMP_LDFLAGS ?= -fopenmp

ifeq ($(UNAME_S),Darwin)
OPENMP_ROOT ?= /opt/homebrew/opt/libomp
OPENMP_CXXFLAGS := -Xpreprocessor -fopenmp -I$(OPENMP_ROOT)/include
OPENMP_LDFLAGS := -L$(OPENMP_ROOT)/lib -lomp
endif

CUDA_ARCH_FLAGS := $(if $(CUDA_ARCH),-arch=$(CUDA_ARCH),)

ALL_TARGETS := tests openmp
ifneq ($(shell command -v $(MPICXX) >/dev/null 2>&1 && echo yes),)
ALL_TARGETS += mpi
endif
ifneq ($(shell command -v $(NVCC) >/dev/null 2>&1 && echo yes),)
ALL_TARGETS += cuda
endif

all: $(ALL_TARGETS)

$(TESTS_DIR) $(OPENMP_DIR) $(MPI_DIR) $(CUDA_DIR):
	mkdir -p $@

tests: $(TESTS_DIR)/core_tests

$(TESTS_DIR)/core_tests: tests/core_tests.cpp $(COMMON_HEADERS) | $(TESTS_DIR)
	$(CXX) $(COMMON_CXXFLAGS) $(COMMON_CPPFLAGS) $< -o $@

openmp: $(OPENMP_DIR)/openmp_app

$(OPENMP_DIR)/openmp_app: openmp/main.cpp $(COMMON_HEADERS) | $(OPENMP_DIR)
	$(CXX) $(COMMON_CXXFLAGS) $(OPENMP_CXXFLAGS) $(COMMON_CPPFLAGS) $< -o $@ $(OPENMP_LDFLAGS)

mpi: $(MPI_DIR)/mpi_app

$(MPI_DIR)/mpi_app: mpi/main.cpp $(COMMON_HEADERS) | $(MPI_DIR)
	@command -v $(MPICXX) >/dev/null 2>&1 || { echo "error: $(MPICXX) not found. Load an OpenMPI module or override MPICXX."; exit 1; }
	$(MPICXX) $(COMMON_CXXFLAGS) $(COMMON_CPPFLAGS) $< -o $@

cuda: $(CUDA_DIR)/cuda_app

$(CUDA_DIR)/cuda_app: cuda/main.cu $(COMMON_HEADERS) | $(CUDA_DIR)
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "error: $(NVCC) not found. Load a CUDA toolkit module or override NVCC."; exit 1; }
	@command -v $(NVCC_CCBIN) >/dev/null 2>&1 || { echo "error: $(NVCC_CCBIN) not found. Load a C++20-capable host compiler or override NVCC_CCBIN."; exit 1; }
	$(NVCC) $(CUDA_CXXFLAGS) $(CUDA_ARCH_FLAGS) -ccbin $(NVCC_CCBIN) $(COMMON_CPPFLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)

help:
	@printf '%s\n' \
	  'Targets:' \
	  '  make all      Build every backend available in PATH (tests/openmp plus mpi/cuda when present).' \
	  '  make tests     Build the dependency-free regression test binary.' \
	  '  make openmp    Build the OpenMP backend.' \
	  '  make mpi       Build the MPI backend (requires MPICXX in PATH).' \
	  '  make cuda      Build the CUDA backend (requires NVCC in PATH).' \
	  '  make clean     Remove build artifacts.' \
	  '' \
	  'Variables you can override:' \
	  '  CXX=<compiler>           Default: g++' \
	  '  MPICXX=<wrapper>         Default: mpicxx' \
	  '  NVCC=<compiler>          Default: nvcc' \
	  '  NVCC_CCBIN=<compiler>    Host compiler for nvcc, default: $(CXX)' \
	  '  CUDA_ARCH=<sm>           Example: 80, 86, 90a' \
	  '  BUILD_DIR=<dir>          Default: build'
