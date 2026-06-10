# Makefile di fallback per il cluster: nessuna dipendenza da CMake/Ninja.
#
# CMake + Ninja restano il percorso di build principale per lo sviluppo locale
# (vedi CMakePresets.json). Questo file è scritto a mano per gli ambienti Slurm
# dove CMake/Ninja non sono disponibili sui nodi di calcolo.

.DEFAULT_GOAL := help

BUILD_DIR ?= build

CXX ?= g++
MPICXX ?= mpicxx
NVCC ?= nvcc
NVCC_CCBIN ?=

CXXSTD ?= -std=c++17
CXXFLAGS ?= -O3 -Wall -Wextra
LDFLAGS ?=
LDLIBS ?=

OPENMP_FLAGS ?= -fopenmp
MPI_CXXFLAGS ?=
MPI_LDLIBS ?=

CUDAFLAGS ?= -O3 -std=c++17
CUDA_ARCH ?=
CUDA_LDFLAGS ?=

ifneq ($(strip $(NVCC_CCBIN)),)
NVCC_CCBIN_FLAG := -ccbin $(NVCC_CCBIN)
endif

SEQUENTIAL_SRC := src/sequential.cpp
OPENMP_SRC := src/openmp.cpp
MPI_SRC := src/mpi.cpp
CUDA_SRC := src/cuda.cu

SEQUENTIAL_BIN := $(BUILD_DIR)/sequential_app
OPENMP_BIN := $(BUILD_DIR)/openmp_app
MPI_BIN := $(BUILD_DIR)/mpi_app
CUDA_BIN := $(BUILD_DIR)/cuda_app

.PHONY: help all sequential openmp mpi cuda clean

help:
	@printf '%s\n' 'Target per il cluster (Slurm), senza CMake/Ninja:'
	@printf '%s\n' '  make sequential CXX=g++'
	@printf '%s\n' '  make openmp CXX=g++'
	@printf '%s\n' '  make mpi MPICXX=mpicxx'
	@printf '%s\n' '  make cuda NVCC=nvcc NVCC_CCBIN=g++'
	@printf '%s\n' '  make all'
	@printf '%s\n' ''
	@printf '%s\n' 'Per lo sviluppo locale preferire i preset CMake (Ninja).'

all: sequential openmp mpi cuda

sequential: $(SEQUENTIAL_BIN)
openmp: $(OPENMP_BIN)
mpi: $(MPI_BIN)
cuda: $(CUDA_BIN)

$(SEQUENTIAL_BIN): $(SEQUENTIAL_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CXXSTD) $< -o $@ $(LDFLAGS) $(LDLIBS)

$(OPENMP_BIN): $(OPENMP_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CXXSTD) $(OPENMP_FLAGS) $< -o $@ $(LDFLAGS) $(OPENMP_FLAGS) $(LDLIBS)

$(MPI_BIN): $(MPI_SRC) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(CXXSTD) $(MPI_CXXFLAGS) $< -o $@ $(LDFLAGS) $(MPI_LDLIBS) $(LDLIBS)

$(CUDA_BIN): $(CUDA_SRC) | $(BUILD_DIR)
	$(NVCC) $(CUDAFLAGS) $(CUDA_ARCH) $(NVCC_CCBIN_FLAG) $< -o $@ $(CUDA_LDFLAGS)

$(BUILD_DIR):
	mkdir -p $@

clean:
	$(RM) -r $(BUILD_DIR)
