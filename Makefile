# ==========================================
# MTP Pruning - Makefile
# ==========================================
# Override any auto-detected value from the command line, e.g.:
#   make BLAS=-lmkl_rt CXX=g++

# --- Compiler ---
# Prefer Intel compilers (icpx > icpc > g++). MPI wrapper is used if available.
# On HPC clusters, load the appropriate module first:
#   module load intel  (for icpx/MKL)
#   module load mpi    (for mpicxx)
ifneq ($(shell command -v icpx 2>/dev/null),)
    BASE_CXX := icpx
else ifneq ($(shell command -v icpc 2>/dev/null),)
    BASE_CXX := icpc
else
    BASE_CXX := g++
endif

USE_MPI ?= 1
ifneq ($(shell command -v mpicxx 2>/dev/null),)
    ifeq ($(USE_MPI),1)
        CXX      := mpicxx
        CPPFLAGS += -DUSE_MPI
        OBJ_DIR  := obj/mpi
    endif
endif
CXX     ?= $(BASE_CXX)
OBJ_DIR ?= obj/serial

# --- BLAS/LAPACK (sequential — no hidden threads fighting MPI ranks) ---
# Detection order:
#   1. Intel compiler    → -qmkl=sequential (built-in, no paths needed)
#   2. $MKLROOT set      → explicit sequential MKL (HPC module system)
#   3. Fedora/RHEL       → openblas-serial RPM (-lopenblas-serial)
#   4. pkg-config        → Debian/Ubuntu libopenblas-dev
#   5. libopenblas.so    → conda, manual installs, /usr/local
#   6. Fallback          → netlib -lblas (slow, warns)
ifneq ($(filter icpx icpc,$(BASE_CXX)),)
    BLAS := -qmkl=sequential
    $(info BLAS: Intel MKL sequential (compiler flag))
else ifdef MKLROOT
    BLAS := -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
    $(info BLAS: Intel MKL sequential (MKLROOT))
else ifneq ($(shell find /usr/lib64 /usr/lib -name "libopenblas-serial.so*" 2>/dev/null | head -1),)
    BLAS := -lopenblas-serial -lm -ldl
    $(info BLAS: OpenBLAS serial (Fedora/RHEL))
else ifneq ($(shell pkg-config --exists openblas 2>/dev/null && echo 1),)
    BLAS := $(shell pkg-config --libs openblas) -lm -ldl
    $(info BLAS: OpenBLAS (pkg-config))
else ifneq ($(shell find /usr/lib64 /usr/lib /usr/local/lib -name "libopenblas.so*" 2>/dev/null | head -1),)
    BLAS := -lopenblas -lm -ldl
    $(info BLAS: OpenBLAS (filesystem))
else
    BLAS := -llapack -lblas -lm -ldl
    $(warning BLAS: Netlib reference BLAS (SLOW). Install openblas-serial (Fedora) or libopenblas-dev (Debian).)
endif

LDLIBS := $(BLAS)

# --- Flags ---
# No -fopenmp/-qopenmp: prevents OpenBLAS/MKL from spawning threads inside MPI ranks.
CPPFLAGS += -Iexternal -Isrc
CXXFLAGS += -O3 -std=c++17 -march=native -pthread

# --- Files ---
SRC_DIR := src
TARGET  := bin/prune
SRC     := $(shell find $(SRC_DIR) -name "*.cpp")
OBJ     := $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# --- Rules ---
$(TARGET): $(OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@
	@echo "Built: $@ | CXX=$(CXX) | $(BLAS)"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	@rm -rf obj bin

# On HPC, ensure BLAS doesn't spawn threads behind MPI's back:
#   export OPENBLAS_NUM_THREADS=1
#   export MKL_NUM_THREADS=1