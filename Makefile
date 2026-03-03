# ==========================================
# MTP Pruning - Auto-Detecting Makefile
# ==========================================

# 1. Auto-detect Base C++ Compiler (icpx > icpc > g++)
ifeq ($(shell command -v icpx 2> /dev/null),)
    ifeq ($(shell command -v icpc 2> /dev/null),)
        BASE_CXX := g++
    else
        BASE_CXX := icpc
    endif
else
    BASE_CXX := icpx
endif

# 2. Auto-detect LAPACK/BLAS based on compiler
ifneq (,$(filter icpx icpc,$(BASE_CXX)))
    LDLIBS := -qmkl
else
    ifdef MKLROOT
        LDLIBS := -L$(MKLROOT)/lib/intel64 -lmkl_rt
    else
        LDLIBS := -llapack -lblas
    endif
endif

# 3. Auto-detect MPI
USE_MPI ?= 1
ifeq ($(USE_MPI), 1)
    ifneq ($(shell command -v mpicxx 2> /dev/null),)
        CXX := mpicxx
        CPPFLAGS += -DUSE_MPI
        OBJ_DIR := obj/mpi
        $(info -> MPI detected: Using mpicxx)
    else
        $(info -> Warning: mpicxx not found. Falling back to Serial mode.)
        CXX := $(BASE_CXX)
        OBJ_DIR := obj/serial
    endif
else
    CXX := $(BASE_CXX)
    OBJ_DIR := obj/serial
    $(info -> MPI explicitly disabled. Using Serial mode.)
endif

$(info -> Base Compiler: $(BASE_CXX))
$(info -> LAPACK Linker: $(LDLIBS))

# 4. Compilation Flags
CPPFLAGS += -Iexternal -Isrc
CXXFLAGS += -O3 -std=c++17 -fPIC -march=native
# CXXFLAGS += -O0 -g -std=c++17 -fPIC -march=native

# 5. Files and Directories
SRC_DIR := src
BIN_DIR := bin
TARGET := $(BIN_DIR)/prune

SRC := $(shell find $(SRC_DIR) -name "*.cpp")
OBJ := $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# ==========================================
# Build Rules
# ==========================================

.PHONY: all clean clean-all

all: $(TARGET)

# Compile objects (creates directories dynamically)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Link executable
$(TARGET): $(OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
	@echo "=========================================="
	@echo "Build successful: $(TARGET)"
	@echo "=========================================="

# Cleanup
clean:
	@echo "Cleaning compiled objects..."
	@rm -rf obj

clean-all: clean
	@echo "Cleaning binaries..."
	@rm -rf bin