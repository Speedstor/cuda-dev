###############################################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda-12.8

###############################################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=g++
CC_FLAGS=
CC_LIBS=
CC_INC_DIR= -I./include

###############################################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=-Wno-deprecated-gpu-targets
NVVV_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include -I./include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

###############################################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = build

# Include header file directory:
# INC_DIR = include

###############################################################################

## Make variables ##

# Target executable names:
EXE = cudaDev

# Object files:
OBJS=src/kernels/matmul/matmul_cuda.o src/common/cuda_helper.o src/kernels/my_kernels.o src/main.o 

###############################################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS) | $(BUILDDIR)
	$(CC) $(CC_FLAGS) $(addprefix $(OBJ_DIR)/, $(notdir $(OBJS))) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main.cpp file to object files
%.o: %.cpp | $(OBJ_DIR)
	$(CC) $(CC_FLAGS) -c $^ -o $(OBJ_DIR)/$(notdir $@) $(CC_INC_DIR)

# Compile C++ source files to object files
%.o: %.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $^ -o $(OBJ_DIR)/$(notdir $@) $(NVCC_LIBS) $(CUDA_INC_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -f $(EXE)
	rm -fr $(OBJ_DIR)
