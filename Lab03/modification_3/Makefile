# Makefile for compiling a CUDA program

# Compiler
NVCC = nvcc

# Target executable name
TARGET = Convolution2D

# Source file
SRC = Convolution2D.cu

# Compiler flags
CFLAGS = -O4  # Modify the architecture to match your GPU's compute capability

# Default rule
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

# Clean rule to remove the executable
clean:
	rm -f $(TARGET)

# Phony targets to avoid conflicts with files named "clean" or "all"
.PHONY: all clean
