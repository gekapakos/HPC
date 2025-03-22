# Makefile for compiling a CUDA program

# Compiler
NVCC = nvcc

# Target executable name
TARGET = contrast-enhancement

# Source file
SRC = contrast-enhancement.cu histogram-equalization.cu main.cu

# Compiler flags
CFLAGS = -O4 -g #-G  # Modify the architecture to match your GPU's compute capability

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