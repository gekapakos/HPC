# Makefile for nbody project

# Compiler
CC = nvcc

# Compiler flags
CFLAGS = -I. -arch=sm_37 --use_fast_math -ftz=true

# Source files
SRC = nbody.cu

# Header files
HEADERS = timer.h

# Output executable
TARGET = nbody

# Build target
$(TARGET): $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

# Clean target
.PHONY: clean
clean:
	rm -f $(TARGET)
