# Makefile for nbody project

# Compiler
CC = icx

# Compiler flags
CFLAGS = -Wall -fast -qopenmp

# Source files
SRC = nbody.c

# Header files
HEADERS = timer.h

# Output executable
TARGET = nbody

# Build target
$(TARGET): $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

# Clean target
.PHONY: clean
clean:
	rm -f $(TARGET)