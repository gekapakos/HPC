# Compiler
CC = icx

# Compiler flags
CFLAGS = -qopenmp -Wall -fast

# Source files
SRC = contrast-enhancement.c histogram-equalization.c main.c

# Header files
HEADERS = hist-equ.h

# Output executable
TARGET = contrast-enhancement

# Build the program
all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

# Clean the build
clean:
	rm -f $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)
