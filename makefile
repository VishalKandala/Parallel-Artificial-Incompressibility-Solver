# Makefile for the PETSc-based CFD Solver

# Include PETSc's default variables and rules
# Ensure PETSC_DIR and PETSC_ARCH are set in your environment
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# Compiler and flags
# CFLAGS can be adjusted for optimization, e.g., -O3 for release builds
CFLAGS = -Wall -g -O2

# Source files
SOURCES_C = src/solver.c

# Object files (derived from source files)
OBJECTS_C = ${SOURCES_C:.c=.o}

# Executable name
EXECUTABLE = build/solver

# Phony targets do not represent actual files
.PHONY: all build clean run

# Default target: build the executable
all: build

# Rule to build the executable
build: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS_C) chkopts
	@echo "********************************************"
	@echo " Linking executable: $(EXECUTABLE)"
	@echo "********************************************"
	@mkdir -p build
	-${CLINKER} -o $(EXECUTABLE) $(OBJECTS_C) ${PETSC_LIB}
	@echo "Build complete. Executable is at $(EXECUTABLE)"

# Rule to clean up build files
clean:
	@echo "Cleaning up build files..."
	-@rm -f $(EXECUTABLE) $(OBJECTS_C)
	-@rm -rf build
	@echo "Clean complete."

# Example run target
run: build
	@echo "Running example simulation with 4 processes..."
	mpiexec -n 4 $(EXECUTABLE) -re 100 -nmax 1000

# Include PETSc's test rules (useful for more advanced makefiles)
include ${PETSC_DIR}/lib/petsc/conf/test
