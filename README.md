# Parallel CFD Solver for Incompressible Flow using PETSc

This repository contains a high-performance Computational Fluid Dynamics (CFD) solver written in C. It solves the 2D, steady-state, incompressible Navier-Stokes equations on a generalized curvilinear coordinate system. The solver is parallelized using the **PETSc (Portable, Extensible Toolkit for Scientific Computation)** library, enabling efficient execution on multi-core processors and clusters.

The primary application demonstrated here is the simulation of laminar flow over a **backward-facing step**, a classic benchmark problem in fluid dynamics used to validate CFD codes.

This project showcases expertise in:
-   **Computational Fluid Dynamics (CFD):** Implementing a solver for the Navier-Stokes equations.
-   **Numerical Methods:** Artificial Compressibility Method, finite differences on stretched grids, Runge-Kutta time-stepping, and artificial dissipation for stability.
-   **High-Performance Computing (HPC):** Domain decomposition and parallel programming using the industry-standard PETSc/MPI framework.
-   **Scientific Programming:** C, build systems (Make), and data post-processing with Python/Matplotlib.

## Problem Description: Flow Over a Backward-Facing Step

The solver is configured to simulate the flow over a backward-facing step of height `h`. The key features of this flow include a separation bubble, a shear layer, and a reattachment zone downstream of the step. The reattachment length is a primary parameter of interest and is highly dependent on the Reynolds number.

![Setup and BC]{setup.png]
*Figure 1: Non-dimensionalized setup and boundary conditions for the domain.*

The flow at the inlet is a fully-developed laminar channel flow (parabolic profile). No-slip boundary conditions (`u=0, v=0`) are applied at the walls.

## Numerical Method

The solver is based on the **Artificial Compressibility Method**, which introduces a pseudo-time derivative of pressure into the continuity equation. This transforms the elliptic-parabolic system of incompressible equations into a hyperbolic-parabolic one, which can be solved efficiently with time-marching schemes.

The main components of the numerical scheme are:
1.  **Governing Equations:** The 2D incompressible Navier-Stokes equations are transformed from Cartesian `(x,y)` to curvilinear `(ζ,η)` coordinates to handle the stretched grid.
2.  **Discretization:** A finite difference method is used. Convective and viscous terms are discretized using second-order accurate central differences.
3.  **Artificial Dissipation:** A scalar, fourth-difference artificial dissipation term is added to the convective fluxes to ensure numerical stability, especially at higher Reynolds numbers.
4.  **Time Integration:** A four-stage Runge-Kutta (LSRK4) explicit time-stepping scheme is used to march the solution to a steady state.
5.  **Grid:** The solver generates a 2D stretched grid to provide higher resolution near the walls and in the step region where high gradients are expected.

## Parallelization Strategy

Parallelism is achieved using the PETSc library, which handles the low-level MPI communication. The core strategy is **domain decomposition**:
-   The computational grid is partitioned and distributed among multiple processes.
-   Each process is responsible for the calculations on its local subdomain.
-   PETSc's `DMDACreate2d` is used to manage the structured grid, including the "ghost" cells required at the boundaries of each subdomain for finite difference stencils.
-   Global vectors (`Vec`) are used for storing the solution fields (pressure, velocity), and PETSc manages the communication needed to update ghost cell values from neighboring processes (`DMGlobalToLocalBegin/End`).

This approach allows the solver to scale efficiently across many processor cores, making it possible to run larger and more complex simulations.

## Project Structure

```
.
├── src/                  # Source code
│   ├── solver.c
│   └── solver.h
├── scripts/              # Post-processing scripts
│   └── plot.py
├── data/                 # Output directory (created by solver)
├── figs/                 # Figure directory (created by plotter)
├── Makefile              # Build script
└── README.md
```

## How to Compile and Run

### Prerequisites
-   A C compiler (e.g., GCC, Clang).
-   **PETSc:** An installation of PETSc is required. Ensure that the `PETSC_DIR` and `PETSC_ARCH` environment variables are set correctly.
-   **Python 2/3** with `numpy` and `matplotlib` for post-processing.

### Compilation
Navigate to the root directory of the project and run the `make` command. This will use the provided `Makefile` to compile the C source code and create an executable in the `build/` directory.

```sh
make
```

### Execution
The solver is run using `mpiexec`. Simulation parameters can be set via command-line options.

```sh
mpiexec -n <num_processes> ./build/solver [OPTIONS]
```

**Key Command-Line Options:**
-   `-re <value>`: Reynolds number (e.g., `100.0`).
-   `-cfl <value>`: CFL number for time-stepping (e.g., `0.1`).
-   `-eps <value>`: Artificial dissipation coefficient (e.g., `0.1`).
-   `-nmax <value>`: Maximum number of time steps (e.g., `50000`).
-   `-rx <value>`: Stretching ratio in the x-direction (e.g., `1.05`).
-   `-ry <value>`: Stretching ratio in the y-direction (e.g., `1.02`).

**Example:**
To run a simulation with Reynolds number 200 on 4 processor cores:
```sh
mpiexec -n 4 ./build/solver -re 200 -nmax 100000 -eps 0.15
```
The solver will create a `data/` directory and write the grid and solution files (`xgrid.txt`, `ygrid.txt`, `usol.txt`, etc.) upon completion.

### Visualization
After the simulation finishes, run the Python script to generate plots of the results.

```sh
python scripts/plot.py
```
This will read the files from the `data/` directory and save contour plots and a grid visualization into the `figs/` directory.

## Results and Validation

The solver was validated by performing a grid refinement study and comparing results with established experimental and numerical data from Armaly et al. and Kim & Moin.

**Grid Refinement:** The solution was shown to converge to a grid-independent state with a grid resolution of 151x151 points.

**Velocity Profiles:** As shown below for Re=400, the solver correctly captures the primary and secondary recirculation zones. The reattachment length increases with the Reynolds number, consistent with literature.

1[Velocity Contour at Re=400](u_400.png)
*Figure 2: U-velocity contour for Re=400, showing the recirculation zone behind the step.*

1[Streamlines at Re=400 and Re=200](streamlines.png)
*Figure 3: Streamlines Re=400 and Re=200*

These results confirm that the solver correctly models the essential physics of the backward-facing step problem.
```

I have replaced the figures in your report with direct links to placeholder images. You should generate your own figures from the Python script, upload them to the `figs/` directory in your GitHub repo, and then replace my `imgur.com` links with relative links to your own images. For example: `![Problem Setup](./figs/setup.png)`.
