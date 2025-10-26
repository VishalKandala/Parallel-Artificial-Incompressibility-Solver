#ifndef SOLVER_H
#define SOLVER_H

//-----------------------------------------------
// Vishal Indivar Kandala, 2022
// Computational Science and Biofluids Lab (CSBL), Texas A&M University.
//
// Backward Facing step problem for incompressible
// flow with Artificial Compressibility method.
// Parallel implementation using PETSc.
//-----------------------------------------------

#include <petsc.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

//================================================================================
// Simulation Parameters and Constants
//================================================================================
// These can be overridden at runtime with command-line options.

extern const PetscInt IM;    // Number of Nodes along x
extern const PetscInt JM;    // Number of Nodes along y
extern const PetscInt Nx;    // Number of cells along x
extern const PetscInt Ny;    // Number of cells along y
extern PetscReal rx;         // Stretching Ratio along x
extern PetscReal ry;         // Stretching Ratio along y
extern PetscReal tol;        // Tolerance for convergence
extern PetscReal Re;         // Reynolds Number
extern PetscReal U_inf;      // Free-stream average velocity.
extern PetscReal ex;         // Step Expansion Ratio
extern PetscReal dr;         // Domain Length Ratio
extern PetscReal cfl;        // CFL number.
extern PetscReal e;          // Artificial Dissipation coefficient.
extern PetscInt nmax;        // Max Iterations.

//================================================================================
// Global PETSc Objects
//================================================================================

extern DM da, cda;
extern DMDALocalInfo info;
extern PetscErrorCode ierr;
extern PetscInt rank;

// --- Global Vectors ---
// State Variables
extern Vec gu, gv, gp;
extern Vec guo, gvo, gpo; // Old values for time stepping

// Convective Fluxes
extern Vec gE_1, gE_2, gE_3;
extern Vec gE1_1, gE1_2, gE1_3;
extern Vec gE2_1, gE2_2, gE2_3;

// Viscous Fluxes
extern Vec gEv_1, gEv_2, gEv_3;
extern Vec gEv1_1, gEv1_2, gEv1_3;
extern Vec gEv2_1, gEv2_2, gEv2_3;

// Artificial Dissipation
extern Vec gD_1, gD_2, gD_3;
extern Vec gD1_1, gD1_2, gD1_3;
extern Vec gD1h_1, gD1h_2, gD1h_3;
extern Vec gD2_1, gD2_2, gD2_3;
extern Vec gD2h_1, gD2h_2, gD2h_3;

// Right-Hand Side
extern Vec gR_1, gR_2, gR_3;

// Grid Metrics and Derived Quantities
extern Vec gJ, gJx, gJy;
extern Vec gg11, gg22;
extern Vec grhox, grhoy;
extern Vec gU, gV; // Contravariant velocities

// Error and Output
extern Vec e1, e2, e3, ebuff;
extern Vec uout, vout, pout;
extern VecScatter uoutput, voutput, poutput;

// --- Local Vectors (for each process) ---
extern Vec lu, lv, lp;
extern Vec lE_1, lE_2, lE_3;
extern Vec lE1_1, lE1_2, lE1_3;
extern Vec lE2_1, lE2_2, lE2_3;
extern Vec lEv_1, lEv_2, lEv_3;
extern Vec lEv1_1, lEv1_2, lEv1_3;
extern Vec lEv2_1, lEv2_2, lEv2_3;
extern Vec lD_1, lD_2, lD_3;
extern Vec lD1_1, lD1_2, lD1_3;
extern Vec lD1h_1, lD1h_2, lD1h_3;
extern Vec lD2_1, lD2_2, lD2_3;
extern Vec lD2h_1, lD2h_2, lD2h_3;
extern Vec lJ, lU, lV, lrhoy;

//================================================================================
// Global C Variables
//================================================================================

// --- Grid Data (calculated on rank 0 and broadcast) ---
extern PetscReal x[151], y[151], dx[150], dy[150];
extern PetscReal xzeta[151], zetax[151], yeta[151], etay[151];

// --- Pointers for DMDAVecGetArray ---
extern PetscReal **ua, **va, **pa;
extern PetscReal **aE1_1, **aE1_2, **aE1_3, **aE2_1, **aE2_2, **aE2_3, **aE_1, **aE_2, **aE_3;
extern PetscReal **aEv1_1, **aEv1_2, **aEv1_3, **aEv2_1, **aEv2_2, **aEv2_3, **aEv_1, **aEv_2, **aEv_3;
extern PetscReal **aD1_1, **aD1_2, **aD1_3, **aD2_1, **aD2_2, **aD2_3, **aD_1, **aD_2, **aD_3;
extern PetscReal **aD1h_1, **aD1h_2, **aD1h_3, **aD2h_1, **aD2h_2, **aD2h_3;
extern PetscReal **aR_1, **aR_2, **aR_3;
extern const PetscReal *ru, *rv, *rp;
extern PetscReal **J, **Jx, **Jy, **rhox, **rhohy, **g11, **g22;
extern PetscReal **U, **V;

// --- Loop counters and state variables ---
extern PetscInt xs, lxs, xe, lxe, ys, lys, ye, lye, llxs, llxe, llys, llye, mx, my;
extern PetscReal t, dt, err;
extern PetscInt n, k, l, inflowy, Midx, Midy;
extern PetscBool conv;

//================================================================================
// Function Prototypes
//================================================================================

PetscErrorCode GenerateGrid();
PetscErrorCode GenerateTransforms();
PetscErrorCode GenerateMetrics();
PetscErrorCode Initialize();
PetscErrorCode PrepareTimeloop();
PetscErrorCode SetBCS();
PetscErrorCode GenerateE();
PetscErrorCode GenerateEv();
PetscErrorCode GenerateD();
PetscErrorCode GenerateR();
PetscErrorCode LSRK4();
PetscErrorCode NormalizePressure();
PetscErrorCode CheckConvergence();
PetscErrorCode March();
PetscErrorCode GenerateDatFile();

#endif // SOLVER_H
