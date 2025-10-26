//-----------------------------------------------
// Vishal Indivar Kandala, 2022
// Computational Science and Biofluids Lab (CSBL), Texas A&M University.
//
// Backward Facing step problem for incompressible
// flow with Artificial Compressibility method.
// Parallel implementation using PETSc.
//-----------------------------------------------
#include "solver.h"

//================================================================================
// Global Variable Definitions
//================================================================================

// --- Simulation Parameters ---
const PetscInt IM = 151;
const PetscInt JM = 151;
const PetscInt Nx = IM - 1;
const PetscInt Ny = JM - 1;
PetscReal rx = 1.0;
PetscReal ry = 1.0;
PetscReal tol = 1.0e-6;
PetscReal Re = 100.0;
PetscReal U_inf = 1.0;
PetscReal ex = 0.5;
PetscReal dr = 15.0;
PetscReal cfl = 0.1;
PetscReal e = 0.1;
PetscInt nmax = 200000;

// --- PETSc Objects ---
DM da, cda;
DMDALocalInfo info;
PetscErrorCode ierr;
PetscInt rank;

// --- Global and Local Vectors (defined without extern) ---
Vec gu, gv, gp, guo, gvo, gpo;
Vec gE_1, gE_2, gE_3, gE1_1, gE1_2, gE1_3, gE2_1, gE2_2, gE2_3;
Vec gEv_1, gEv_2, gEv_3, gEv1_1, gEv1_2, gEv1_3, gEv2_1, gEv2_2, gEv2_3;
Vec gD_1, gD_2, gD_3, gD1_1, gD1_2, gD1_3, gD1h_1, gD1h_2, gD1h_3;
Vec gD2_1, gD2_2, gD2_3, gD2h_1, gD2h_2, gD2h_3;
Vec gR_1, gR_2, gR_3;
Vec gJ, gJx, gJy, gg11, gg22, grhox, grhoy, gU, gV;
Vec e1, e2, e3, ebuff;
Vec uout, vout, pout;
VecScatter uoutput, voutput, poutput;
Vec lu, lv, lp;
Vec lE_1, lE_2, lE_3, lE1_1, lE1_2, lE1_3, lE2_1, lE2_2, lE2_3;
Vec lEv_1, lEv_2, lEv_3, lEv1_1, lEv1_2, lEv1_3, lEv2_1, lEv2_2, lEv2_3;
Vec lD_1, lD_2, lD_3, lD1_1, lD1_2, lD1_3, lD1h_1, lD1h_2, lD1h_3;
Vec lD2_1, lD2_2, lD2_3, lD2h_1, lD2h_2, lD2h_3;
Vec lJ, lU, lV, lrhoy;

// --- Global C Variables ---
PetscReal x[151], y[151], dx[150], dy[150];
PetscReal xzeta[151], zetax[151], yeta[151], etay[151];
PetscReal **ua, **va, **pa;
PetscReal **aE1_1, **aE1_2, **aE1_3, **aE2_1, **aE2_2, **aE2_3, **aE_1, **aE_2, **aE_3;
PetscReal **aEv1_1, **aEv1_2, **aEv1_3, **aEv2_1, **aEv2_2, **aEv2_3, **aEv_1, **aEv_2, **aEv_3;
PetscReal **aD1_1, **aD1_2, **aD1_3, **aD2_1, **aD2_2, **aD2_3, **aD_1, **aD_2, **aD_3;
PetscReal **aD1h_1, **aD1h_2, **aD1h_3, **aD2h_1, **aD2h_2, **aD2h_3;
PetscReal **aR_1, **aR_2, **aR_3;
const PetscReal *ru, *rv, *rp;
PetscReal **J, **Jx, **Jy, **rhox, **rhohy, **g11, **g22;
PetscReal **U, **V;
PetscInt xs, lxs, xe, lxe, ys, lys, ye, lye, llxs, llxe, llys, llye, mx, my;
PetscReal t = 0.0, dt, err = 0.0;
PetscInt n = 0, k, l, inflowy, Midx, Midy;
PetscBool conv = PETSC_FALSE;


//********************************************************************************
//                            MAIN FUNCTION
//********************************************************************************
int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, NULL, "2D Incompressible Navier-Stokes Solver for flow over a backward-facing step.");
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // --- Read runtime options ---
    PetscOptionsGetReal(PETSC_NULL, "-re", &Re, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-cfl", &cfl, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-eps", &e, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-ry", &ry, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-rx", &rx, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-nmax", &nmax, PETSC_NULL);

    // --- PETSc DMDA Setup for the structured grid ---
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, IM, JM, PETSC_DECIDE, PETSC_DECIDE, 1, 2, NULL, NULL, &da); CHKERRQ(ierr);
    DMDASetUniformCoordinates(da, 0, 1, 0, 1, 0, 0);
    DMGetCoordinateDM(da, &cda);
    DMDAGetLocalInfo(da, &info);

    // --- Vector Allocation (Global) ---
    DMCreateGlobalVector(da, &gu); VecZeroEntries(gu);
    VecDuplicate(gu, &gv); VecDuplicate(gu, &gp);
    VecDuplicate(gu, &guo); VecDuplicate(gu, &gpo); VecDuplicate(gu, &gvo);
    VecDuplicate(gu, &gE1_1); VecDuplicate(gu, &gE1_2); VecDuplicate(gu, &gE1_3);
    VecDuplicate(gu, &gE2_1); VecDuplicate(gu, &gE2_2); VecDuplicate(gu, &gE2_3);
    VecDuplicate(gu, &gE_1); VecDuplicate(gu, &gE_2); VecDuplicate(gu, &gE_3);
    VecDuplicate(gu, &gEv1_1); VecDuplicate(gu, &gEv1_2); VecDuplicate(gu, &gEv1_3);
    VecDuplicate(gu, &gEv2_1); VecDuplicate(gu, &gEv2_2); VecDuplicate(gu, &gEv2_3);
    VecDuplicate(gu, &gEv_1); VecDuplicate(gu, &gEv_2); VecDuplicate(gu, &gEv_3);
    VecDuplicate(gu, &gD1_1); VecDuplicate(gu, &gD1_2); VecDuplicate(gu, &gD1_3);
    VecDuplicate(gu, &gD1h_1); VecDuplicate(gu, &gD1h_2); VecDuplicate(gu, &gD1h_3);
    VecDuplicate(gu, &gD2_1); VecDuplicate(gu, &gD2_2); VecDuplicate(gu, &gD2_3);
    VecDuplicate(gu, &gD2h_1); VecDuplicate(gu, &gD2h_2); VecDuplicate(gu, &gD2h_3);
    VecDuplicate(gu, &gD_1); VecDuplicate(gu, &gD_2); VecDuplicate(gu, &gD_3);
    VecDuplicate(gu, &gR_1); VecDuplicate(gu, &gR_2); VecDuplicate(gu, &gR_3);
    VecDuplicate(gu, &gJ); VecDuplicate(gu, &gJx); VecDuplicate(gu, &gJy);
    VecDuplicate(gu, &gg11); VecDuplicate(gu, &gg22);
    VecDuplicate(gu, &grhox); VecDuplicate(gu, &grhoy);
    VecDuplicate(gu, &gU); VecDuplicate(gu, &gV);
    VecDuplicate(gu, &e1); VecDuplicate(gu, &e2); VecDuplicate(gu, &e3); VecDuplicate(gu, &ebuff);
    
    // --- Vector Allocation (Local) ---
    DMCreateLocalVector(da, &lu); VecZeroEntries(lu);
    VecDuplicate(lu, &lv); VecDuplicate(lu, &lp);
    VecDuplicate(lu, &lE1_1); VecDuplicate(lu, &lE1_2); VecDuplicate(lu, &lE1_3);
    VecDuplicate(lu, &lE2_1); VecDuplicate(lu, &lE2_2); VecDuplicate(lu, &lE2_3);
    VecDuplicate(lu, &lE_1); VecDuplicate(lu, &lE_2); VecDuplicate(lu, &lE_3);
    VecDuplicate(lu, &lEv1_1); VecDuplicate(lu, &lEv1_2); VecDuplicate(lu, &lEv1_3);
    VecDuplicate(lu, &lEv2_1); VecDuplicate(lu, &lEv2_2); VecDuplicate(lu, &lEv2_3);
    VecDuplicate(lu, &lEv_1); VecDuplicate(lu, &lEv_2); VecDuplicate(lu, &lEv_3);
    VecDuplicate(lu, &lD1_1); VecDuplicate(lu, &lD1_2); VecDuplicate(lu, &lD1_3);
    VecDuplicate(lu, &lD1h_1); VecDuplicate(lu, &lD1h_2); VecDuplicate(lu, &lD1h_3);
    VecDuplicate(lu, &lD2_1); VecDuplicate(lu, &lD2_2); VecDuplicate(lu, &lD2_3);
    VecDuplicate(lu, &lD2h_1); VecDuplicate(lu, &lD2h_2); VecDuplicate(lu, &lD2h_3);
    VecDuplicate(lu, &lD_1); VecDuplicate(lu, &lD_2); VecDuplicate(lu, &lD_3);
    VecDuplicate(lu, &lJ); VecDuplicate(lu, &lU); VecDuplicate(lu, &lV); VecDuplicate(lu, &lrhoy);

    // --- Solver Execution ---
    GenerateGrid();
    Initialize();
    PetscBarrier(PETSC_NULL);
    GenerateMetrics();
    PrepareTimeloop();
    March();
    
    // --- Memory Cleanup ---
    VecDestroy(&gE_1); VecDestroy(&gE_2); VecDestroy(&gE_3); VecDestroy(&lE_1); VecDestroy(&lE_2); VecDestroy(&lE_3);
    VecDestroy(&gE1_1); VecDestroy(&gE1_2); VecDestroy(&gE1_3); VecDestroy(&lE1_1); VecDestroy(&lE1_2); VecDestroy(&lE1_3);
    VecDestroy(&gE2_1); VecDestroy(&gE2_2); VecDestroy(&gE2_3); VecDestroy(&lE2_1); VecDestroy(&lE2_2); VecDestroy(&lE2_3);
    VecDestroy(&gEv_1); VecDestroy(&gEv_2); VecDestroy(&gEv_3); VecDestroy(&lEv_1); VecDestroy(&lEv_2); VecDestroy(&lEv_3);
    VecDestroy(&gEv1_1); VecDestroy(&gEv1_2); VecDestroy(&gEv1_3); VecDestroy(&lEv1_1); VecDestroy(&lEv1_2); VecDestroy(&lEv1_3);
    VecDestroy(&gEv2_1); VecDestroy(&gEv2_2); VecDestroy(&gEv2_3); VecDestroy(&lEv2_1); VecDestroy(&lEv2_2); VecDestroy(&lEv2_3);
    VecDestroy(&gD_1); VecDestroy(&gD_2); VecDestroy(&gD_3); VecDestroy(&lD_1); VecDestroy(&lD_2); VecDestroy(&lD_3);
    VecDestroy(&gD1_1); VecDestroy(&gD1_2); VecDestroy(&gD1_3); VecDestroy(&lD1_1); VecDestroy(&lD1_2); VecDestroy(&lD1_3);
    VecDestroy(&gD1h_1); VecDestroy(&gD1h_2); VecDestroy(&gD1h_3); VecDestroy(&lD1h_1); VecDestroy(&lD1h_2); VecDestroy(&lD1h_3);
    VecDestroy(&gD2_1); VecDestroy(&gD2_2); VecDestroy(&gD2_3); VecDestroy(&lD2_1); VecDestroy(&lD2_2); VecDestroy(&lD2_3);
    VecDestroy(&gD2h_1); VecDestroy(&gD2h_2); VecDestroy(&gD2h_3); VecDestroy(&lD2h_1); VecDestroy(&lD2h_2); VecDestroy(&lD2h_3);
    VecDestroy(&gR_1); VecDestroy(&gR_2); VecDestroy(&gR_3);
    VecDestroy(&guo); VecDestroy(&e1); VecDestroy(&gvo); VecDestroy(&e2); VecDestroy(&gpo); VecDestroy(&e3); VecDestroy(&ebuff);
    VecDestroy(&gu); VecDestroy(&lu); VecDestroy(&gv); VecDestroy(&lv); VecDestroy(&gp); VecDestroy(&lp);
    VecDestroy(&lU); VecDestroy(&lV); VecDestroy(&gU); VecDestroy(&gV);
    VecDestroy(&grhox); VecDestroy(&grhoy); VecDestroy(&lrhoy);
    VecDestroy(&gJx); VecDestroy(&gJy); VecDestroy(&gg11); VecDestroy(&lJ); VecDestroy(&gJ); VecDestroy(&gg22);
    VecDestroy(&uout); VecDestroy(&vout); VecDestroy(&pout);
    DMDestroy(&da); DMDestroy(&cda);
    
    PetscFinalize();
    return 0;
}

/**
 * @brief Main time-marching loop that advances the solution to a steady state.
 */
PetscErrorCode March() {
    FILE *tfp;
    const char tmon[] = "data/tmon.txt";
    PetscFOpen(PETSC_COMM_WORLD, tmon, "w", &tfp);

    while (conv == PETSC_FALSE) {
        t = t + dt;
        n = n + 1;
        LSRK4();
        SetBCS();
        NormalizePressure();
        CheckConvergence();

        if (isnan(err) != 0) {
            PetscPrintf(PETSC_COMM_WORLD, "Divergence at Timestep:%d\n", n);
            break;
        }
        if (n % 1000 == 0) {
            PetscPrintf(PETSC_COMM_WORLD, "n err -- %d %lf\n", n, err);
            PetscFPrintf(PETSC_COMM_WORLD, tfp, "%d,%lf\n", n, err);
        }
        if (n >= nmax) {
            break;
        }
    }
    if (isnan(err) != 0) {
        PetscPrintf(PETSC_COMM_WORLD, "Divergence at Timestep:%d\n", n);
    }
    PetscFClose(PETSC_COMM_WORLD, tfp);
    ierr = GenerateDatFile();
    return 0;
}

/**
 * @brief Implements the four-stage, low-storage Runge-Kutta (LSRK4) scheme.
 */
PetscErrorCode LSRK4() {
    PetscReal a[4] = {1.0, 0.5, 0.3333333333, 0.25};
    
    ierr = GenerateR();
    VecCopy(gp, gpo);
    VecCopy(gu, guo);
    VecCopy(gv, gvo);

    for (k = 3; k >= 0; k--) {
        VecWAXPY(gp, a[k] * dt, gR_1, gpo);
        VecWAXPY(gu, a[k] * dt, gR_2, guo);
        VecWAXPY(gv, a[k] * dt, gR_3, gvo);
        if (k > 0) {
            ierr = GenerateR();
        }
    }
    
    if (e > 0.0) {
        GenerateD();
        VecAXPY(gp, -1.0 * dt, gD_1);
        VecAXPY(gu, -1.0 * dt, gD_2);
        VecAXPY(gv, -1.0 * dt, gD_3);
    }
    return 0;
}

/**
 * @brief Assembles the Right-Hand Side (RHS) vector: RHS = Ev - E.
 */
PetscErrorCode GenerateR() {
    xs = info.xs; xe = info.xs + info.xm; ys = info.ys; ye = info.ys + info.ym;
    mx = info.mx; my = info.my;
    lxs = (xs == 0) ? xs + 1 : xs; lys = (ys == 0) ? ys + 1 : ys;
    lxe = (xe == mx) ? xe - 1 : xe; lye = (ye == my) ? ye - 1 : ye;
    
    ierr = GenerateE();
    ierr = GenerateEv();

    DMDAVecGetArray(da, gR_1, &aR_1); DMDAVecGetArray(da, gR_2, &aR_2); DMDAVecGetArray(da, gR_3, &aR_3);
    DMDAVecGetArray(da, gE_1, &aE_1); DMDAVecGetArray(da, gE_2, &aE_2); DMDAVecGetArray(da, gE_3, &aE_3);
    DMDAVecGetArray(da, gEv_1, &aEv_1); DMDAVecGetArray(da, gEv_2, &aEv_2); DMDAVecGetArray(da, gEv_3, &aEv_3);

    for (j = lys; j < lye; j++) {
        for (i = lxs; i < lxe; i++) {
            aR_1[j][i] = aEv_1[j][i] - aE_1[j][i];
            aR_2[j][i] = aEv_2[j][i] - aE_2[j][i];
            aR_3[j][i] = aEv_3[j][i] - aE_3[j][i];
        }
    }
    
    DMDAVecRestoreArray(da, gR_1, &aR_1); DMDAVecRestoreArray(da, gR_2, &aR_2); DMDAVecRestoreArray(da, gR_3, &aR_3);
    DMDAVecRestoreArray(da, gE_1, &aE_1); DMDAVecRestoreArray(da, gE_2, &aE_2); DMDAVecRestoreArray(da, gE_3, &aE_3);
    DMDAVecRestoreArray(da, gEv_1, &aEv_1); DMDAVecRestoreArray(da, gEv_2, &aEv_2); DMDAVecRestoreArray(da, gEv_3, &aEv_3);
    
    return 0;
}

/**
 * @brief Calculates the artificial dissipation terms (D).
 */
PetscErrorCode GenerateD() {
    PetscReal Uh, Vh;
    
    xs = info.xs; xe = info.xs + info.xm; ys = info.ys; ye = info.ys + info.ym;
    mx = info.mx; my = info.my;
    lxs = (xs == 0) ? xs + 1 : xs; lys = (ys == 0) ? ys + 1 : ys;
    lxe = (xe == mx) ? xe - 1 : xe; lye = (ye == my) ? ye - 1 : ye;
    llxs = xs; llxe = xe; llys = ys; llye = ye;
    if (xe == mx) llxe = xe - 2;
    if (ye == my) llye = ye - 2;

    DMGlobalToLocalBegin(da, gp, INSERT_VALUES, lp); DMGlobalToLocalEnd(da, gp, INSERT_VALUES, lp);
    DMGlobalToLocalBegin(da, gu, INSERT_VALUES, lu); DMGlobalToLocalEnd(da, gu, INSERT_VALUES, lu);
    DMGlobalToLocalBegin(da, gv, INSERT_VALUES, lv); DMGlobalToLocalEnd(da, gv, INSERT_VALUES, lv);
    
    DMDAVecGetArray(da, gD1_1, &aD1_1); DMDAVecGetArray(da, gD1h_1, &aD1h_1);
    DMDAVecGetArray(da, gD1_2, &aD1_2); DMDAVecGetArray(da, gD1h_2, &aD1h_2);
    DMDAVecGetArray(da, gD1_3, &aD1_3); DMDAVecGetArray(da, gD1h_3, &aD1h_3);
    DMDAVecGetArray(da, gD2_1, &aD2_1); DMDAVecGetArray(da, gD2h_1, &aD2h_1);
    DMDAVecGetArray(da, gD2_2, &aD2_2); DMDAVecGetArray(da, gD2h_2, &aD2h_2);
    DMDAVecGetArray(da, gD2_3, &aD2_3); DMDAVecGetArray(da, gD2h_3, &aD2h_3);
    DMDAVecGetArray(da, gD_1, &aD_1); DMDAVecGetArray(da, gD_2, &aD_2); DMDAVecGetArray(da, gD_3, &aD_3);
    
    DMDAVecGetArray(da, lu, &ua); DMDAVecGetArray(da, lv, &va); DMDAVecGetArray(da, lp, &pa);
    DMDAVecGetArray(da, lU, &U); DMDAVecGetArray(da, lV, &V);
    DMDAVecGetArray(da, grhox, &rhox); DMDAVecGetArray(da, grhoy, &rhohy);
    DMDAVecGetArray(da, gJx, &Jx); DMDAVecGetArray(da, gJy, &Jy);
    DMDAVecGetArray(da, gg11, &g11); DMDAVecGetArray(da, gg22, &g22);

    // --- X-axis Dissipation ---
    for (j = ys; j < ye; j++) {
        for (i = xs; i < lxe; i++) {
            Uh = 0.5 * (U[j][i] + U[j][i + 1]);
            rhox[j][i] = (1 / Jx[j][i]) * (fabs(Uh) + sqrt((Uh * Uh) + g11[j][i]));
        }
    }

    for (j = lys; j < lye; j++) {
        for (i = lxs; i < llxe; i++) {
            aD1h_1[j][i] = rhox[j][i] * e * (pa[j][i + 2] - (3 * pa[j][i + 1]) + (3 * pa[j][i]) - (pa[j][i - 1]));
            aD1h_2[j][i] = rhox[j][i] * e * (ua[j][i + 2] - (3 * ua[j][i + 1]) + (3 * ua[j][i]) - (ua[j][i - 1]));
            aD1h_3[j][i] = rhox[j][i] * e * (va[j][i + 2] - (3 * va[j][i + 1]) + (3 * va[j][i]) - (va[j][i - 1]));
        }
    }
    
    for (j = lys; j < lye; j++) {
        for (i = xs; i < xe; i++) {
            if (i == 0) {
                aD1h_1[j][i] = aD1h_1[j][i + 1]; aD1h_2[j][i] = aD1h_2[j][i + 1]; aD1h_3[j][i] = aD1h_3[j][i + 1];
            }
            if (i == mx - 2) {
                aD1h_1[j][i] = aD1h_1[j][i - 1]; aD1h_2[j][i] = aD1h_2[j][i - 1]; aD1h_3[j][i] = aD1h_3[j][i - 1];
            }
        }
    }
    
    DMDAVecRestoreArray(da, gD1h_1, &aD1h_1); DMDAVecRestoreArray(da, gD1h_2, &aD1h_2); DMDAVecRestoreArray(da, gD1h_3, &aD1h_3);
    DMGlobalToLocalBegin(da, gD1h_1, INSERT_VALUES, lD1h_1); DMGlobalToLocalEnd(da, gD1h_1, INSERT_VALUES, lD1h_1);
    DMGlobalToLocalBegin(da, gD1h_2, INSERT_VALUES, lD1h_2); DMGlobalToLocalEnd(da, gD1h_2, INSERT_VALUES, lD1h_2);
    DMGlobalToLocalBegin(da, gD1h_3, INSERT_VALUES, lD1h_3); DMGlobalToLocalEnd(da, gD1h_3, INSERT_VALUES, lD1h_3);
    DMDAVecGetArray(da, lD1h_1, &aD1h_1); DMDAVecGetArray(da, lD1h_2, &aD1h_2); DMDAVecGetArray(da, lD1h_3, &aD1h_3);

    for (j = lys; j < lye; j++) {
        for (i = lxs; i < lxe; i++) {
            aD1_1[j][i] = (aD1h_1[j][i] - aD1h_1[j][i - 1]);
            aD1_2[j][i] = (aD1h_2[j][i] - aD1h_2[j][i - 1]);
            aD1_3[j][i] = (aD1h_3[j][i] - aD1h_3[j][i - 1]);
        }
    }
    
    DMDAVecRestoreArray(da, lD1h_1, &aD1h_1); DMDAVecRestoreArray(da, lD1h_2, &aD1h_2); DMDAVecRestoreArray(da, lD1h_3, &aD1h_3);
    DMDAVecRestoreArray(da, gD1_1, &aD1_1); DMDAVecRestoreArray(da, gD1_2, &aD1_2); DMDAVecRestoreArray(da, gD1_3, &aD1_3);
    DMLocalToGlobalBegin(da, lD1h_1, INSERT_VALUES, gD1h_1); DMLocalToGlobalEnd(da, lD1h_1, INSERT_VALUES, gD1h_1);
    DMLocalToGlobalBegin(da, lD1h_2, INSERT_VALUES, gD1h_2); DMLocalToGlobalEnd(da, lD1h_2, INSERT_VALUES, gD1h_2);
    DMLocalToGlobalBegin(da, lD1h_3, INSERT_VALUES, gD1h_3); DMLocalToGlobalEnd(da, lD1h_3, INSERT_VALUES, gD1h_3);

    // --- Y-axis Dissipation ---
    for (j = ys; j < lye; j++) {
        for (i = xs; i < xe; i++) {
            Vh = 0.5 * (V[j][i] + V[j + 1][i]);
            rhohy[j][i] = (1 / Jy[j][i]) * (fabs(Vh) + sqrt((Vh * Vh) + g22[j][i]));
        }
    }

    for (j = lys; j < llye; j++) {
        for (i = xs; i < xe; i++) {
            aD2h_1[j][i] = rhohy[j][i] * e * (pa[j + 2][i] - (3 * pa[j + 1][i]) + (3 * pa[j][i]) - (pa[j - 1][i]));
            aD2h_2[j][i] = rhohy[j][i] * e * (ua[j + 2][i] - (3 * ua[j + 1][i]) + (3 * ua[j][i]) - (ua[j - 1][i]));
            aD2h_3[j][i] = rhohy[j][i] * e * (va[j + 2][i] - (3 * va[j + 1][i]) + (3 * va[j][i]) - (va[j - 1][i]));
        }
    }
    
    for (j = ys; j < ye; j++) {
        for (i = lxs; i < lxe; i++) {
            if (j == 0) {
                aD2h_1[j][i] = aD2h_1[j + 1][i]; aD2h_2[j][i] = aD2h_2[j + 1][i]; aD2h_3[j][i] = aD2h_3[j + 1][i];
            }
            if (j == my - 2) {
                aD2h_1[j][i] = aD2h_1[j - 1][i]; aD2h_2[j][i] = aD2h_2[j - 1][i]; aD2h_3[j][i] = aD2h_3[j - 1][i];
            }
        }
    }
    
    DMDAVecRestoreArray(da, gD2h_1, &aD2h_1); DMDAVecRestoreArray(da, gD2h_2, &aD2h_2); DMDAVecRestoreArray(da, gD2h_3, &aD2h_3);
    DMGlobalToLocalBegin(da, gD2h_1, INSERT_VALUES, lD2h_1); DMGlobalToLocalEnd(da, gD2h_1, INSERT_VALUES, lD2h_1);
    DMGlobalToLocalBegin(da, gD2h_2, INSERT_VALUES, lD2h_2); DMGlobalToLocalEnd(da, gD2h_2, INSERT_VALUES, lD2h_2);
    DMGlobalToLocalBegin(da, gD2h_3, INSERT_VALUES, lD2h_3); DMGlobalToLocalEnd(da, gD2h_3, INSERT_VALUES, lD2h_3);
    DMDAVecGetArray(da, lD2h_1, &aD2h_1); DMDAVecGetArray(da, lD2h_2, &aD2h_2); DMDAVecGetArray(da, lD2h_3, &aD2h_3);

    for (j = lys; j < lye; j++) {
        for (i = xs; i < xe; i++) {
            aD2_1[j][i] = (aD2h_1[j][i] - aD2h_1[j - 1][i]);
            aD2_2[j][i] = (aD2h_2[j][i] - aD2h_2[j - 1][i]);
            aD2_3[j][i] = (aD2h_3[j][i] - aD2h_3[j - 1][i]);
        }
    }
    
    DMDAVecRestoreArray(da, lD2h_1, &aD2h_1); DMDAVecRestoreArray(da, lD2h_2, &aD2h_2); DMDAVecRestoreArray(da, lD2h_3, &aD2h_3);
    DMDAVecRestoreArray(da, gD2_1, &aD2_1); DMDAVecRestoreArray(da, gD2_2, &aD2_2); DMDAVecRestoreArray(da, gD2_3, &aD2_3);
    DMLocalToGlobalBegin(da, lD2h_1, INSERT_VALUES, gD2h_1); DMLocalToGlobalEnd(da, lD2h_1, INSERT_VALUES, gD2h_1);
    DMLocalToGlobalBegin(da, lD2h_2, INSERT_VALUES, gD2h_2); DMLocalToGlobalEnd(da, lD2h_2, INSERT_VALUES, gD2h_2);
    DMLocalToGlobalBegin(da, lD2h_3, INSERT_VALUES, gD2h_3); DMLocalToGlobalEnd(da, lD2h_3, INSERT_VALUES, gD2h_3);

    // --- Total Dissipation ---
    DMDAVecGetArray(da, gD1_1, &aD1_1); DMDAVecGetArray(da, gD1_2, &aD1_2); DMDAVecGetArray(da, gD1_3, &aD1_3);
    DMDAVecGetArray(da, gD2_1, &aD2_1); DMDAVecGetArray(da, gD2_2, &aD2_2); DMDAVecGetArray(da, gD2_3, &aD2_3);
    DMDAVecGetArray(da, gD_1, &aD_1); DMDAVecGetArray(da, gD_2, &aD_2); DMDAVecGetArray(da, gD_3, &aD_3);

    for (j = lys; j < lye; j++) {
        for (i = lxs; i < lxe; i++) {
            aD_1[j][i] = aD1_1[j][i] + aD2_1[j][i];
            aD_2[j][i] = aD1_2[j][i] + aD2_2[j][i];
            aD_3[j][i] = aD1_3[j][i] + aD2_3[j][i];
        }
    }
    
    DMDAVecRestoreArray(da, gD1_1, &aD1_1); DMDAVecRestoreArray(da, gD1_2, &aD1_2); DMDAVecRestoreArray(da, gD1_3, &aD1_3);
    DMDAVecRestoreArray(da, gD2_1, &aD2_1); DMDAVecRestoreArray(da, gD2_2, &aD2_2); DMDAVecRestoreArray(da, gD2_3, &aD2_3);
    DMDAVecRestoreArray(da, gD_1, &aD_1); DMDAVecRestoreArray(da, gD_2, &aD_2); DMDAVecRestoreArray(da, gD_3, &aD_3);

    DMDAVecRestoreArray(da, lu, &ua); DMDAVecRestoreArray(da, lv, &va); DMDAVecRestoreArray(da, lp, &pa);
    DMDAVecRestoreArray(da, lU, &U); DMDAVecRestoreArray(da, lV, &V);
    DMDAVecRestoreArray(da, grhox, &rhox); DMDAVecRestoreArray(da, lrhoy, &rhohy);
    DMDAVecRestoreArray(da, gJx, &Jx); DMDAVecRestoreArray(da, gJy, &Jy);
    DMDAVecRestoreArray(da, gg11, &g11); DMDAVecRestoreArray(da, gg22, &g22);

    DMLocalToGlobalBegin(da, lp, INSERT_VALUES, gp); DMLocalToGlobalEnd(da, lp, INSERT_VALUES, gp);
    DMLocalToGlobalBegin(da, lu, INSERT_VALUES, gu); DMLocalToGlobalEnd(da, lu, INSERT_VALUES, gu);
    DMLocalToGlobalBegin(da, lv, INSERT_VALUES, gv); DMLocalToGlobalEnd(da, lv, INSERT_VALUES, gv);
    
    return 0;
}

/**
 * @brief Calculates the viscous flux terms (Ev).
 */
PetscErrorCode GenerateEv() {
    xs = info.xs; xe = info.xs + info.xm; ys = info.ys; ye = info.ys + info.ym;
    mx = info.mx; my = info.my;
    lxs = (xs == 0) ? xs + 1 : xs; lys = (ys == 0) ? ys + 1 : ys;
    lxe = (xe == mx) ? xe - 1 : xe; lye = (ye == my) ? ye - 1 : ye;

    DMGlobalToLocalBegin(da, gp, INSERT_VALUES, lp); DMGlobalToLocalEnd(da, gp, INSERT_VALUES, lp);
    DMGlobalToLocalBegin(da, gu, INSERT_VALUES, lu); DMGlobalToLocalEnd(da, gu, INSERT_VALUES, lu);
    DMGlobalToLocalBegin(da, gv, INSERT_VALUES, lv); DMGlobalToLocalEnd(da, gv, INSERT_VALUES, lv);

    DMDAVecGetArray(da, gEv1_1, &aEv1_1); DMDAVecGetArray(da, gEv1_2, &aEv1_2); DMDAVecGetArray(da, gEv1_3, &aEv1_3);
    DMDAVecGetArray(da, gEv2_1, &aEv2_1); DMDAVecGetArray(da, gEv2_2, &aEv2_2); DMDAVecGetArray(da, gEv2_3, &aEv2_3);
    DMDAVecGetArray(da, gEv_1, &aEv_1); DMDAVecGetArray(da, gEv_2, &aEv_2); DMDAVecGetArray(da, gEv_3, &aEv_3);
    DMDAVecGetArray(da, gJx, &Jx); DMDAVecGetArray(da, gJy, &Jy);
    DMDAVecGetArray(da, gg11, &g11); DMDAVecGetArray(da, gg22, &g22);
    DMDAVecGetArray(da, gJ, &J);
    DMDAVecGetArray(da, lu, &ua); DMDAVecGetArray(da, lv, &va); DMDAVecGetArray(da, lp, &pa);

    for (j = ys; j < lye; j++) {
        for (i = lxs; i < lxe; i++) {
            aEv1_1[j][i] = 0.0;
            aEv1_2[j][i] = (1 / (Jx[j][i])) * (g11[j][i]) * (ua[j][i + 1] - ua[j][i]);
            aEv1_3[j][i] = (1 / (Jx[j][i])) * (g11[j][i]) * (va[j][i + 1] - va[j][i]);
        }
    }

    for (j = lys; j < lye; j++) {
        for (i = xs; i < lxe; i++) {
            aEv2_1[j][i] = 0.0;
            aEv2_2[j][i] = (1 / (Jy[j][i])) * (g22[j][i]) * (ua[j + 1][i] - ua[j][i]);
            aEv2_3[j][i] = (1 / (Jy[j][i])) * (g22[j][i]) * (va[j + 1][i] - va[j][i]);
        }
    }

    DMDAVecRestoreArray(da, gEv1_1, &aEv1_1); DMDAVecRestoreArray(da, gEv1_2, &aEv1_2); DMDAVecRestoreArray(da, gEv1_3, &aEv1_3);
    DMDAVecRestoreArray(da, gEv2_1, &aEv2_1); DMDAVecRestoreArray(da, gEv2_2, &aEv2_2); DMDAVecRestoreArray(da, gEv2_3, &aEv2_3);
    
    PetscBarrier(PETSC_NULL);

    DMGlobalToLocalBegin(da, gEv1_1, INSERT_VALUES, lEv1_1); DMGlobalToLocalEnd(da, gEv1_1, INSERT_VALUES, lEv1_1);
    DMGlobalToLocalBegin(da, gEv1_2, INSERT_VALUES, lEv1_2); DMGlobalToLocalEnd(da, gEv1_2, INSERT_VALUES, lEv1_2);
    DMGlobalToLocalBegin(da, gEv1_3, INSERT_VALUES, lEv1_3); DMGlobalToLocalEnd(da, gEv1_3, INSERT_VALUES, lEv1_3);
    DMGlobalToLocalBegin(da, gEv2_1, INSERT_VALUES, lEv2_1); DMGlobalToLocalEnd(da, gEv2_1, INSERT_VALUES, lEv2_1);
    DMGlobalToLocalBegin(da, gEv2_2, INSERT_VALUES, lEv2_2); DMGlobalToLocalEnd(da, gEv2_2, INSERT_VALUES, lEv2_2);
    DMGlobalToLocalBegin(da, gEv2_3, INSERT_VALUES, lEv2_3); DMGlobalToLocalEnd(da, gEv2_3, INSERT_VALUES, lEv2_3);

    PetscBarrier(PETSC_NULL);
    
    DMDAVecGetArray(da, lEv1_1, &aEv1_1); DMDAVecGetArray(da, lEv1_2, &aEv1_2); DMDAVecGetArray(da, lEv1_3, &aEv1_3);
    DMDAVecGetArray(da, lEv2_1, &aEv2_1); DMDAVecGetArray(da, lEv2_2, &aEv2_2); DMDAVecGetArray(da, lEv2_3, &aEv2_3);

    for (j = lys; j < lye; j++) {
        for (i = lxs; i < lxe; i++) {
            aEv_1[j][i] = (1 / Re) * ((aEv1_1[j][i] - aEv1_1[j][i - 1]) + (aEv2_1[j][i] - aEv2_1[j - 1][i])) * J[j][i];
            aEv_2[j][i] = (1 / Re) * ((aEv1_2[j][i] - aEv1_2[j][i - 1]) + (aEv2_2[j][i] - aEv2_2[j - 1][i])) * J[j][i];
            aEv_3[j][i] = (1 / Re) * ((aEv1_3[j][i] - aEv1_3[j][i - 1]) + (aEv2_3[j][i] - aEv2_3[j - 1][i])) * J[j][i];
        }
    }

    DMDAVecRestoreArray(da, lEv1_1, &aEv1_1); DMDAVecRestoreArray(da, lEv1_2, &aEv1_2); DMDAVecRestoreArray(da, lEv1_3, &aEv1_3);
    DMDAVecRestoreArray(da, lEv2_1, &aEv2_1); DMDAVecRestoreArray(da, lEv2_2, &aEv2_2); DMDAVecRestoreArray(da, lEv2_3, &aEv2_3);
    DMDAVecRestoreArray(da, gEv_1, &aEv_1); DMDAVecRestoreArray(da, gEv_2, &aEv_2); DMDAVecRestoreArray(da, gEv_3, &aEv_3);
    DMDAVecRestoreArray(da, gJx, &Jx); DMDAVecRestoreArray(da, gJy, &Jy);
    DMDAVecRestoreArray(da, gg11, &g11); DMDAVecRestoreArray(da, gg22, &g22);
    DMDAVecRestoreArray(da, gJ, &J);
    DMDAVecRestoreArray(da, lu, &ua); DMDAVecRestoreArray(da, lv, &va); DMDAVecRestoreArray(da, lp, &pa);
    
    PetscBarrier(PETSC_NULL);

    DMLocalToGlobalBegin(da, lEv1_1, INSERT_VALUES, gEv1_1); DMLocalToGlobalEnd(da, lEv1_1, INSERT_VALUES, gEv1_1);
    DMLocalToGlobalBegin(da, lEv1_2, INSERT_VALUES, gEv1_2); DMLocalToGlobalEnd(da, lEv1_2, INSERT_VALUES, gEv1_2);
    DMLocalToGlobalBegin(da, lEv1_3, INSERT_VALUES, gEv1_3); DMLocalToGlobalEnd(da, lEv1_3, INSERT_VALUES, gEv1_3);
    DMLocalToGlobalBegin(da, lEv2_1, INSERT_VALUES, gEv2_1); DMLocalToGlobalEnd(da, lEv2_1, INSERT_VALUES, gEv2_1);
    DMLocalToGlobalBegin(da, lEv2_2, INSERT_VALUES, gEv2_2); DMLocalToGlobalEnd(da, lEv2_2, INSERT_VALUES, gEv2_2);
    DMLocalToGlobalBegin(da, lEv2_3, INSERT_VALUES, gEv2_3); DMLocalToGlobalEnd(da, lEv2_3, INSERT_VALUES, gEv2_3);
    DMLocalToGlobalBegin(da, lp, INSERT_VALUES, gp); DMLocalToGlobalEnd(da, lp, INSERT_VALUES, gp);
    DMLocalToGlobalBegin(da, lu, INSERT_VALUES, gu); DMLocalToGlobalEnd(da, lu, INSERT_VALUES, gu);
    DMLocalToGlobalBegin(da, lv, INSERT_VALUES, gv); DMLocalToGlobalEnd(da, lv, INSERT_VALUES, gv);

    return 0;
}


/**
 * @brief Calculates the convective flux terms (E).
 */
PetscErrorCode GenerateE() {
    xs = info.xs; xe = info.xs + info.xm; ys = info.ys; ye = info.ys + info.ym;
    mx = info.mx; my = info.my;
    lxs = (xs == 0) ? xs + 1 : xs; lys = (ys == 0) ? ys + 1 : ys;
    lxe = (xe == mx) ? xe - 1 : xe; lye = (ye == my) ? ye - 1 : ye;

    DMGlobalToLocalBegin(da, gp, INSERT_VALUES, lp); DMGlobalToLocalEnd(da, gp, INSERT_VALUES, lp);
    DMGlobalToLocalBegin(da, gu, INSERT_VALUES, lu); DMGlobalToLocalEnd(da, gu, INSERT_VALUES, lu);
    DMGlobalToLocalBegin(da, gv, INSERT_VALUES, lv); DMGlobalToLocalEnd(da, gv, INSERT_VALUES, lv);

    DMDAVecGetArray(da, lu, &ua); DMDAVecGetArray(da, lv, &va); DMDAVecGetArray(da, lp, &pa);
    DMDAVecGetArray(da, gE1_1, &aE1_1); DMDAVecGetArray(da, gE1_2, &aE1_2); DMDAVecGetArray(da, gE1_3, &aE1_3);
    DMDAVecGetArray(da, gE2_1, &aE2_1); DMDAVecGetArray(da, gE2_2, &aE2_2); DMDAVecGetArray(da, gE2_3, &aE2_3);
    DMDAVecGetArray(da, gE_1, &aE_1); DMDAVecGetArray(da, gE_2, &aE_2); DMDAVecGetArray(da, gE_3, &aE_3);
    DMDAVecGetArray(da, gJ, &J); DMDAVecGetArray(da, gU, &U); DMDAVecGetArray(da, gV, &V);

    for (j = ys; j < ye; j++) {
        for (i = xs; i < xe; i++) {
            U[j][i] = ua[j][i] * zetax[i];
            V[j][i] = va[j][i] * etay[j];
            
            aE1_1[j][i] = U[j][i] / J[j][i];
            aE2_1[j][i] = V[j][i] / J[j][i];
            
            aE1_2[j][i] = (1 / J[j][i]) * ((ua[j][i] * U[j][i]) + (pa[j][i] * zetax[i]));
            aE2_2[j][i] = (1 / J[j][i]) * ((ua[j][i] * V[j][i]));
            
            aE1_3[j][i] = (1 / J[j][i]) * ((va[j][i] * U[j][i]));
            aE2_3[j][i] = (1 / J[j][i]) * ((va[j][i] * V[j][i]) + (pa[j][i] * etay[j]));
        }
    }
    
    DMDAVecRestoreArray(da, gU, &U); DMDAVecRestoreArray(da, gV, &V);
    DMDAVecRestoreArray(da, gE1_1, &aE1_1); DMDAVecRestoreArray(da, gE1_2, &aE1_2); DMDAVecRestoreArray(da, gE1_3, &aE1_3);
    DMDAVecRestoreArray(da, gE2_1, &aE2_1); DMDAVecRestoreArray(da, gE2_2, &aE2_2); DMDAVecRestoreArray(da, gE2_3, &aE2_3);

    PetscBarrier(PETSC_NULL);

    DMGlobalToLocalBegin(da, gE1_1, INSERT_VALUES, lE1_1); DMGlobalToLocalEnd(da, gE1_1, INSERT_VALUES, lE1_1);
    DMGlobalToLocalBegin(da, gE1_2, INSERT_VALUES, lE1_2); DMGlobalToLocalEnd(da, gE1_2, INSERT_VALUES, lE1_2);
    DMGlobalToLocalBegin(da, gE1_3, INSERT_VALUES, lE1_3); DMGlobalToLocalEnd(da, gE1_3, INSERT_VALUES, lE1_3);
    DMGlobalToLocalBegin(da, gE2_1, INSERT_VALUES, lE2_1); DMGlobalToLocalEnd(da, gE2_1, INSERT_VALUES, lE2_1);
    DMGlobalToLocalBegin(da, gE2_2, INSERT_VALUES, lE2_2); DMGlobalToLocalEnd(da, gE2_2, INSERT_VALUES, lE2_2);
    DMGlobalToLocalBegin(da, gE2_3, INSERT_VALUES, lE2_3); DMGlobalToLocalEnd(da, gE2_3, INSERT_VALUES, lE2_3);
    
    DMDAVecGetArray(da, lE1_1, &aE1_1); DMDAVecGetArray(da, lE1_2, &aE1_2); DMDAVecGetArray(da, lE1_3, &aE1_3);
    DMDAVecGetArray(da, lE2_1, &aE2_1); DMDAVecGetArray(da, lE2_2, &aE2_2); DMDAVecGetArray(da, lE2_3, &aE2_3);

    for (j = lys; j < lye; j++) {
        for (i = lxs; i < lxe; i++) {
            aE_1[j][i] = 0.5 * ((aE1_1[j][i + 1] - aE1_1[j][i - 1]) + (aE2_1[j + 1][i] - aE2_1[j - 1][i])) * J[j][i];
            aE_2[j][i] = 0.5 * ((aE1_2[j][i + 1] - aE1_2[j][i - 1]) + (aE2_2[j + 1][i] - aE2_2[j - 1][i])) * J[j][i];
            aE_3[j][i] = 0.5 * ((aE1_3[j][i + 1] - aE1_3[j][i - 1]) + (aE2_3[j + 1][i] - aE2_3[j - 1][i])) * J[j][i];
        }
    }
    
    DMDAVecRestoreArray(da, lE1_1, &aE1_1); DMDAVecRestoreArray(da, lE1_2, &aE1_2); DMDAVecRestoreArray(da, lE1_3, &aE1_3);
    DMDAVecRestoreArray(da, lE2_1, &aE2_1); DMDAVecRestoreArray(da, lE2_2, &aE2_2); DMDAVecRestoreArray(da, lE2_3, &aE2_3);
    DMDAVecRestoreArray(da, gE_1, &aE_1); DMDAVecRestoreArray(da, gE_2, &aE_2); DMDAVecRestoreArray(da, gE_3, &aE_3);
    DMDAVecRestoreArray(da, gJ, &J);
    DMDAVecRestoreArray(da, lu, &ua); DMDAVecRestoreArray(da, lv, &va); DMDAVecRestoreArray(da, lp, &pa);
    
    DMLocalToGlobalBegin(da, lE1_1, INSERT_VALUES, gE1_1); DMLocalToGlobalEnd(da, lE1_1, INSERT_VALUES, gE1_1);
    DMLocalToGlobalBegin(da, lE1_2, INSERT_VALUES, gE1_2); DMLocalToGlobalEnd(da, lE1_2, INSERT_VALUES, gE1_2);
    DMLocalToGlobalBegin(da, lE1_3, INSERT_VALUES, gE1_3); DMLocalToGlobalEnd(da, lE1_3, INSERT_VALUES, gE1_3);
    DMLocalToGlobalBegin(da, lE2_1, INSERT_VALUES, gE2_1); DMLocalToGlobalEnd(da, lE2_1, INSERT_VALUES, gE2_1);
    DMLocalToGlobalBegin(da, lE2_2, INSERT_VALUES, gE2_2); DMLocalToGlobalEnd(da, lE2_2, INSERT_VALUES, gE2_2);
    DMLocalToGlobalBegin(da, lE2_3, INSERT_VALUES, gE2_3); DMLocalToGlobalEnd(da, lE2_3, INSERT_VALUES, gE2_3);
    DMLocalToGlobalBegin(da, lp, INSERT_VALUES, gp); DMLocalToGlobalEnd(da, lp, INSERT_VALUES, gp);
    DMLocalToGlobalBegin(da, lu, INSERT_VALUES, gu); DMLocalToGlobalEnd(da, lu, INSERT_VALUES, gu);
    DMLocalToGlobalBegin(da, lv, INSERT_VALUES, gv); DMLocalToGlobalEnd(da, lv, INSERT_VALUES, gv);
    
    return 0;
}


/**
 * @brief Sets the boundary conditions for the flow variables.
 */
PetscErrorCode SetBCS() {
    xs = info.xs; xe = info.xs + info.xm; ys = info.ys; ye = info.ys + info.ym;
    mx = info.mx; my = info.my;
    
    DMGlobalToLocalBegin(da, gv, INSERT_VALUES, lv); DMGlobalToLocalEnd(da, gv, INSERT_VALUES, lv);
    DMGlobalToLocalBegin(da, gu, INSERT_VALUES, lu); DMGlobalToLocalEnd(da, gu, INSERT_VALUES, lu);
    DMGlobalToLocalBegin(da, gp, INSERT_VALUES, lp); DMGlobalToLocalEnd(da, gp, INSERT_VALUES, lp);

    DMDAVecGetArray(da, lu, &ua); DMDAVecGetArray(da, lv, &va); DMDAVecGetArray(da, lp, &pa);
    
    for (j = ys; j < ye; j++) {
        if (y[j] >= ex) inflowy = j;
        break;
    }

    for (j = ys; j < ye; j++) {
        for (i = xs; i < xe; i++) {
            if (i == 0) { // Inlet
                if (j > inflowy) {
                    pa[j][i] = pa[j][i + 1];
                    ua[j][i] = 1.5 * U_inf * ((24 * y[j]) - (16 * y[j] * y[j]) - 8);
                    va[j][i] = 0.0;
                } else if (j <= inflowy) { // Step
                    pa[j][i] = pa[j][i + 1];
                    ua[j][i] = 0.0;
                    va[j][i] = 0.0;
                }
            }
            if (j == 0) { // Bottom Wall
                pa[j][i] = pa[j + 1][i];
                ua[j][i] = 0.0;
                va[j][i] = 0.0;
            }
            if (j == my - 1) { // Top Wall
                pa[j][i] = pa[j - 1][i];
                ua[j][i] = 0.0;
                va[j][i] = 0.0;
            }
            if (i == mx - 1 && j > 0 && j < my - 1) { // Outlet
                pa[j][i] = pa[j][i - 1];
                ua[j][i] = ua[j][i - 1];
                va[j][i] = va[j][i - 1];
            }
        }
    }
    
    DMDAVecRestoreArray(da, lu, &ua); DMDAVecRestoreArray(da, lp, &pa); DMDAVecRestoreArray(da, lv, &va);
    
    DMLocalToGlobalBegin(da, lu, INSERT_VALUES, gu); DMLocalToGlobalEnd(da, lu, INSERT_VALUES, gu);
    DMLocalToGlobalBegin(da, lv, INSERT_VALUES, gv); DMLocalToGlobalEnd(da, lv, INSERT_VALUES, gv);
    DMLocalToGlobalBegin(da, lp, INSERT_VALUES, gp); DMLocalToGlobalEnd(da, lp, INSERT_VALUES, gp);
    
    return 0;
}

/**
 * @brief Calculates the time step size based on the CFL condition.
 */
PetscErrorCode PrepareTimeloop() {
    if (!rank) {
        PetscReal min = dx[0];
        for (i = 0; i < Nx; i++) {
            if (dx[i] < min) { min = dx[i]; }
        }
        for (j = 0; j < Ny; j++) {
            if (dy[j] < min) { min = dy[j]; }
        }
        dt = cfl * min;
    }
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "Time stepping prepared. dt = %lf\n", dt);
    return 0;
}

/**
 * @brief Calculates the transformation metrics for the curvilinear grid.
 */
PetscErrorCode GenerateMetrics() {
    GenerateTransforms();

    xs = info.xs; xe = info.xs + info.xm; ys = info.ys; ye = info.ys + info.ym;
    mx = info.mx; my = info.my;
    lxs = (xs == 0) ? xs + 1 : xs; lys = (ys == 0) ? ys + 1 : ys;
    lxe = (xe == mx) ? xe - 1 : xe; lye = (ye == my) ? ye - 1 : ye;

    DMDAVecGetArray(da, gJ, &J);
    for (j = ys; j < ye; j++) {
        for (i = xs; i < xe; i++) {
            J[j][i] = 1 / (xzeta[i] * yeta[j]);
        }
    }
    DMDAVecRestoreArray(da, gJ, &J);
    
    DMGlobalToLocalBegin(da, gJ, INSERT_VALUES, lJ); DMGlobalToLocalEnd(da, gJ, INSERT_VALUES, lJ);
    
    DMDAVecGetArray(da, lJ, &J);
    DMDAVecGetArray(da, gJx, &Jx); DMDAVecGetArray(da, gJy, &Jy);
    DMDAVecGetArray(da, gg11, &g11); DMDAVecGetArray(da, gg22, &g22);

    for (j = ys; j < ye; j++) {
        for (i = xs; i < lxe; i++) {
            Jx[j][i] = 0.5 * (J[j][i] + J[j][i + 1]);
            g11[j][i] = (0.5 * (zetax[i] + zetax[i + 1])) * (0.5 * (zetax[i] + zetax[i + 1]));
        }
    }
    for (j = lys; j < lye; j++) {
        for (i = xs; i < xe; i++) {
            Jy[j][i] = 0.5 * (J[j][i] + J[j + 1][i]);
            g22[j][i] = (0.5 * (etay[j] + etay[j + 1])) * (0.5 * (etay[j] + etay[j + 1]));
        }
    }
    
    DMDAVecRestoreArray(da, lJ, &J);
    DMDAVecRestoreArray(da, gJx, &Jx); DMDAVecRestoreArray(da, gJy, &Jy);
    DMDAVecRestoreArray(da, gg11, &g11); DMDAVecRestoreArray(da, gg22, &g22);
    
    DMLocalToGlobalBegin(da, lJ, INSERT_VALUES, gJ); DMLocalToGlobalEnd(da, lJ, INSERT_VALUES, gJ);

    return 0;
}

/**
 * @brief Calculates the derivatives of the coordinate transformation.
 */
PetscErrorCode GenerateTransforms() {
    if (!rank) {
        for (i = 1; i < IM - 1; i++) {
            xzeta[i] = 0.5 * (x[i + 1] - x[i - 1]);
            zetax[i] = 1 / xzeta[i];
        }
        xzeta[0] = 0.5 * (x[1] - x[0] + dx[0] / rx);
        zetax[0] = 1 / xzeta[0];
        xzeta[IM - 1] = 0.5 * (x[IM - 1] - x[IM - 2] + dx[IM - 2] / rx);
        zetax[IM - 1] = 1 / xzeta[IM - 1];

        for (j = 1; j < JM - 1; j++) {
            yeta[j] = 0.5 * (y[j + 1] - y[j - 1]);
            etay[j] = 1 / yeta[j];
        }
        yeta[0] = 0.5 * (y[1] - y[0] + dy[0] / ry);
        etay[0] = 1 / yeta[0];
        yeta[JM - 1] = 0.5 * (y[JM - 1] - y[JM - 2] + dy[JM - 2] / ry);
        etay[JM - 1] = 1 / yeta[JM - 1];
    }
    MPI_Bcast(&xzeta, IM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&etay, JM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&zetax, IM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&yeta, JM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    return 0;
}

/**
 * @brief Generates the physical grid coordinates (x, y).
 */
PetscErrorCode GenerateGrid() {
    FILE *xfp, *yfp;
    char xgrid[] = "data/xgrid.txt";
    char ygrid[] = "data/ygrid.txt";
    
    if (!rank) {
        Midx = (IM - 1) / 2;
        dx[0] = (rx == 1.0) ? (ex * dr) / (double)Midx : ex * dr * ((1 - rx) / (1 - (pow(rx, Midx))));
        for (i = 1; i < Midx; i++) {
            dx[i] = rx * dx[i - 1];
        }
        for (i = Midx; i < Nx; i++) {
            dx[i] = dx[i - 1] / rx;
        }
        x[0] = 0.0;
        for (i = 1; i < IM; i++) {
            x[i] = x[i - 1] + dx[i - 1];
        }
        
        Midy = (JM - 1) / 2;
        dy[0] = (ry == 1.0) ? (ex * 2) / (double)Ny : ex * ((1 - ry) / (1 - (pow(ry, (double)Midy))));
        for (j = 1; j < Midy; j++) {
            dy[j] = ry * dy[j - 1];
        }
        for (j = Midy; j < Ny; j++) {
            dy[j] = dy[j - 1] / ry;
        }
        y[0] = 0.0;
        for (j = 1; j < JM; j++) {
            y[j] = y[j - 1] + dy[j - 1];
        }

        PetscFOpen(PETSC_COMM_WORLD, xgrid, "w", &xfp);
        for (i = 0; i < IM; i++) {
            PetscFPrintf(PETSC_COMM_WORLD, xfp, (i == IM - 1) ? "%lf" : "%lf,", x[i]);
        }
        PetscFClose(PETSC_COMM_WORLD, xfp);
        
        PetscFOpen(PETSC_COMM_WORLD, ygrid, "w", &yfp);
        for (j = 0; j < JM; j++) {
            PetscFPrintf(PETSC_COMM_WORLD, yfp, (j == JM - 1) ? "%lf" : "%lf,", y[j]);
        }
        PetscFClose(PETSC_COMM_WORLD, yfp);
    }
    MPI_Bcast(&x, IM, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&y, JM, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    
    return 0;
}

/**
 * @brief Initializes the velocity and pressure fields.
 */
PetscErrorCode Initialize() {
    xs = info.xs; xe = info.xs + info.xm; ys = info.ys; ye = info.ys + info.ym;
    
    DMGlobalToLocalBegin(da, gu, INSERT_VALUES, lu); DMGlobalToLocalEnd(da, gu, INSERT_VALUES, lu);
    DMDAVecGetArray(da, lu, &ua);

    for (j = ys; j < ye; j++) {
        if (y[j] >= ex) inflowy = j;
        break;
    }

    for (j = ys; j < ye; j++) {
        for (i = xs; i < xe; i++) {
            if (i == 0 && j > inflowy) {
                ua[j][i] = 1.5 * U_inf * ((24 * y[j]) - (16 * y[j] * y[j]) - 8);
            } else {
                ua[j][i] = 0.0;
            }
        }
    }
    
    DMDAVecRestoreArray(da, lu, &ua);
    DMLocalToGlobalBegin(da, lu, INSERT_VALUES, gu); DMLocalToGlobalEnd(da, lu, INSERT_VALUES, gu);
    
    return 0;
}

/**
 * @brief Normalizes the pressure field by subtracting a reference value.
 */
PetscErrorCode NormalizePressure() {
    PetscReal ref;
    PetscInt ref_rank = 0;

    DMDAVecGetArray(da, gp, &pa);
    if (info.xs == 0 && info.ys == 0) {
        ref = pa[0][0];
        ref_rank = rank;
    }
    MPI_Bcast(&ref_rank, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&ref, 1, MPI_DOUBLE, ref_rank, PETSC_COMM_WORLD);

    for (j = info.ys; j < info.ys + info.ym; j++) {
        for (i = info.xs; i < info.xs + info.xm; i++) {
            pa[j][i] = pa[j][i] - ref;
        }
    }
    DMDAVecRestoreArray(da, gp, &pa);
    
    return 0;
}

/**
 * @brief Checks for convergence by calculating the L2 norm of the residual.
 */
PetscErrorCode CheckConvergence() {
    VecWAXPY(e1, -1.0, gpo, gp);
    VecWAXPY(e2, -1.0, guo, gu);
    VecWAXPY(e3, -1.0, gvo, gv);
    
    VecPointwiseMult(ebuff, e1, e1); VecCopy(ebuff, e1);
    VecPointwiseMult(ebuff, e2, e2); VecCopy(ebuff, e2);
    VecPointwiseMult(ebuff, e3, e3); VecCopy(ebuff, e3);
    
    VecWAXPY(ebuff, 1.0, e1, e2);
    VecWAXPY(e1, 1.0, e3, ebuff);
    
    VecSum(e1, &err);
    err = (sqrt(err) / (IM * JM));
    
    if (err <= tol) {
        conv = PETSC_TRUE;
        PetscPrintf(PETSC_COMM_WORLD, "Convergence achieved.\n");
    } else {
        conv = PETSC_FALSE;
    }
    return 0;
}

/**
 * @brief Writes the final solution fields to data files.
 */
PetscErrorCode GenerateDatFile() {
    FILE *ufp, *vfp, *pfp;
    const char usol[] = "data/usol.txt";
    const char vsol[] = "data/vsol.txt";
    const char psol[] = "data/psol.txt";
    
    PetscFOpen(PETSC_COMM_WORLD, usol, "w", &ufp);
    PetscFOpen(PETSC_COMM_WORLD, psol, "w", &pfp);
    PetscFOpen(PETSC_COMM_WORLD, vsol, "w", &vfp);

    VecScatterCreateToZero(gu, &uoutput, &uout);
    VecScatterBegin(uoutput, gu, uout, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(uoutput, gu, uout, INSERT_VALUES, SCATTER_FORWARD);
    
    VecScatterCreateToZero(gv, &voutput, &vout);
    VecScatterBegin(voutput, gv, vout, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(voutput, gv, vout, INSERT_VALUES, SCATTER_FORWARD);
    
    VecScatterCreateToZero(gp, &poutput, &pout);
    VecScatterBegin(poutput, gp, pout, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(poutput, gp, pout, INSERT_VALUES, SCATTER_FORWARD);

    VecGetArrayRead(uout, &ru);
    VecGetArrayRead(vout, &rv);
    VecGetArrayRead(pout, &rp);

    if (!rank) {
        for (l = 0; l < IM * JM; l++) {
            j = (PetscInt)l / IM;
            i = (PetscInt)l % IM;
            
            PetscFPrintf(PETSC_COMM_WORLD, ufp, (i == IM - 1) ? "%lf\n" : "%lf,", ru[l]);
            PetscFPrintf(PETSC_COMM_WORLD, pfp, (i == IM - 1) ? "%lf\n" : "%lf,", rp[l]);
            PetscFPrintf(PETSC_COMM_WORLD, vfp, (i == IM - 1) ? "%lf\n" : "%lf,", rv[l]);
        }
    }
    
    VecRestoreArrayRead(uout, &ru);
    VecRestoreArrayRead(pout, &rp);
    VecRestoreArrayRead(vout, &rv);

    PetscFClose(PETSC_COMM_WORLD, ufp);
    PetscFClose(PETSC_COMM_WORLD, pfp);
    PetscFClose(PETSC_COMM_WORLD, vfp);
    
    VecScatterDestroy(&uoutput);
    VecScatterDestroy(&voutput);
    VecScatterDestroy(&poutput);
    
    return 0;
}
