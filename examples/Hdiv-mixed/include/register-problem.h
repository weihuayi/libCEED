#ifndef register_problems_h
#define register_problems_h

#include "structs.h"

// Register problems to be available on the command line
PetscErrorCode RegisterProblems_Hdiv(AppCtx app_ctx);
// -----------------------------------------------------------------------------
// Set up problems function prototype
// -----------------------------------------------------------------------------
// 1) darcy2d
PetscErrorCode Hdiv_DARCY2D(ProblemData problem_data, void *ctx);

// 2) darcy3d
PetscErrorCode Hdiv_DARCY3D(ProblemData problem_data, void *ctx);

// 3) darcy3dprism

// 4) richard

#endif // register_problems_h
