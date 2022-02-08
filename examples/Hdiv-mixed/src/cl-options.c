// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

/// @file
/// Command line option processing for H(div) example using PETSc

#include "../include/cl-options.h"
#include "../include/problems.h"

// Register problems to be available on the command line
PetscErrorCode RegisterProblems_Hdiv(AppCtx app_ctx) {
  app_ctx->problems = NULL;
  PetscErrorCode   ierr;
  PetscFunctionBeginUser;
  // 1) darcy2d (Hdiv_DARCY2D is created in darcy2d.c)
  ierr = PetscFunctionListAdd(&app_ctx->problems, "darcy2d",
                              Hdiv_DARCY2D); CHKERRQ(ierr);
  // 2) darcy3d (Hdiv_DARCY3D is created in dacry3d.c)
  ierr = PetscFunctionListAdd(&app_ctx->problems, "darcy3d",
                              Hdiv_DARCY3D); CHKERRQ(ierr);
  // 3) darcy3d-prism

  // 4) richard

  PetscFunctionReturn(0);
}

// Process general command line options
PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx app_ctx) {

  PetscBool problem_flag = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  PetscOptionsBegin(comm, NULL, "H(div) examples in PETSc with libCEED", NULL);

  ierr = PetscOptionsFList("-problem", "Problem to solve", NULL,
                           app_ctx->problems,
                           app_ctx->problem_name, app_ctx->problem_name, sizeof(app_ctx->problem_name),
                           &problem_flag); CHKERRQ(ierr);

  app_ctx->degree = 1;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of finite elements",
                         NULL, app_ctx->degree, &app_ctx->degree, NULL); CHKERRQ(ierr);

  app_ctx->q_extra = 0;
  ierr = PetscOptionsInt("-q_extra", "Number of extra quadrature points",
                         NULL, app_ctx->q_extra, &app_ctx->q_extra, NULL); CHKERRQ(ierr);

  // Neumann boundary conditions
  app_ctx->bc_traction_count = 16;
  ierr = PetscOptionsIntArray("-bc_traction",
                              "Face IDs to apply traction (Neumann) BC",
                              NULL, app_ctx->bc_traction_faces,
                              &app_ctx->bc_traction_count, NULL); CHKERRQ(ierr);
  // Set vector for each traction BC
  for (PetscInt i = 0; i < app_ctx->bc_traction_count; i++) {
    // Traction vector
    char option_name[25];
    for (PetscInt j = 0; j < 3; j++)
      app_ctx->bc_traction_vector[i][j] = 0.;

    snprintf(option_name, sizeof option_name, "-bc_traction_%d",
             app_ctx->bc_traction_faces[i]);
    PetscInt max_n = 3;
    PetscBool set = false;
    ierr = PetscOptionsScalarArray(option_name,
                                   "Traction vector for constrained face", NULL,
                                   app_ctx->bc_traction_vector[i], &max_n, &set);
    CHKERRQ(ierr);

    if (!set)
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,
              "Traction vector must be set for all traction boundary conditions.");
  }
  // Provide default problem if not specified
  if (!problem_flag) {
    const char *problem_name = "darcy2d";
    strncpy(app_ctx->problem_name, problem_name, 16);
  }

  PetscOptionsEnd();

  PetscFunctionReturn(0);
}
