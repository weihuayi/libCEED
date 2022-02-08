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
/// Utility functions for setting up Darcy problem in 3D

#include "../include/setup-libceed.h"
#include "../include/problems.h"
#include "../qfunctions/darcy-rhs3d.h"
#include "../qfunctions/darcy-mass3d.h"
#include "../qfunctions/darcy-error3d.h"
#include "../qfunctions/face-geo3d.h"

// Hdiv_DARCY3D is registered in cl-option.c
PetscErrorCode Hdiv_DARCY3D(ProblemData *problem_data, void *ctx) {
  User              user = *(User *)ctx;
  MPI_Comm          comm = PETSC_COMM_WORLD;
  PetscInt          ierr;
  PetscFunctionBeginUser;

  ierr = PetscCalloc1(1, &user->phys->darcy3d_ctx); CHKERRQ(ierr);

  // ------------------------------------------------------
  //               SET UP POISSON_QUAD2D
  // ------------------------------------------------------
  problem_data->dim                     = 3;
  problem_data->elem_node               = 8;
  problem_data->q_data_size_face        = 4;
  problem_data->quadrature_mode         = CEED_GAUSS;
  problem_data->setup_rhs               = SetupDarcyRhs3D;
  problem_data->setup_rhs_loc           = SetupDarcyRhs3D_loc;
  problem_data->residual                = SetupDarcyMass3D;
  problem_data->residual_loc            = SetupDarcyMass3D_loc;
  problem_data->setup_error             = SetupDarcyError3D;
  problem_data->setup_error_loc         = SetupDarcyError3D_loc;
  problem_data->setup_face_geo          = SetupFaceGeo3D;
  problem_data->setup_face_geo_loc      = SetupFaceGeo3D_loc;
  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  PetscOptionsBegin(comm, NULL, "Options for Hdiv-mixed problem", NULL);

  PetscOptionsEnd();

  PetscFunctionReturn(0);
}
