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
/// Utility functions for setting up Darcy problem in 2D

#include "../include/setup-libceed.h"
#include "../include/problems.h"
#include "../qfunctions/darcy-rhs2d.h"
#include "../qfunctions/darcy-mass2d.h"
#include "../qfunctions/darcy-error2d.h"
#include "../qfunctions/face-geo2d.h"

// Hdiv_DARCY2D is registered in cl-option.c
PetscErrorCode Hdiv_DARCY2D(ProblemData *problem_data, void *ctx) {
  User              user = *(User *)ctx;
  MPI_Comm          comm = PETSC_COMM_WORLD;
  PetscInt          ierr;
  PetscFunctionBeginUser;

  ierr = PetscCalloc1(1, &user->phys->darcy2d_ctx); CHKERRQ(ierr);

  // ------------------------------------------------------
  //               SET UP POISSON_QUAD2D
  // ------------------------------------------------------
  problem_data->dim                     = 2;
  problem_data->elem_node               = 4;
  problem_data->q_data_size_face        = 3;
  problem_data->quadrature_mode         = CEED_GAUSS;
  problem_data->setup_rhs               = SetupDarcyRhs2D;
  problem_data->setup_rhs_loc           = SetupDarcyRhs2D_loc;
  problem_data->residual                = SetupDarcyMass2D;
  problem_data->residual_loc            = SetupDarcyMass2D_loc;
  problem_data->setup_error             = SetupDarcyError2D;
  problem_data->setup_error_loc         = SetupDarcyError2D_loc;
  problem_data->setup_face_geo          = SetupFaceGeo2D;
  problem_data->setup_face_geo_loc      = SetupFaceGeo2D_loc;

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  PetscOptionsBegin(comm, NULL, "Options for Hdiv-mixed problem", NULL);

  PetscOptionsEnd();

  PetscFunctionReturn(0);
}
