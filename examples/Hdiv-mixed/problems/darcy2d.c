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

#include "../include/register-problem.h"
#include "../qfunctions/darcy-force2d.h"
#include "../qfunctions/darcy-mass2d.h"
#include "../qfunctions/darcy-error2d.h"
#include "../qfunctions/pressure-boundary2d.h"

// Hdiv_DARCY2D is registered in cl-option.c
PetscErrorCode Hdiv_DARCY2D(ProblemData problem_data, void *ctx) {
  Physics           phys = *(Physics *)ctx;
  MPI_Comm          comm = PETSC_COMM_WORLD;
  PetscFunctionBeginUser;

  PetscCall( PetscCalloc1(1, &phys->darcy2d_ctx) );

  // ------------------------------------------------------
  //               SET UP POISSON_QUAD2D
  // ------------------------------------------------------
  problem_data->dim                     = 2;
  problem_data->elem_node               = 4;
  problem_data->q_data_size_face        = 3;
  problem_data->quadrature_mode         = CEED_GAUSS;
  problem_data->force                   = DarcyForce2D;
  problem_data->force_loc               = DarcyForce2D_loc;
  problem_data->residual                = DarcyMass2D;
  problem_data->residual_loc            = DarcyMass2D_loc;
  problem_data->jacobian                = JacobianDarcyMass2D;
  problem_data->jacobian_loc            = JacobianDarcyMass2D_loc;
  problem_data->error                   = DarcyError2D;
  problem_data->error_loc               = DarcyError2D_loc;
  problem_data->bc_pressure             = BCPressure2D;
  problem_data->bc_pressure_loc         = BCPressure2D_loc;

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  PetscOptionsBegin(comm, NULL, "Options for Hdiv-mixed problem", NULL);

  PetscOptionsEnd();

  PetscCall( PetscFree(phys->darcy2d_ctx) );
  PetscFunctionReturn(0);
}
