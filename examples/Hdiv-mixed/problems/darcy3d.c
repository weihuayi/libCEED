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

#include "../include/register-problem.h"
#include "../qfunctions/darcy-force3d.h"
#include "../qfunctions/darcy-mass3d.h"
#include "../qfunctions/darcy-error3d.h"
#include "../qfunctions/pressure-boundary3d.h"

// Hdiv_DARCY3D is registered in cl-option.c
PetscErrorCode Hdiv_DARCY3D(ProblemData problem_data, void *ctx) {
  Physics           phys = *(Physics *)ctx;
  MPI_Comm          comm = PETSC_COMM_WORLD;
  PetscFunctionBeginUser;

  PetscCall( PetscCalloc1(1, &phys->darcy3d_ctx) );

  // ------------------------------------------------------
  //               SET UP POISSON_QUAD2D
  // ------------------------------------------------------
  problem_data->dim                     = 3;
  problem_data->elem_node               = 8;
  problem_data->q_data_size_face        = 4;
  problem_data->quadrature_mode         = CEED_GAUSS;
  problem_data->force                   = DarcyForce3D;
  problem_data->force_loc               = DarcyForce3D_loc;
  problem_data->residual                = DarcyMass3D;
  problem_data->residual_loc            = DarcyMass3D_loc;
  problem_data->jacobian                = JacobianDarcyMass3D;
  problem_data->jacobian_loc            = JacobianDarcyMass3D_loc;
  problem_data->error                   = DarcyError3D;
  problem_data->error_loc               = DarcyError3D_loc;
  problem_data->bc_pressure             = BCPressure3D;
  problem_data->bc_pressure_loc         = BCPressure3D_loc;
  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  PetscOptionsBegin(comm, NULL, "Options for Hdiv-mixed problem", NULL);

  PetscOptionsEnd();
  PetscCall( PetscFree(phys->darcy3d_ctx) );

  PetscFunctionReturn(0);
}
