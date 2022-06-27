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
/// Utility functions for setting up Richard problem in 2D

#include "../include/register-problem.h"
#include "../qfunctions/richard-system2d.h"
#include "../qfunctions/pressure-boundary2d.h"

PetscErrorCode Hdiv_RICHARD2D(Ceed ceed, ProblemData problem_data, void *ctx) {
  AppCtx               app_ctx = *(AppCtx *)ctx;
  //RICHARDContext       richard_ctx;
  //CeedQFunctionContext richard_context;

  PetscFunctionBeginUser;

  //PetscCall( PetscCalloc1(1, &richard_ctx) );

  // ------------------------------------------------------
  //               SET UP POISSON_QUAD2D
  // ------------------------------------------------------
  problem_data->dim                     = 2;
  problem_data->elem_node               = 4;
  problem_data->q_data_size_face        = 3;
  problem_data->quadrature_mode         = CEED_GAUSS;
  //problem_data->force                   = DarcyForce2D;
  //problem_data->force_loc               = DarcyForce2D_loc;
  problem_data->residual                = RichardSystem2D;
  problem_data->residual_loc            = RichardSystem2D_loc;
  problem_data->jacobian                = JacobianRichardSystem2D;
  problem_data->jacobian_loc            = JacobianRichardSystem2D_loc;
  //problem_data->error                   = DarcyError2D;
  //problem_data->error_loc               = DarcyError2D_loc;
  problem_data->bc_pressure             = BCPressure2D;
  problem_data->bc_pressure_loc         = BCPressure2D_loc;

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  PetscOptionsBegin(app_ctx->comm, NULL, "Options for Hdiv-mixed problem", NULL);
  PetscOptionsEnd();

  //PetscCall( PetscFree(richard_ctx) );
  PetscFunctionReturn(0);
}
