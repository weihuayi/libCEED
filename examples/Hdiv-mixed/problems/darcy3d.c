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
#include "../qfunctions/darcy-system3d.h"
#include "../qfunctions/darcy-error3d.h"
#include "../qfunctions/pressure-boundary3d.h"

PetscErrorCode Hdiv_DARCY3D(Ceed ceed, ProblemData problem_data, void *ctx) {
  AppCtx            app_ctx = *(AppCtx *)ctx;
  DARCYContext         darcy_ctx;
  CeedQFunctionContext darcy_context;

  PetscFunctionBeginUser;

  PetscCall( PetscCalloc1(1, &darcy_ctx) );

  // ------------------------------------------------------
  //               SET UP POISSON_QUAD2D
  // ------------------------------------------------------
  problem_data->dim                     = 3;
  problem_data->elem_node               = 8;
  problem_data->q_data_size_face        = 4;
  problem_data->quadrature_mode         = CEED_GAUSS;
  problem_data->force                   = DarcyForce3D;
  problem_data->force_loc               = DarcyForce3D_loc;
  problem_data->residual                = DarcySystem3D;
  problem_data->residual_loc            = DarcySystem3D_loc;
  problem_data->jacobian                = JacobianDarcySystem3D;
  problem_data->jacobian_loc            = JacobianDarcySystem3D_loc;
  problem_data->error                   = DarcyError3D;
  problem_data->error_loc               = DarcyError3D_loc;
  problem_data->bc_pressure             = BCPressure3D;
  problem_data->bc_pressure_loc         = BCPressure3D_loc;

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  CeedScalar kappa = 1.;
  PetscOptionsBegin(app_ctx->comm, NULL, "Options for Hdiv-mixed problem", NULL);
  PetscCall( PetscOptionsScalar("-kappa", "Hydraulic Conductivity", NULL,
                                kappa, &kappa, NULL));
  PetscOptionsEnd();

  darcy_ctx->kappa = kappa;
  CeedQFunctionContextCreate(ceed, &darcy_context);
  CeedQFunctionContextSetData(darcy_context, CEED_MEM_HOST, CEED_COPY_VALUES,
                              sizeof(*darcy_ctx), darcy_ctx);
  problem_data->qfunction_context = darcy_context;
  CeedQFunctionContextSetDataDestroy(darcy_context, CEED_MEM_HOST,
                                     FreeContextPetsc);

  PetscFunctionReturn(0);
}
