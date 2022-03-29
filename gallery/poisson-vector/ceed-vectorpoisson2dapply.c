// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <string.h>
#include "ceed-vectorpoisson2dapply.h"

/**
  @brief Set fields for Ceed QFunction applying the 2D Poisson operator
           on a vector system with three components
**/
static int CeedQFunctionInit_Vector3Poisson2DApply(Ceed ceed,
    const char *requested,
    CeedQFunction qf) {
  int ierr;

  // Check QFunction name
  const char *name = "Vector3Poisson2DApply";
  if (strcmp(name, requested))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                     "QFunction '%s' does not match requested name: %s",
                     name, requested);
  // LCOV_EXCL_STOP

  // Add QFunction fields
  const CeedInt dim = 2, num_comp = 3;
  ierr = CeedQFunctionAddInput(qf, "du", num_comp*dim, CEED_EVAL_GRAD);
  CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qf, "qdata", dim*(dim+1)/2, CEED_EVAL_NONE);
  CeedChk(ierr);
  ierr = CeedQFunctionAddOutput(qf, "dv", num_comp*dim, CEED_EVAL_GRAD);
  CeedChk(ierr);

  ierr = CeedQFunctionSetUserFlopsEstimate(qf, num_comp * 6); CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register Ceed QFunction for applying the 2D Poisson operator
           on a vector system with three components
**/
CEED_INTERN int CeedQFunctionRegister_Vector3Poisson2DApply(void) {
  return CeedQFunctionRegister("Vector3Poisson2DApply", Vector3Poisson2DApply_loc,
                               1, Vector3Poisson2DApply,
                               CeedQFunctionInit_Vector3Poisson2DApply);
}
