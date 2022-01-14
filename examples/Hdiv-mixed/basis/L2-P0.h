// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
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

// Build L2 constant basis

static void L2BasisP0(CeedInt Q, CeedScalar *q_ref, CeedScalar *q_weights,
                      CeedScalar *interp, CeedQuadMode quad_mode) {

  // Get 1D quadrature on [-1,1]
  CeedScalar q_ref_1d[Q], q_weight_1d[Q];
  switch (quad_mode) {
  case CEED_GAUSS:
    CeedGaussQuadrature(Q, q_ref_1d, q_weight_1d);
    break;
  case CEED_GAUSS_LOBATTO:
    CeedLobattoQuadrature(Q, q_ref_1d, q_weight_1d);
    break;
  }

  // P0 L2 basis is just a constant
  CeedScalar P0[1] = {1.0};
  // Loop over quadrature points
  for (CeedInt i=0; i<Q; i++) {
    for (CeedInt j=0; j<Q; j++) {
      CeedInt k1 = Q*i+j;
      q_ref[k1] = q_ref_1d[j];
      q_ref[k1 + Q*Q] = q_ref_1d[i];
      q_weights[k1] = q_weight_1d[j]*q_weight_1d[i];
      interp[k1] = P0[0];
    }
  }
}


