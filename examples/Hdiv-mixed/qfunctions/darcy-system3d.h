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
/// Darcy problem 3D (hex element) using PETSc

#ifndef DARCY_SYSTEM3D_H
#define DARCY_SYSTEM3D_H

#include <math.h>
#include "utils.h"

// -----------------------------------------------------------------------------
// Strong form:
//  u       = -K * \grad(p)  on \Omega
//  \div(u) = f              on \Omega
//  p = p0                   on \Gamma_D
//  u.n = g                  on \Gamma_N
// Weak form: Find (u,p) \in VxQ (V=H(div), Q=L^2) on \Omega
//  (v, K^{-1}*u) - (\div(v), p) = -<v, p0 n>_{\Gamma_D}
// -(q, \div(u))                 = -(q, f)
// This QFunction setup the mixed form of the above equation
// Inputs:
//   w     : weight of quadrature
//   J     : dx/dX. x physical coordinate, X reference coordinate [-1,1]^dim
//   u     : basis_u at quadrature points
// div(u)  : divergence of basis_u at quadrature points
//   p     : basis_p at quadrature points
//
// Output:
// Note we need to apply Piola map on the basis_u, which is J*u/detJ
//   v     : (v, K^{-1} * u) = \int (v^T * K^{-1} u detJ*w) ==> \int (v^T J^T*K^{-1}*J*u*w/detJ)
// div(v)  : -(\div(v), p) = -\int (div(v)^T * p *w)
//   q     : -(q, \div(u)) = -\int (q^T * div(u) *w)
// which create the following coupled system
//                            D = [ M  B^T ]
//                                [ B   0  ]
// M = (v, K^{-1} * u), B = -(q, \div(u))
// -----------------------------------------------------------------------------
#ifndef DARCY_CTX
#define DARCY_CTX
typedef struct DARCYContext_ *DARCYContext;
struct DARCYContext_ {
  CeedScalar kappa;
};
#endif
// -----------------------------------------------------------------------------
// Residual evaluation for Darcy problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(DarcySystem3D)(void *ctx, CeedInt Q,
                              const CeedScalar *const *in,
                              CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*dxdX)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1],
                   (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*div_u) = (const CeedScalar(*))in[3],
                   (*p) = (const CeedScalar(*))in[4];

  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*div_v) = (CeedScalar(*))out[1],
             (*q) = (CeedScalar(*))out[2];
  // Context
  DARCYContext  context = (DARCYContext)ctx;
  const CeedScalar    kappa   = context->kappa;
  // *INDENT-ON*

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // *INDENT-OFF*
    // Setup, J = dx/dX
    const CeedScalar J[3][3] = {{dxdX[0][0][i], dxdX[1][0][i], dxdX[2][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i], dxdX[2][1][i]},
                                {dxdX[0][2][i], dxdX[1][2][i], dxdX[2][2][i]}};
    const CeedScalar det_J = MatDet3x3(J);

    // Piola map: J^T*K^{-1}*J*u*w/detJ
    // 1) Compute K^{-1}, note K = kappa*I
    CeedScalar K[3][3] = {{kappa, 0., 0.},
                          {0., kappa, 0.},
                          {0., 0., kappa}
                         };
    const CeedScalar det_K = MatDet3x3(K);
    CeedScalar K_inv[3][3];
    MatInverse3x3(K, det_K, K_inv);

    // 2) Compute K^{-1}*J
    CeedScalar Kinv_J[3][3];
    AlphaMatMatMult3x3(1., K_inv, J, Kinv_J);

    // 3) Compute J^T * (K^{-1}*J)
    CeedScalar JT_Kinv_J[3][3];
    AlphaMatTransposeMatMult3x3(1, J, Kinv_J, JT_Kinv_J);

    // 4) Compute (J^T*K^{-1}*J) * u * w /detJ
    CeedScalar u1[3] = {u[0][i], u[1][i], u[2][i]}, v1[3];
    AlphaMatVecMult3x3(w[i]/det_J, JT_Kinv_J, u1, v1);

    // Output at quadrature points
    for (CeedInt k = 0; k < 3; k++) {
      v[k][i] = v1[k];
    }

    div_v[i] = -p[i] * w[i];
    q[i] = -div_u[i] * w[i];
  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Jacobian evaluation for Darcy problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(JacobianDarcySystem3D)(void *ctx, CeedInt Q,
                                      const CeedScalar *const *in,
                                      CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*dxdX)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1],
                   (*du)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*div_du) = (const CeedScalar(*))in[3],
                   (*dp) = (const CeedScalar(*))in[4];

  // Outputs
  CeedScalar (*dv)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*div_dv) = (CeedScalar(*))out[1],
             (*dq) = (CeedScalar(*))out[2];
  // Context
  DARCYContext  context = (DARCYContext)ctx;
  const CeedScalar    kappa   = context->kappa;
  // *INDENT-ON*

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // *INDENT-OFF*
    // Setup, J = dx/dX
    const CeedScalar J[3][3] = {{dxdX[0][0][i], dxdX[1][0][i], dxdX[2][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i], dxdX[2][1][i]},
                                {dxdX[0][2][i], dxdX[1][2][i], dxdX[2][2][i]}};
    const CeedScalar det_J = MatDet3x3(J);

    // *INDENT-ON*
    // Piola map: J^T*K^{-1}*J*du*w/detJ
    // 1) Compute K^{-1}, note K = kappa*I
    CeedScalar K[3][3] = {{kappa, 0., 0.},
      {0., kappa, 0.},
      {0., 0., kappa}
    };
    const CeedScalar det_K = MatDet3x3(K);
    CeedScalar K_inv[3][3];
    MatInverse3x3(K, det_K, K_inv);

    // 2) Compute K^{-1}*J
    CeedScalar Kinv_J[3][3];
    AlphaMatMatMult3x3(1., K_inv, J, Kinv_J);

    // 3) Compute J^T * (K^{-1}*J)
    CeedScalar JT_Kinv_J[3][3];
    AlphaMatTransposeMatMult3x3(1, J, Kinv_J, JT_Kinv_J);

    // 4) Compute (J^T*K^{-1}*J) * du * w /detJ
    CeedScalar du1[3] = {du[0][i], du[1][i], du[2][i]}, dv1[3];
    AlphaMatVecMult3x3(w[i]/det_J, JT_Kinv_J, du1, dv1);


    // Output at quadrature points
    for (CeedInt k = 0; k < 3; k++) {
      dv[k][i] = dv1[k];
    }

    div_dv[i] = -dp[i] * w[i];
    dq[i] = -div_du[i] * w[i];
  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------

#endif //End of DARCY_MASS3D_H
