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
/// Force of Darcy problem 2D (quad element) using PETSc

#ifndef DARCY_FORCE2D_H
#define DARCY_FORCE2D_H

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
// This QFunction sets up the force and true solution for the above problem
// Inputs:
//   x     : interpolation of the physical coordinate
//   w     : weight of quadrature
//   J     : dx/dX. x physical coordinate, X reference coordinate [-1,1]^dim
//
// Output:
//   force_u     : which is 0.0 for this problem (-<v, p0 n> is in pressure-boundary qfunction)
//   force_p     : -(q, f) = -\int( q * f * w*detJ)dx
// -----------------------------------------------------------------------------
#ifndef DARCY_CTX
#define DARCY_CTX
typedef struct DARCYContext_ *DARCYContext;
struct DARCYContext_ {
  CeedScalar kappa;
};
#endif
CEED_QFUNCTION(DarcyForce2D)(void *ctx, const CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*coords) = in[0],
                   (*w) = in[1],
                   (*dxdX)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar (*rhs_u) = out[0], (*rhs_p) = out[1],
             (*true_soln) = out[2];
  // Context
  DARCYContext  context = (DARCYContext)ctx;
  const CeedScalar    kappa   = context->kappa;
  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Setup, (x,y) and J = dx/dX
    CeedScalar x = coords[i+0*Q], y = coords[i+1*Q];
    const CeedScalar J[2][2] = {{dxdX[0][0][i], dxdX[1][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i]}};
    const CeedScalar det_J = MatDet2x2(J);
    // *INDENT-ON*
    CeedScalar pe = sin(PI_DOUBLE*x) * sin(PI_DOUBLE*y);
    CeedScalar grad_pe[2] = {PI_DOUBLE*cos(PI_DOUBLE*x) *sin(PI_DOUBLE*y), PI_DOUBLE*sin(PI_DOUBLE*x) *cos(PI_DOUBLE*y)};
    CeedScalar K[2][2] = {{kappa, 0.},{0., kappa}};
    CeedScalar ue[2];
    AlphaMatVecMult2x2(-1., K, grad_pe, ue);
    CeedScalar f = 2*PI_DOUBLE*PI_DOUBLE*sin(PI_DOUBLE*x)*sin(PI_DOUBLE*y);

    // 1st eq: component 1
    rhs_u[i+0*Q] = 0.;
    // 1st eq: component 2
    rhs_u[i+1*Q] = 0.;
    // 2nd eq
    rhs_p[i] = -f*w[i]*det_J;
    // True solution Ue=[p,u]
    true_soln[i+0*Q] = pe;
    true_soln[i+1*Q] = ue[0];
    true_soln[i+2*Q] = ue[1];
  } // End of Quadrature Point Loop
  return 0;
}
// -----------------------------------------------------------------------------

#endif //End of DARCY_FORCE2D_H
