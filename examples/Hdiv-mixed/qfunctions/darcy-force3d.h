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
/// RHS of Darcy problem 3D (hex element) using PETSc

#ifndef DARCY_FORCE3D_H
#define DARCY_FORCE3D_H

#include <math.h>

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

// -----------------------------------------------------------------------------
// Compute determinant of 3x3 matrix
// -----------------------------------------------------------------------------
#ifndef DetMat
#define DetMat
CEED_QFUNCTION_HELPER CeedScalar ComputeDetMat(const CeedScalar A[3][3]) {
  // Compute det(A)
  const CeedScalar B11 = A[1][1]*A[2][2] - A[1][2]*A[2][1];
  const CeedScalar B12 = A[0][2]*A[2][1] - A[0][1]*A[2][2];
  const CeedScalar B13 = A[0][1]*A[1][2] - A[0][2]*A[1][1];
  CeedScalar detA = A[0][0]*B11 + A[1][0]*B12 + A[2][0]*B13;

  return detA;
};
#endif

// -----------------------------------------------------------------------------
// Strong form:
//  u       = -\grad(p)      on \Omega
//  \div(u) = f              on \Omega
//  p = p0                   on \Gamma_D
//  u.n = g                  on \Gamma_N
// Weak form: Find (u,p) \in VxQ (V=H(div), Q=L^2) on \Omega
//  (v, u) - (\div(v), p) = -<v, p0 n>_{\Gamma_D}
// -(q, \div(u))          = -(q, f)
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
CEED_QFUNCTION(DarcyForce3D)(void *ctx, const CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*coords) = in[0],
                   (*w) = in[1],
                   (*dxdX)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar (*rhs_u) = out[0], (*rhs_p) = out[1],
             (*true_soln) = out[2];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Setup, (x,y,z) and J = dx/dX
    CeedScalar x = coords[i+0*Q], y = coords[i+1*Q], z = coords[i+2*Q];
    const CeedScalar J[3][3] = {{dxdX[0][0][i], dxdX[1][0][i], dxdX[2][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i], dxdX[2][1][i]},
                                {dxdX[0][2][i], dxdX[1][2][i], dxdX[2][2][i]}};
    const CeedScalar detJ = ComputeDetMat(J);             
    // *INDENT-ON*
    CeedScalar pe = sin(M_PI*x) * sin(M_PI*y) * sin(M_PI*z) + M_PI*x*y*z;
    CeedScalar ue[3] = {-M_PI*cos(M_PI*x) *sin(M_PI*y) *sin(M_PI*z) - M_PI *y*z,
                        -M_PI*sin(M_PI*x) *cos(M_PI*y) *sin(M_PI*z) - M_PI *x*z,
                        -M_PI*sin(M_PI*x) *sin(M_PI*y) *cos(M_PI*z) - M_PI *x *y
                       };
    CeedScalar f = 3*M_PI*M_PI*sin(M_PI*x)*sin(M_PI*y)*sin(M_PI*z);

    // 1st eq: component 1
    rhs_u[i+0*Q] = 0.;
    // 1st eq: component 2
    rhs_u[i+1*Q] = 0.;
    // 1st eq: component 2
    rhs_u[i+2*Q] = 0.;
    // 2nd eq
    rhs_p[i] = -f*w[i]*detJ;
    // True solution Ue=[p,u]
    true_soln[i+0*Q] = pe;
    true_soln[i+1*Q] = ue[0];
    true_soln[i+2*Q] = ue[1];
    true_soln[i+3*Q] = ue[2];
  } // End of Quadrature Point Loop
  return 0;
}
// -----------------------------------------------------------------------------

#endif //End of DARCY_FORCE3D_H
