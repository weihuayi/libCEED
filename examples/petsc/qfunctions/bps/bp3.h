// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// libCEED QFunctions for diffusion operator example using PETSc

#ifndef bp3_h
#define bp3_h

#include <ceed.h>
#include <math.h>

// -----------------------------------------------------------------------------
// This QFunction sets up the geometric factors required to apply the
//   diffusion operator
//
// We require the product of the inverse of the Jacobian and its transpose to
//   properly compute integrals of the form: int( gradv gradu)
//
// Determinant of Jacobian:
//   detJ = J11*A11 + J21*A12 + J31*A13
//     Jij = Jacobian entry ij
//     Aij = Adjoint ij
//
// Inverse of Jacobian:
//   Bij = Aij / detJ
//
// Product of Inverse and Transpose:
//   BBij = sum( Bik Bkj )
//
// Stored: w B^T B detJ = w A^T A / detJ
//   Note: This matrix is symmetric, so we only store 6 distinct entries
//     qd: 0 3 6
//         1 4 7
//         2 5 8
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupDiffGeo)(void *ctx, CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out) {
  const CeedScalar *J = in[1], *w = in[2]; // Note: *X = in[0]
  CeedScalar *qd = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar J11 = J[i+Q*0];
    const CeedScalar J21 = J[i+Q*1];
    const CeedScalar J31 = J[i+Q*2];
    const CeedScalar J12 = J[i+Q*3];
    const CeedScalar J22 = J[i+Q*4];
    const CeedScalar J32 = J[i+Q*5];
    const CeedScalar J13 = J[i+Q*6];
    const CeedScalar J23 = J[i+Q*7];
    const CeedScalar J33 = J[i+Q*8];
    const CeedScalar A11 = J22*J33 - J23*J32;
    const CeedScalar A12 = J13*J32 - J12*J33;
    const CeedScalar A13 = J12*J23 - J13*J22;
    const CeedScalar A21 = J23*J31 - J21*J33;
    const CeedScalar A22 = J11*J33 - J13*J31;
    const CeedScalar A23 = J13*J21 - J11*J23;
    const CeedScalar A31 = J21*J32 - J22*J31;
    const CeedScalar A32 = J12*J31 - J11*J32;
    const CeedScalar A33 = J11*J22 - J12*J21;
    const CeedScalar qw = w[i] / (J11*A11 + J21*A12 + J31*A13);
    qd[i+Q*0] = qw * (A11*A11 + A12*A12 + A13*A13);
    qd[i+Q*1] = qw * (A11*A21 + A12*A22 + A13*A23);
    qd[i+Q*2] = qw * (A11*A31 + A12*A32 + A13*A33);
    qd[i+Q*3] = qw * (A21*A21 + A22*A22 + A23*A23);
    qd[i+Q*4] = qw * (A21*A31 + A22*A32 + A23*A33);
    qd[i+Q*5] = qw * (A31*A31 + A32*A32 + A33*A33);
    qd[i+Q*6] = w[i] * (J11*A11 + J21*A12 + J31*A13);
  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// This QFunction sets up the rhs and true solution for the problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupDiffRhs)(void *ctx, CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out) {
#ifndef M_PI
#  define M_PI    3.14159265358979323846
#endif
  const CeedScalar *x = in[0], *w = in[1];
  CeedScalar *true_soln = out[0], *rhs = out[1];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar c[3] = { 0, 1., 2. };
    const CeedScalar k[3] = { 1., 2., 3. };

    true_soln[i] = sin(M_PI*(c[0] + k[0]*x[i+Q*0])) *
                   sin(M_PI*(c[1] + k[1]*x[i+Q*1])) *
                   sin(M_PI*(c[2] + k[2]*x[i+Q*2]));

    rhs[i] = w[i+Q*6] * M_PI*M_PI * (k[0]*k[0] + k[1]*k[1] + k[2]*k[2]) *
             true_soln[i];
  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// This QFunction applies the diffusion operator for a scalar field.
//
// Inputs:
//   ug     - Input vector gradient at quadrature points
//   q_data  - Geometric factors
//
// Output:
//   vg     - Output vector (test functions) gradient at quadrature points
//
// -----------------------------------------------------------------------------
CEED_QFUNCTION(Diff)(void *ctx, CeedInt Q,
                     const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *ug = in[0], *q_data = in[1];
  CeedScalar *vg = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of u
    const CeedScalar du[3]            =  {ug[i+Q*0],
                                          ug[i+Q*1],
                                          ug[i+Q*2]
                                         };
    // Read q_data (dXdxdXdx_T symmetric matrix)
    const CeedScalar dXdxdXdx_T[3][3] = {{q_data[i+0*Q],
                                          q_data[i+1*Q],
                                          q_data[i+2*Q]},
                                         {q_data[i+1*Q],
                                          q_data[i+3*Q],
                                          q_data[i+4*Q]},
                                         {q_data[i+2*Q],
                                          q_data[i+4*Q],
                                          q_data[i+5*Q]}
                                        };

    for (int j=0; j<3; j++) // j = direction of vg
      vg[i+j*Q] = (du[0] * dXdxdXdx_T[0][j] +
                   du[1] * dXdxdXdx_T[1][j] +
                   du[2] * dXdxdXdx_T[2][j]);

  } // End of Quadrature Point Loop
  return 0;
}
// -----------------------------------------------------------------------------

#endif // bp3_h
