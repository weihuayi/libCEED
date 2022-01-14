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
/// Compute true solution of the H(div) example using PETSc
//     d4     d5
//    2---------3
//  d7|         |d3
//    |         |
//  d6|         |d2
//    0---------1
//     d0     d1
#ifndef TRUE_H
#define TRUE_H

#include <math.h>
#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif
// -----------------------------------------------------------------------------
// Compuet true solution
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupTrueSoln2D)(void *ctx, const CeedInt Q,
                                const CeedScalar *const *in,
                                CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*coords) = in[0],
                   (*dxdX)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[1];
  // Outputs
  CeedScalar (*true_u) = out[0];//, (*true_p) = out[1];

  // Setup, J = dx/dX
  const CeedScalar J0[2][2] = {{dxdX[0][0][0], dxdX[1][0][0]},
                               {dxdX[0][1][0], dxdX[1][1][0]}};
  const CeedScalar J1[2][2] = {{dxdX[0][0][1], dxdX[1][0][1]},
                               {dxdX[0][1][1], dxdX[1][1][1]}};
  const CeedScalar J2[2][2] = {{dxdX[0][0][2], dxdX[1][0][2]},
                               {dxdX[0][1][2], dxdX[1][1][2]}};
  const CeedScalar J3[2][2] = {{dxdX[0][0][3], dxdX[1][0][3]},
                               {dxdX[0][1][3], dxdX[1][1][3]}};
  CeedScalar x0 = coords[0+0*Q], y0 = coords[0+1*Q];
  CeedScalar x1 = coords[1+0*Q], y1 = coords[1+1*Q];
  CeedScalar x2 = coords[2+0*Q], y2 = coords[2+1*Q];
  CeedScalar x3 = coords[3+0*Q], y3 = coords[3+1*Q];
  CeedScalar ue0[2] = {-M_PI*cos(M_PI*x0)*sin(M_PI*y0),
                       -M_PI*sin(M_PI*x0)*cos(M_PI*y0)};
  CeedScalar ue1[2] = {-M_PI*cos(M_PI*x1)*sin(M_PI*y1),
                       -M_PI*sin(M_PI*x1)*cos(M_PI*y1)};
  CeedScalar ue2[2] = {-M_PI*cos(M_PI*x2)*sin(M_PI*y2),
                       -M_PI*sin(M_PI*x2)*cos(M_PI*y2)};
  CeedScalar ue3[2] = {-M_PI*cos(M_PI*x3)*sin(M_PI*y3),
                       -M_PI*sin(M_PI*x3)*cos(M_PI*y3)};
  CeedScalar nl0[2] = {-J0[1][1],J0[0][1]};
  CeedScalar nb0[2] = {J0[1][0],-J0[0][0]};
  CeedScalar nr1[2] = {J1[1][1],-J1[0][1]};
  CeedScalar nb1[2] = {J1[1][0],-J1[0][0]};
  CeedScalar nl2[2] = {-J2[1][1],J2[0][1]};
  CeedScalar nt2[2] = {-J2[1][0],J2[0][0]};
  CeedScalar nr3[2] = {J3[1][1],-J3[0][1]};
  CeedScalar nt3[2] = {-J3[1][0],J3[0][0]};
  CeedScalar d0, d1, d2, d3, d4, d5, d6, d7;
  d0 = ue0[0]*nb0[0]+ue0[1]*nb0[1];
  d1 = ue1[0]*nb1[0]+ue1[1]*nb1[1];
  d2 = ue1[0]*nr1[0]+ue1[1]*nr1[1];
  d3 = ue3[0]*nr3[0]+ue3[1]*nr3[1];
  d4 = ue2[0]*nt2[0]+ue2[1]*nt2[1];
  d5 = ue3[0]*nt3[0]+ue3[1]*nt3[1];
  d6 = ue0[0]*nl0[0]+ue0[1]*nl0[1];
  d7 = ue2[0]*nl2[0]+ue2[1]*nl2[1];
  // True solution projected in H(div) space
  true_u[0] = d0;
  true_u[1] = d1;
  true_u[2] = d2;
  true_u[3] = d3;
  true_u[4] = d4;
  true_u[5] = d5;
  true_u[6] = d6;
  true_u[7] = d7;
  //CeedScalar x = 0.5*(x1+x2), y = 0.5*(y1+y3);
  //CeedScalar pe = x*(1-x)*y*(1-y);
  //true_p[0] = pe;

  return 0;
}
// -----------------------------------------------------------------------------

#endif // End TRUE_H
