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
/// Richard problem 2D (quad element) using PETSc

#ifndef RICHARD_SYSTEM2D_H
#define RICHARD_SYSTEM2D_H

#include <math.h>

// See Matthew Farthing, Christopher Kees, Cass Miller (2003)
// https://www.sciencedirect.com/science/article/pii/S0309170802001872
// -----------------------------------------------------------------------------
// Strong form:
//  (rho_0^2*norm(g)/rho*k_r)*K^{-1} * u = -\grad(p) + rho*g          in \Omega x [0,T]
//  -\div(u)                             = -f  + d (rho/rho_0*theta)/dt    in \Omega x [0,T]
//  p                                    = p_b                        on \Gamma_D x [0,T]
//  u.n                                  = u_b                        on \Gamma_N x [0,T]
//  p                                    = p_0                        in \Omega, t = 0
//
//  Note: g is gravity vector, rho = rho_0*exp(beta * (p - p0)), p0 = 101325 Pa is atmospheric pressure
//  f = fs/rho_0, where g is gravity, rho_0 is the density at p_0, K = K_star*I, and
//  k_r = b_a + alpha_a * (psi - x2), where psi = p / (rho_0 * norm(g)) and x2 is vertical axis
//
// Weak form: Find (u, p) \in VxQ (V=H(div), Q=L^2) on \Omega
//  (v, (rho_0^2*norm(g)/rho*k_r)*K^{-1} * u) - (\div(v), p) =  (v, rho*g) - <v, p_b*n>_{\Gamma_D}
// -(q, \div(u))                                             = -(q, f)     + (v, d (rho*theta)/dt )
//
// This QFunction setup the mixed form of the above equation
// Inputs:
//   w     : weight of quadrature
//   J     : dx/dX. x physical coordinate, X reference coordinate [-1,1]^dim
//   u     : basis_u at quadrature points
// div(u)  : divergence of basis_u at quadrature points
//   p     : basis_p at quadrature points
//
// Output:
//   v     : (v,u) = \int (v^T * u detJ*w) ==> \int (v^T J^T*J*u*w/detJ)
// div(v)  : -(\div(v), p) = -\int (div(v)^T * p *w)
//   q     : -(q, \div(u)) = -\int (q^T * div(u) *w)
// which create the following coupled system
//                            D = [ M  B^T ]
//                                [ B   0  ]
// M = (v,u), B = -(q, \div(u))
// Note we need to apply Piola map on the basis_u, which is J*u/detJ
// So (v,u) = \int (v^T * u detJ*w) ==> \int (v^T J^T*J*u*w/detJ)
// -----------------------------------------------------------------------------
// We have 3 experiment parameters as described in Table 1:P1, P2, P3
// Matthew Farthing, Christopher Kees, Cass Miller (2003)
// https://www.sciencedirect.com/science/article/pii/S0309170802001872
typedef struct RICHARDP1Context_ *RICHARDP1Context;
struct RICHARDP1Context_ {
  CeedScalar K_star;
  CeedScalar alpha_a;
  CeedScalar b_a;
  CeedScalar rho_0;
  CeedScalar beta;
};
// -----------------------------------------------------------------------------
// Residual evaluation for Richard problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(RichardSystem2D)(void *ctx, CeedInt Q,
                                const CeedScalar *const *in,
                                CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*dxdX)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[1],
                   (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*div_u) = (const CeedScalar(*))in[3],
                   (*p) = (const CeedScalar(*))in[4],
                   (*coords) = in[5];

  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*div_v) = (CeedScalar(*))out[1],
             (*q) = (CeedScalar(*))out[2];
  // Context
  //const RICHARDP1Context  context = (RICHARDP1Context)ctx;
  const CeedScalar K_star  = 10.;
  const CeedScalar alpha_a = 1.;
  const CeedScalar b_a     = 10.;
  const CeedScalar rho_0   = 998.2;
  const CeedScalar beta    = 0.;
  const CeedScalar g       = 9.8;
  const CeedScalar p0      = 101325; // atmospheric pressure
  // *INDENT-ON*

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // *INDENT-OFF*
    // Setup, J = dx/dX
    CeedScalar y = coords[i+1*Q];
    const CeedScalar J[2][2] = {{dxdX[0][0][i], dxdX[1][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i]}};
    const CeedScalar detJ = J[0][0]*J[1][1] - J[0][1]*J[1][0];

    // *INDENT-ON*
    // psi = p / (rho_0 * norm(g))
    CeedScalar psi = p[i] / (rho_0 * g);
    // k_r = b_a + alpha_a * (psi - x2)
    CeedScalar k_r = b_a + alpha_a * (psi - y);
    // rho = rho_0*exp(beta * (p - p0))
    CeedScalar rho = rho_0 * exp(beta * (p[i] - p0));
    //k = rho_0^2*norm(g)/(rho*k_r)
    CeedScalar k = rho_0 * rho_0 * g / (rho * k_r);

    // Piola map: J^T*k*K^{-1}*J*u*w/detJ
    // Note K = K_star*I ==> K^{-1} = 1/K_star * I
    // 1) Compute J^T*k*K^{-1}*J
    CeedScalar JTkJ[2][2];
    for (CeedInt j = 0; j < 2; j++) {
      for (CeedInt l = 0; l < 2; l++) {
        JTkJ[j][l] = 0;
        for (CeedInt m = 0; m < 2; m++)
          JTkJ[j][l] += (k / K_star) * J[m][j] * J[m][l];
      }
    }
    // 2) Compute J^T*k*K^{-1}*J*u*w/detJ
    for (CeedInt l = 0; l < 2; l++) {
      v[l][i] = 0;
      for (CeedInt m = 0; m < 2; m++)
        v[l][i] += JTkJ[l][m] * u[m][i] * w[i]/detJ;
    }

    div_v[i] = -p[i] * w[i];
    q[i] = -div_u[i] * w[i];
  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Jacobian evaluation for Richard problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(JacobianRichardSystem2D)(void *ctx, CeedInt Q,
                                        const CeedScalar *const *in,
                                        CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*dxdX)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[1],
                   (*du)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*div_du) = (const CeedScalar(*))in[3],
                   (*dp) = (const CeedScalar(*))in[4],
                   (*coords) = in[5],
                   (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[6],
                   (*p) = (const CeedScalar(*))in[7];

  // Outputs
  CeedScalar (*dv)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*div_dv) = (CeedScalar(*))out[1],
             (*dq) = (CeedScalar(*))out[2];
  // Context
  //const RICHARDP1Context  context = (RICHARDP1Context)ctx;
  const CeedScalar K_star  = 10.;
  const CeedScalar alpha_a = 1.;
  const CeedScalar b_a     = 10.;
  const CeedScalar rho_0   = 998.2;
  const CeedScalar beta    = 0.;
  const CeedScalar g       = 9.8;
  const CeedScalar p0      = 101325; // atmospheric pressure
  // *INDENT-ON*

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // *INDENT-OFF*
    // Setup, J = dx/dX
    CeedScalar y = coords[i+1*Q];
    const CeedScalar J[2][2] = {{dxdX[0][0][i], dxdX[1][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i]}};
    const CeedScalar detJ = J[0][0]*J[1][1] - J[0][1]*J[1][0];

    // *INDENT-ON*
    // psi = p / (rho_0 * norm(g))
    CeedScalar psi = p[i] / (rho_0 * g);
    // k_r = b_a + alpha_a * (psi - x2)
    CeedScalar k_r = b_a + alpha_a * (psi - y);
    // rho = rho_0*exp(beta * (p - p0))
    CeedScalar rho = rho_0 * exp(beta * (p[i] - p0));
    //k = rho_0^2*norm(g)/(rho*k_r)
    CeedScalar k = rho_0 * rho_0 * g / (rho * k_r);

    // Piola map: J^T*k*K^{-1}*J*u*w/detJ
    // Note K = K_star*I ==> K^{-1} = 1/K_star * I
    // The jacobian term
    // dv = J^T* (k*K^{-1}) *J*du*w/detJ - J^T*(k*K^{-1} [(rho*k_r)/dp]*dp/(rho*k_r)) *J*u

    // 1) Compute J^T* (k*K^{-1}) *J
    CeedScalar JTkJ[2][2];
    for (CeedInt j = 0; j < 2; j++) {
      for (CeedInt l = 0; l < 2; l++) {
        JTkJ[j][l] = 0;
        for (CeedInt m = 0; m < 2; m++)
          JTkJ[j][l] += (k / K_star) * J[m][j] * J[m][l];
      }
    }

    // 2) Compute [(rho*k_r)/dp]*dp/(rho*k_r))
    CeedScalar d_rhokr_dp = (beta + alpha_a/(rho_0*g*k_r))*dp[i];
    // 3) Compute dv
    for (CeedInt l = 0; l < 2; l++) {
      dv[l][i] = 0;
      for (CeedInt m = 0; m < 2; m++)
        dv[l][i] += JTkJ[l][m] * du[m][i] * w[i]/detJ  - JTkJ[l][m]*d_rhokr_dp*u[m][i];
    }

    div_dv[i] = -dp[i] * w[i];
    dq[i] = -div_du[i] * w[i];
  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------

#endif //End of DARCY_MASS2D_H
