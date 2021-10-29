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
/// Hyperelasticity, finite strain for solid mechanics example using PETSc

#ifndef ELAS_FSInitialNH1_AD_H
#define ELAS_FSInitialNH1_AD_H

#ifndef __CUDACC__
#  include <math.h>
#endif

#ifndef PHYSICS_STRUCT
#define PHYSICS_STRUCT
typedef struct Physics_private *Physics;
struct Physics_private {
  CeedScalar   nu;      // Poisson's ratio
  CeedScalar   E;       // Young's Modulus
};
#endif

// -----------------------------------------------------------------------------
// Series approximation of log1p()
//  log1p() is not vectorized in libc
//
//  The series expansion is accurate to 1e-7 in the range sqrt(2)/2 < J < sqrt(2),
//  with machine precision accuracy near J=1.  The initialization extends this range
//  to 0.35 ~= sqrt(2)/4 < J < sqrt(2)*2 ~= 2.83, which should be sufficient for
//  applications of the Neo-Hookean model.
// -----------------------------------------------------------------------------
#ifndef LOG1P_SERIES_SHIFTED
#define LOG1P_SERIES_SHIFTED
CEED_QFUNCTION_HELPER CeedScalar log1p_series_shifted(CeedScalar x) {
  const CeedScalar left = sqrt(2.)/2 - 1, right = sqrt(2.) - 1;
  CeedScalar sum = 0;
  if (1) { // Disable if the smaller range sqrt(2) < J < sqrt(2) is sufficient
    if (x < left) { // Replace if with while for arbitrary range (may hurt vectorization)
      sum -= log(2.) / 2;
      x = 1 + 2 * x;
    } else if (right < x) {
      sum += log(2.) / 2;
      x = (x - 1) / 2;
    }
  }
  CeedScalar y = x / (2. + x);
  const CeedScalar y2 = y*y;
  sum += y;
  y *= y2;
  sum += y / 3;
  y *= y2;
  sum += y / 5;
  y *= y2;
  sum += y / 7;
  return 2 * sum;
};
#endif

// -----------------------------------------------------------------------------
// Compute det F - 1
// -----------------------------------------------------------------------------
#ifndef DETJM1
#define DETJM1
CEED_QFUNCTION_HELPER CeedScalar computeJM1(const CeedScalar grad_u[3][3]) {
  return grad_u[0][0]*(grad_u[1][1]*grad_u[2][2]-grad_u[1][2]*grad_u[2][1]) +
         grad_u[0][1]*(grad_u[1][2]*grad_u[2][0]-grad_u[1][0]*grad_u[2][2]) +
         grad_u[0][2]*(grad_u[1][0]*grad_u[2][1]-grad_u[2][0]*grad_u[1][1]) +
         grad_u[0][0] + grad_u[1][1] + grad_u[2][2] +
         grad_u[0][0]*grad_u[1][1] + grad_u[0][0]*grad_u[2][2] +
         grad_u[1][1]*grad_u[2][2] - grad_u[0][1]*grad_u[1][0] -
         grad_u[0][2]*grad_u[2][0] - grad_u[1][2]*grad_u[2][1];
};
#endif

// -----------------------------------------------------------------------------
// Compute det C - 1
// -----------------------------------------------------------------------------
#ifndef DETCM1
#define DETCM1
CEED_QFUNCTION_HELPER CeedScalar computeDetCM1(const CeedScalar E2work[6]) {
  return E2work[0]*(E2work[1]*E2work[2]-E2work[3]*E2work[3]) +
         E2work[5]*(E2work[4]*E2work[3]-E2work[5]*E2work[2]) +
         E2work[4]*(E2work[5]*E2work[3]-E2work[4]*E2work[1]) +
         E2work[0] + E2work[1] + E2work[2] +
         E2work[0]*E2work[1] + E2work[0]*E2work[2] +
         E2work[1]*E2work[2] - E2work[5]*E2work[5] -
         E2work[4]*E2work[4] - E2work[3]*E2work[3];
};
#endif

// -----------------------------------------------------------------------------
// Compute matrix^(-1), where matrix is symetric, returns array of 6
// -----------------------------------------------------------------------------
#ifndef MatinvSym
#define MatinvSym
CEED_QFUNCTION_HELPER int computeMatinvSym(const CeedScalar A[3][3],
    const CeedScalar detA, CeedScalar Ainv[6]) {
  // Compute A^(-1) : A-Inverse
  CeedScalar B[6] = {A[1][1]*A[2][2] - A[1][2]*A[2][1], /* *NOPAD* */
                     A[0][0]*A[2][2] - A[0][2]*A[2][0], /* *NOPAD* */
                     A[0][0]*A[1][1] - A[0][1]*A[1][0], /* *NOPAD* */
                     A[0][2]*A[1][0] - A[0][0]*A[1][2], /* *NOPAD* */
                     A[0][1]*A[1][2] - A[0][2]*A[1][1], /* *NOPAD* */
                     A[0][2]*A[2][1] - A[0][1]*A[2][2] /* *NOPAD* */
                    };
  for (CeedInt m = 0; m < 6; m++)
    Ainv[m] = B[m] / (detA);

  return 0;
};
#endif

// -----------------------------------------------------------------------------
// Compute Second Piola-Kirchhoff S(E)
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER int computeS(CeedScalar Swork[6], CeedScalar E2work[6],
                                   const CeedScalar lambda, const CeedScalar mu) {
  // *INDENT-OFF*
  /*
  CeedScalar E2[3][3] = {{E2work[0], E2work[5], E2work[4]},
                         {E2work[5], E2work[1], E2work[3]},
                         {E2work[4], E2work[3], E2work[2]}
                        };
  // *INDENT-ON*

  // C : right Cauchy-Green tensor
  // C = I + 2E
  // *INDENT-OFF*
  const CeedScalar C[3][3] = {{1 + E2[0][0], E2[0][1], E2[0][2]},
                              {E2[0][1], 1 + E2[1][1], E2[1][2]},
                              {E2[0][2], E2[1][2], 1 + E2[2][2]}
                             };
  // *INDENT-ON*

  // Compute C^(-1) : C-Inverse
  CeedScalar Cinvwork[6];
  */
  //const CeedScalar detCm1 = computeDetCM1(E2work);
  /*
  computeMatinvSym(C, detCm1+1, Cinvwork);

  // *INDENT-OFF*
  const CeedScalar C_inv[3][3] = {{Cinvwork[0], Cinvwork[5], Cinvwork[4]},
                                  {Cinvwork[5], Cinvwork[1], Cinvwork[3]},
                                  {Cinvwork[4], Cinvwork[3], Cinvwork[2]}
                                 };
  */
  // *INDENT-ON*

  // Compute the Second Piola-Kirchhoff (S)
  //const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
  //const CeedScalar logJ = log1p_series_shifted(detCm1) / 2.;

  for (CeedInt m = 0; m < 6; m++) {
    Swork[m] = E2work[m]*E2work[m];
    //for (CeedInt n = 0; n < 3; n++)
    //  Swork[m] += mu*C_inv[indj[m]][n]*E2[n][indk[m]];
  }


  return 0;
};

// -----------------------------------------------------------------------------
// Enzyme-AD to compute J[6][6] = \partial Swork / \partial E2work
// -----------------------------------------------------------------------------
int  __enzyme_augmentsize(void *, ...);
void __enzyme_augmentfwd(void *, ...);
void __enzyme_reverse(void *, ...);

int enzyme_dup;
int enzyme_tape;
int enzyme_const;
int enzyme_allocated;

CEED_QFUNCTION_HELPER int getEnzymeSize(void *computeSfwd) {
  return __enzyme_augmentsize(computeSfwd, enzyme_dup, enzyme_dup,
                              enzyme_const, enzyme_const);
}

CEED_QFUNCTION_HELPER void grad_S_fwd(double *S, double *dS, double *E,
                                      double *dE, const double lambda, const double mu, void *tape) {
  int size = getEnzymeSize((void *)computeS);
  __enzyme_augmentfwd((void *)computeS, enzyme_allocated, size, enzyme_tape, tape,
                      S, dS, E, dE, enzyme_const, lambda, enzyme_const, mu);
}

CEED_QFUNCTION_HELPER void grad_S_rev(double *S, double *dS, double *E,
                                      double *dE, const double lambda, const double mu, void *tape) {
  int size = getEnzymeSize((void *)computeS);
  __enzyme_reverse((void *)computeS, enzyme_allocated, size, enzyme_tape, tape,
                   S, dS, E, dE, enzyme_const, lambda, enzyme_const, mu);
}

// -----------------------------------------------------------------------------
// Residual evaluation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSInitialNH1F_AD)(void *ctx, CeedInt Q,
                                     const CeedScalar *const *in,
                                     CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
                   (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  // Outputs
  CeedScalar (*dvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0];
  // Store grad_u for HyperFSdF (Jacobian of HyperFSF)
  CeedScalar (*grad_u)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[1];
  // Store Swork
  CeedScalar (*Swork)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[2];
  // Store tape for autodiff
  void *(*tape)[CEED_Q_VLA] = (void *(*)[CEED_Q_VLA])out[3];

  // *INDENT-ON*

  // Context
  const Physics context = (Physics)ctx;
  const CeedScalar E  = context->E;
  const CeedScalar nu = context->nu;
  const CeedScalar TwoMu = E / (1 + nu);
  const CeedScalar mu = TwoMu / 2;
  const CeedScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  const CeedScalar lambda = (3*Kbulk - TwoMu) / 3;

  // Formulation Terminology:
  //  I3    : 3x3 Identity matrix
  //  C     : right Cauchy-Green tensor
  //  C_inv  : inverse of C
  //  F     : deformation gradient
  //  S     : 2nd Piola-Kirchhoff (in current config)
  //  P     : 1st Piola-Kirchhoff (in referential config)
  // Formulation:
  //  F =  I3 + grad_ue
  //  J = det(F)
  //  C = F(^T)*F
  //  S = mu*I3 + (lambda*log(J)-mu)*C_inv;
  //  P = F*S

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of u
    // *INDENT-OFF*
    const CeedScalar du[3][3]   = {{ug[0][0][i],
                                    ug[1][0][i],
                                    ug[2][0][i]},
                                   {ug[0][1][i],
                                    ug[1][1][i],
                                    ug[2][1][i]},
                                   {ug[0][2][i],
                                    ug[1][2][i],
                                    ug[2][2][i]}
                                  };
    // -- Qdata
    const CeedScalar wdetJ      =   q_data[0][i];
    const CeedScalar dXdx[3][3] = {{q_data[1][i],
                                    q_data[2][i],
                                    q_data[3][i]},
                                   {q_data[4][i],
                                    q_data[5][i],
                                    q_data[6][i]},
                                   {q_data[7][i],
                                    q_data[8][i],
                                    q_data[9][i]}
                                  };
    // *INDENT-ON*

    // Compute grad_u
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to du = grad_u
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        grad_u[j][k][i] = 0;
        for (CeedInt m = 0; m < 3; m++)
          grad_u[j][k][i] += dXdx[m][k] * du[j][m];
      }

    // I3 : 3x3 Identity matrix
    // Compute The Deformation Gradient : F = I3 + grad_u
    // *INDENT-OFF*
    const CeedScalar F[3][3] =  {{grad_u[0][0][i] + 1,
                                  grad_u[0][1][i],
                                  grad_u[0][2][i]},
                                 {grad_u[1][0][i],
                                  grad_u[1][1][i] + 1,
                                  grad_u[1][2][i]},
                                 {grad_u[2][0][i],
                                  grad_u[2][1][i],
                                  grad_u[2][2][i] + 1}
                                };

    // *INDENT-OFF*
    const CeedScalar tempgradu[3][3] =  {{grad_u[0][0][i],
                                          grad_u[0][1][i],
                                          grad_u[0][2][i]},
                                         {grad_u[1][0][i],
                                          grad_u[1][1][i],
                                          grad_u[1][2][i]},
                                         {grad_u[2][0][i],
                                          grad_u[2][1][i],
                                          grad_u[2][2][i]}
                                        };
    // *INDENT-ON*
    // E - Green-Lagrange strain tensor
    //     E = 1/2 (grad_u + grad_u^T + grad_u^T*grad_u)
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    CeedScalar E2work[6];
    for (CeedInt m = 0; m < 6; m++) {
      E2work[m] = tempgradu[indj[m]][indk[m]] + tempgradu[indk[m]][indj[m]];
      for (CeedInt n = 0; n < 3; n++)
        E2work[m] += tempgradu[n][indj[m]]*tempgradu[n][indk[m]];
    }

    double J[6][6];
    for (int i=0; i<6; i++) for (int j=0; j<6; j++) J[i][j] = 0.;

    int size = getEnzymeSize((void *)computeS);
    for (int j=0; j<6; j++) tape[j][i] = malloc(size);

    CeedScalar Swork_[6];
    for (CeedInt j=0; j<6; j++) {
      double dSwork[6]  = {0., 0., 0., 0., 0., 0.}; dSwork[j] = 1.;
      grad_S_fwd(Swork_, dSwork, E2work, J[j], lambda, mu, tape[j][i]);
    }

    // *INDENT-OFF*
    const CeedScalar S[3][3] = {{Swork_[0], Swork_[5], Swork_[4]},
                                {Swork_[5], Swork_[1], Swork_[3]},
                                {Swork_[4], Swork_[3], Swork_[2]}
                               };
    // *INDENT-ON*

    // Save Swork
    Swork[0][i] = Swork_[0];
    Swork[1][i] = Swork_[1];
    Swork[2][i] = Swork_[2];
    Swork[3][i] = Swork_[3];
    Swork[4][i] = Swork_[4];
    Swork[5][i] = Swork_[5];

    // Compute the First Piola-Kirchhoff : P = F*S
    CeedScalar P[3][3];
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++) {
        P[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++)
          P[j][k] += F[j][m] * S[m][k];
      }

    // Apply dXdx^T and weight to P (First Piola-Kirchhoff)
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        dvdX[k][j][i] = 0;
        for (CeedInt m = 0; m < 3; m++)
          dvdX[k][j][i] += dXdx[k][m] * P[j][m] * wdetJ;
      }

  } // End of Quadrature Point Loop

  return 0;
}

#include <assert.h>

// -----------------------------------------------------------------------------
// Jacobian evaluation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSInitialNH1dF_AD)(void *ctx, CeedInt Q,
                                      const CeedScalar *const *in,
                                      CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*deltaug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
                   (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  // grad_u is used for hyperelasticity (non-linear)
  const CeedScalar (*grad_u)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[2];
  const CeedScalar (*Swork)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  void *(*tape)[CEED_Q_VLA] = (void *(*)[CEED_Q_VLA])in[4];

  // Outputs
  CeedScalar (*deltadvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // Context
  const Physics context = (Physics)ctx;
  const CeedScalar E  = context->E;
  const CeedScalar nu = context->nu;

  // Constants
  const CeedScalar TwoMu = E / (1 + nu);
  const CeedScalar mu = TwoMu / 2;
  const CeedScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  const CeedScalar lambda = (3*Kbulk - TwoMu) / 3;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of delta_u
    // *INDENT-OFF*
    const CeedScalar deltadu[3][3] = {{deltaug[0][0][i],
                                       deltaug[1][0][i],
                                       deltaug[2][0][i]},
                                      {deltaug[0][1][i],
                                       deltaug[1][1][i],
                                       deltaug[2][1][i]},
                                      {deltaug[0][2][i],
                                       deltaug[1][2][i],
                                       deltaug[2][2][i]}
                                     };
    // -- Qdata
    const CeedScalar wdetJ      =      q_data[0][i];
    const CeedScalar dXdx[3][3] =    {{q_data[1][i],
                                       q_data[2][i],
                                       q_data[3][i]},
                                      {q_data[4][i],
                                       q_data[5][i],
                                       q_data[6][i]},
                                      {q_data[7][i],
                                       q_data[8][i],
                                       q_data[9][i]}
                                      };
    // *INDENT-ON*

    // Compute graddeltau
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to deltadu = graddelta
    CeedScalar graddeltau[3][3];
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        graddeltau[j][k] = 0;
        for (CeedInt m =0 ; m < 3; m++)
          graddeltau[j][k] += dXdx[m][k] * deltadu[j][m];
      }

    // I3 : 3x3 Identity matrix
    // Deformation Gradient : F = I3 + grad_u
    // *INDENT-OFF*
    const CeedScalar F[3][3] =  {{grad_u[0][0][i] + 1,
                                  grad_u[0][1][i],
                                  grad_u[0][2][i]},
                                 {grad_u[1][0][i],
                                  grad_u[1][1][i] + 1,
                                  grad_u[1][2][i]},
                                 {grad_u[2][0][i],
                                  grad_u[2][1][i],
                                  grad_u[2][2][i] + 1}
                                };

    const CeedScalar tempgradu[3][3] =  {{grad_u[0][0][i],
                                          grad_u[0][1][i],
                                          grad_u[0][2][i]},
                                         {grad_u[1][0][i],
                                          grad_u[1][1][i],
                                          grad_u[1][2][i]},
                                         {grad_u[2][0][i],
                                          grad_u[2][1][i],
                                          grad_u[2][2][i]}
                                        };
    // *INDENT-ON*

    // deltaE - Green-Lagrange strain tensor
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    CeedScalar deltaEwork[6];
    for (CeedInt m = 0; m < 6; m++) {
      deltaEwork[m] = 0;
      for (CeedInt n = 0; n < 3; n++)
        deltaEwork[m] += (graddeltau[n][indj[m]]*F[n][indk[m]] +
                          F[n][indj[m]]*graddeltau[n][indk[m]])/2.;
    }

    // E - Green-Lagrange strain tensor
    //     E = 1/2 (grad_u + grad_u^T + grad_u^T*grad_u)
    CeedScalar E2work[6];
    for (CeedInt m = 0; m < 6; m++) {
      E2work[m] = tempgradu[indj[m]][indk[m]] + tempgradu[indk[m]][indj[m]];
      for (CeedInt n = 0; n < 3; n++)
        E2work[m] += tempgradu[n][indj[m]]*tempgradu[n][indk[m]];
    }

    CeedScalar Swork_[6];
    Swork_[0] = Swork[0][i];
    Swork_[1] = Swork[1][i];
    Swork_[2] = Swork[2][i];
    Swork_[3] = Swork[3][i];
    Swork_[4] = Swork[4][i];
    Swork_[5] = Swork[5][i];

    // J = \partial Swork / \partial E2work
    CeedScalar J[6][6];
    for (CeedInt i=0; i<6; i++) for (CeedInt j=0; j<6; j++) J[i][j] = 0.;

    //CeedScalar Swork[6];
    for (CeedInt j=0; j<6; j++) {
      double dSwork[6]  = {0., 0., 0., 0., 0., 0.}; dSwork[j] = 1.;
      assert(tape[j][i] != 0xDEADBEEF);
      grad_S_rev(Swork_, dSwork, E2work, J[j], lambda, mu, tape[j][i]);
      free(tape[j][i]);
      tape[j][i] = 0xDEADBEEF;
    }

    CeedScalar deltaSwork[6];
    for (CeedInt i=0; i<6; i++) {
      deltaSwork[i] = 0;
      for (CeedInt j=0; j<6; j++) {
        // deltaSwork = 2 (\partial Swork / \partial E2work) * deltaEwork
        deltaSwork[i] += 2. * J[i][j] * deltaEwork[j];
      }
    }

    // *INDENT-OFF*

    const CeedScalar deltaS[3][3] = {{deltaSwork[0], deltaSwork[5], deltaSwork[4]},
                                     {deltaSwork[5], deltaSwork[1], deltaSwork[3]},
                                     {deltaSwork[4], deltaSwork[3], deltaSwork[2]}
                                    };

    const CeedScalar S[3][3] = {{Swork_[0], Swork_[5], Swork_[4]},
                                {Swork_[5], Swork_[1], Swork_[3]},
                                {Swork_[4], Swork_[3], Swork_[2]}
                               };
    // *INDENT-ON*

    // deltaP = dPdF:deltaF = deltaF*S + F*deltaS
    CeedScalar deltaP[3][3];
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++) {
        deltaP[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++)
          deltaP[j][k] += graddeltau[j][m]*S[m][k] + F[j][m]*deltaS[m][k];
      }

    // Apply dXdx^T and weight
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        deltadvdX[k][j][i] = 0;
        for (CeedInt m = 0; m < 3; m++)
          deltadvdX[k][j][i] += dXdx[k][m] * deltaP[j][m] * wdetJ;
      }

  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Strain energy computation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSInitialNH1Energy_AD)(void *ctx, CeedInt Q,
    const CeedScalar *const *in,
    CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
                   (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  // Outputs
  CeedScalar (*energy) = (CeedScalar(*))out[0];
  // *INDENT-ON*

  // Context
  const Physics context = (Physics)ctx;
  const CeedScalar E  = context->E;
  const CeedScalar nu = context->nu;
  const CeedScalar TwoMu = E / (1 + nu);
  const CeedScalar mu = TwoMu / 2;
  const CeedScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  const CeedScalar lambda = (3*Kbulk - TwoMu) / 3;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of u
    // *INDENT-OFF*
    const CeedScalar du[3][3]   = {{ug[0][0][i],
                                    ug[1][0][i],
                                    ug[2][0][i]},
                                   {ug[0][1][i],
                                    ug[1][1][i],
                                    ug[2][1][i]},
                                   {ug[0][2][i],
                                    ug[1][2][i],
                                    ug[2][2][i]}
                                  };
    // -- Qdata
    const CeedScalar wdetJ      =   q_data[0][i];
    const CeedScalar dXdx[3][3] = {{q_data[1][i],
                                    q_data[2][i],
                                    q_data[3][i]},
                                   {q_data[4][i],
                                    q_data[5][i],
                                    q_data[6][i]},
                                   {q_data[7][i],
                                    q_data[8][i],
                                    q_data[9][i]}
                                  };
    // *INDENT-ON*

    // Compute grad_u
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to du = grad_u
    CeedScalar grad_u[3][3];
    for (int j = 0; j < 3; j++)     // Component
      for (int k = 0; k < 3; k++) { // Derivative
        grad_u[j][k] = 0;
        for (int m = 0; m < 3; m++)
          grad_u[j][k] += dXdx[m][k] * du[j][m];
      }

    // E - Green-Lagrange strain tensor
    //     E = 1/2 (grad_u + grad_u^T + grad_u^T*grad_u)
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    CeedScalar E2work[6];
    for (CeedInt m = 0; m < 6; m++) {
      E2work[m] = grad_u[indj[m]][indk[m]] + grad_u[indk[m]][indj[m]];
      for (CeedInt n = 0; n < 3; n++)
        E2work[m] += grad_u[n][indj[m]]*grad_u[n][indk[m]];
    }
    // *INDENT-OFF*
    CeedScalar E2[3][3] = {{E2work[0], E2work[5], E2work[4]},
                           {E2work[5], E2work[1], E2work[3]},
                           {E2work[4], E2work[3], E2work[2]}
                          };
    // *INDENT-ON*
    const CeedScalar Jm1 = computeJM1(grad_u);
    const CeedScalar logJ = log1p_series_shifted(Jm1);

    // Strain energy Phi(E) for compressible Neo-Hookean
    energy[i] = (lambda*logJ*logJ/2. - mu*logJ +
                 mu*(E2[0][0] + E2[1][1] + E2[2][2])/2.) * wdetJ;

  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Nodal diagnostic quantities for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSInitialNH1Diagnostic_AD)(void *ctx, CeedInt Q,
    const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1],
                   (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar (*diagnostic)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // Context
  const Physics context = (Physics)ctx;
  const CeedScalar E  = context->E;
  const CeedScalar nu = context->nu;
  const CeedScalar TwoMu = E / (1 + nu);
  const CeedScalar mu = TwoMu / 2;
  const CeedScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  const CeedScalar lambda = (3*Kbulk - TwoMu) / 3;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of u
    // *INDENT-OFF*
    const CeedScalar du[3][3]   = {{ug[0][0][i],
                                    ug[1][0][i],
                                    ug[2][0][i]},
                                   {ug[0][1][i],
                                    ug[1][1][i],
                                    ug[2][1][i]},
                                   {ug[0][2][i],
                                    ug[1][2][i],
                                    ug[2][2][i]}
                                  };
    // -- Qdata
    const CeedScalar dXdx[3][3] = {{q_data[1][i],
                                    q_data[2][i],
                                    q_data[3][i]},
                                   {q_data[4][i],
                                    q_data[5][i],
                                    q_data[6][i]},
                                   {q_data[7][i],
                                    q_data[8][i],
                                    q_data[9][i]}
                                  };
    // *INDENT-ON*

    // Compute grad_u
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to du = grad_u
    CeedScalar grad_u[3][3];
    for (int j = 0; j < 3; j++)     // Component
      for (int k = 0; k < 3; k++) { // Derivative
        grad_u[j][k] = 0;
        for (int m = 0; m < 3; m++)
          grad_u[j][k] += dXdx[m][k] * du[j][m];
      }

    // E - Green-Lagrange strain tensor
    //     E = 1/2 (grad_u + grad_u^T + grad_u^T*grad_u)
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    CeedScalar E2work[6];
    for (CeedInt m = 0; m < 6; m++) {
      E2work[m] = grad_u[indj[m]][indk[m]] + grad_u[indk[m]][indj[m]];
      for (CeedInt n = 0; n < 3; n++)
        E2work[m] += grad_u[n][indj[m]]*grad_u[n][indk[m]];
    }
    // *INDENT-OFF*
    CeedScalar E2[3][3] = {{E2work[0], E2work[5], E2work[4]},
                           {E2work[5], E2work[1], E2work[3]},
                           {E2work[4], E2work[3], E2work[2]}
                          };
    // *INDENT-ON*

    // Displacement
    diagnostic[0][i] = u[0][i];
    diagnostic[1][i] = u[1][i];
    diagnostic[2][i] = u[2][i];

    // Pressure
    const CeedScalar Jm1 = computeJM1(grad_u);
    const CeedScalar logJ = log1p_series_shifted(Jm1);
    diagnostic[3][i] = -lambda*logJ;

    // Stress tensor invariants
    diagnostic[4][i] = (E2[0][0] + E2[1][1] + E2[2][2]) / 2.;
    diagnostic[5][i] = 0.;
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt m = 0; m < 3; m++)
        diagnostic[5][i] += E2[j][m] * E2[m][j] / 4.;
    diagnostic[6][i] = Jm1 + 1.;

    // Strain energy
    diagnostic[7][i] = (lambda*logJ*logJ/2. - mu*logJ +
                        mu*(E2[0][0] + E2[1][1] + E2[2][2])/2.);

  } // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif // End of ELAS_FSInitialNH1_H
