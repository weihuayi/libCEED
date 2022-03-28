/// @file
/// Test div in multiple dimensions, NOTRANSPOSE case
/// \test Test div in multiple dimensions, NOTRANSPOSE case
#include <ceed.h>
#include <math.h>

static int Eval(CeedInt dim, const CeedScalar x[], CeedScalar *val) {

  if (dim == 1) {
    val[0] = 1+x[0]+x[0]*x[0];
  } else if (dim == 2) {
    val[0] = x[0]+x[0]*x[0];
    val[1] = 2*x[1]+x[1]*x[1];
  } else {
    val[0] = 1+x[0]+x[0]*x[0];
    val[1] = 1+x[1]+x[1]*x[1];
    val[2] = 1+x[2]+x[2]*x[2];
  }

  return 0;
}

static int Eval_div(CeedInt dim, const CeedScalar x[], CeedScalar *div) {

  if (dim == 1) {
    div[0] = 1+2*x[0];
  } else if (dim == 2) {
    div[0] = 1+2*x[0] + 2+2*x[1];
  } else {
    div[0] = 1+2*x[0] + 1+2*x[1] + 1+2*x[2];
  }

  return 0;
}

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);
  for (CeedInt dim=1; dim<=3; dim++) {
    CeedVector X, X_q, U, DU_q;
    CeedBasis basis_x_lobatto, basis_x_gauss, basis_u_gauss;
    CeedInt Q = 3, Q_dim = CeedIntPow(Q, dim), X_dim = CeedIntPow(2, dim);
    CeedScalar x[X_dim*dim];
    const CeedScalar *xq, *Du;
    CeedScalar u[dim*Q_dim];

    for (CeedInt d=0; d<dim; d++)
      for (CeedInt i=0; i<X_dim; i++)
        x[d*X_dim + i] = (i % CeedIntPow(2, dim-d)) / CeedIntPow(2, dim-d-1) ? 1 : -1;

    CeedVectorCreate(ceed, X_dim*dim, &X);
    CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);
    CeedVectorCreate(ceed, Q_dim*dim, &X_q);
    CeedVectorSetValue(X_q, 0);
    CeedVectorCreate(ceed, dim*Q_dim, &U);

    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, Q, CEED_GAUSS_LOBATTO,
                                    &basis_x_lobatto);

    CeedBasisApply(basis_x_lobatto, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X, X_q);

    CeedVectorGetArrayRead(X_q, CEED_MEM_HOST, &xq);
    for (CeedInt i=0; i<Q_dim; i++) {
      CeedScalar xx[dim], val[dim];
      for (CeedInt d=0; d<dim; d++) {
        xx[d] = xq[d*Q_dim + i];
      }
      // Compute exact value at Gauss quadrature pts
      Eval(dim, xx, val);
      if (dim == 1) {
        u[i] = val[0];
      }
      if (dim == 2) {
        u[i] = val[0];
        u[i+Q_dim] = val[1];
      }
      if (dim == 3) {
        u[i] = val[0];
        u[i+Q_dim] = val[1];
        u[i+2*Q_dim] = val[2];
      }
    }
    CeedVectorRestoreArrayRead(X_q, &xq);
    CeedVectorSetArray(U, CEED_MEM_HOST, CEED_USE_POINTER, u);

    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, Q, CEED_GAUSS,
                                    &basis_x_gauss);
    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, Q, Q, CEED_GAUSS,
                                    &basis_u_gauss);

    CeedBasisApply(basis_x_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X, X_q);

    CeedVectorCreate(ceed, Q_dim, &DU_q);
    // Compute numerical div at Gauss quadrature pts (X_q)
    CeedBasisApply(basis_u_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_DIV, U, DU_q);

    CeedVectorGetArrayRead(X_q, CEED_MEM_HOST, &xq);
    CeedVectorGetArrayRead(DU_q, CEED_MEM_HOST, &Du);
    // Grad uexact
    CeedScalar D_ue[Q_dim];
    for (CeedInt i=0; i<Q_dim; i++) {
      CeedScalar xx[dim], div[1];
      for (CeedInt d=0; d<dim; d++) {
        xx[d] = xq[d*Q_dim + i];
      }
      // Compute exact divergence at Gauss quadrature pts
      Eval_div(dim, xx, div);
      D_ue[i] = div[0];
    }
    // Check if numerical and exact divergence are the same at quadrature pts
    for (CeedInt i=0; i<Q_dim; i++) {
      if (fabs(Du[i] - D_ue[i]) > 100.*CEED_EPSILON)
        // LCOV_EXCL_START
        printf("%f != %f\n", Du[i], D_ue[i]);
      // LCOV_EXCL_STOP
    }
    CeedVectorRestoreArrayRead(X_q, &xq);
    CeedVectorRestoreArrayRead(DU_q, &Du);

    CeedVectorDestroy(&X);
    CeedVectorDestroy(&X_q);
    CeedVectorDestroy(&U);
    CeedVectorDestroy(&DU_q);
    CeedBasisDestroy(&basis_x_lobatto);
    CeedBasisDestroy(&basis_x_gauss);
    CeedBasisDestroy(&basis_u_gauss);
  }
  CeedDestroy(&ceed);
  return 0;
}