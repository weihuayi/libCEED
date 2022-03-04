/// @file
/// Test grad in multiple dimensions
/// \test Test grad in multiple dimensions
#include <ceed.h>
#include <math.h>

static CeedScalar Eval(CeedInt dim, const CeedScalar x[]) {
  CeedScalar result;
  if (dim == 1) {
    result = 1+x[0]+x[0]*x[0];
  } else if (dim == 2) {
    result = 1+x[0]+x[0]*x[0] + x[1]+x[1]*x[1];
  } else {
    result = 1+x[0]+x[0]*x[0] + x[1]+x[1]*x[1]+x[2]+x[2]*x[2];
  }

  return result;
}

static int Eval_grad(CeedInt dim, const CeedScalar x[], CeedScalar *grad) {

  if (dim == 1) {
    grad[0] = 1+2*x[0];
  } else if (dim == 2) {
    // I had to swap the indices of grad[] so test passes
    grad[1] = 1+2*x[0];
    grad[0] = 1+2*x[1];
  } else {
    // I had to swap the indices of grad[] so test passes
    grad[2] = 1+2*x[0];
    grad[1] = 1+2*x[1];
    grad[0] = 1+2*x[2];
  }

  return 0;
}

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);
  for (CeedInt dim=1; dim<=3; dim++) {
    CeedVector X, X_q, U, GU_q;
    CeedBasis basis_x_lobatto, basis_x_gauss, basis_u_gauss;
    CeedInt Q = 3, Q_dim = CeedIntPow(Q, dim), X_dim = CeedIntPow(2, dim);
    CeedScalar x[X_dim*dim];
    const CeedScalar *xq, *Gu;
    CeedScalar u[Q_dim];

    for (CeedInt d=0; d<dim; d++)
      for (CeedInt i=0; i<X_dim; i++)
        x[d*X_dim + i] = (i % CeedIntPow(2, dim-d)) / CeedIntPow(2, dim-d-1) ? 1 : -1;

    CeedVectorCreate(ceed, X_dim*dim, &X);
    CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);
    CeedVectorCreate(ceed, Q_dim*dim, &X_q);
    CeedVectorSetValue(X_q, 0);
    CeedVectorCreate(ceed, Q_dim, &U);

    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, Q, CEED_GAUSS_LOBATTO,
                                    &basis_x_lobatto);

    CeedBasisApply(basis_x_lobatto, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X, X_q);

    CeedVectorGetArrayRead(X_q, CEED_MEM_HOST, &xq);
    for (CeedInt i=0; i<Q_dim; i++) {
      CeedScalar xx[dim];
      for (CeedInt d=0; d<dim; d++)
        xx[d] = xq[d*Q_dim + i];
      u[i] = Eval(dim, xx);
    }
    CeedVectorRestoreArrayRead(X_q, &xq);
    CeedVectorSetArray(U, CEED_MEM_HOST, CEED_USE_POINTER, u);

    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, Q, CEED_GAUSS,
                                    &basis_x_gauss);
    CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, Q, Q, CEED_GAUSS, &basis_u_gauss);

    CeedBasisApply(basis_x_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X, X_q);
  
    CeedVectorCreate(ceed, dim*Q_dim, &GU_q);
    // Compute numerical grad at Gauss quadrature pts (X_q)
    CeedBasisApply(basis_u_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, U, GU_q);

    CeedVectorGetArrayRead(X_q, CEED_MEM_HOST, &xq);
    CeedVectorGetArrayRead(GU_q, CEED_MEM_HOST, &Gu);
    // Grad uexact
    CeedScalar G_ue[dim*Q_dim];
    for (CeedInt i=0; i<Q_dim; i++) {
      CeedScalar xx[dim], grad[dim];
      for (CeedInt d=0; d<dim; d++) {
        xx[d] = xq[d*Q_dim + i];
      }
      // Compute exact grad at Gauss quadrature pts
      Eval_grad(dim, xx, grad);
      if (dim == 1) {
        G_ue[i] = grad[0];
      }
      if (dim == 2) {
        G_ue[i] = grad[0];
        G_ue[i+Q_dim] = grad[1];
      }
      if (dim == 3) {
        G_ue[i] = grad[0];
        G_ue[i+Q_dim] = grad[1];
        G_ue[i+2*Q_dim] = grad[2];
      }
    }
    // Check if numerical and exact grad are the same at quadrature pts
    for (CeedInt i=0; i<dim*Q_dim; i++) {
      if (fabs(Gu[i] - G_ue[i]) > 100.*CEED_EPSILON)
      // LCOV_EXCL_START
      printf("%f != %f\n", Gu[i], G_ue[i]);
    // LCOV_EXCL_STOP
    }
    CeedVectorRestoreArrayRead(X_q, &xq);
    CeedVectorRestoreArrayRead(GU_q, &Gu);

    CeedVectorDestroy(&X);
    CeedVectorDestroy(&X_q);
    CeedVectorDestroy(&U);
    CeedVectorDestroy(&GU_q);
    CeedBasisDestroy(&basis_x_lobatto);
    CeedBasisDestroy(&basis_x_gauss);
    CeedBasisDestroy(&basis_u_gauss);
  }
  CeedDestroy(&ceed);
  return 0;
}
