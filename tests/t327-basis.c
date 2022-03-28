/// @file
/// Test div in multiple dimensions, TRANSPOSE case
/// \test Test div in multiple dimensions, TRANSPOSE case
#include <ceed.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);
  for (CeedInt dim=1; dim<=3; dim++) {
    CeedVector U, DU;
    CeedBasis basis_u_gauss;
    CeedInt P = 2, Q = 3, num_comp = dim,
            Q_dim = CeedIntPow(Q, dim), P_dim = CeedIntPow(P, dim);
    const CeedScalar *du;

    CeedVectorCreate(ceed, dim*P_dim, &DU);
    CeedVectorCreate(ceed, dim*Q_dim, &U);
    CeedVectorSetValue(U, 1.0);

    CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp, P, Q, CEED_GAUSS,
                                    &basis_u_gauss);

    CeedBasisApply(basis_u_gauss, 1, CEED_TRANSPOSE, CEED_EVAL_DIV, U, DU);

    CeedVectorGetArrayRead(DU, CEED_MEM_HOST, &du);
    CeedScalar sum = 0.;
    for (CeedInt i = 0; i<dim*P_dim; i++) {
      sum += du[i];
    }
    if (fabs(sum) > 100.*CEED_EPSILON)
      // LCOV_EXCL_START
      printf("sum of array %f != %f\n", sum, 0.0);
    // LCOV_EXCL_STOP
    CeedVectorRestoreArrayRead(DU, &du);

    CeedVectorDestroy(&U);
    CeedVectorDestroy(&DU);
    CeedBasisDestroy(&basis_u_gauss);
  }
  CeedDestroy(&ceed);
  return 0;
}