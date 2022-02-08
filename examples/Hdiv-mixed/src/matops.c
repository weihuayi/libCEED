#include "../include/matops.h"
#include "../include/setup-libceed.h"

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the Laplacian with
// Dirichlet boundary conditions
// -----------------------------------------------------------------------------
PetscErrorCode ApplyLocal_Ceed(User user, Vec X, Vec Y) {
  PetscErrorCode ierr;
  PetscScalar *x, *y;
  PetscMemType x_mem_type, y_mem_type;

  PetscFunctionBeginUser;

  // Global-to-local
  ierr = DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->X_loc); CHKERRQ(ierr);

  // Setup libCEED vectors
  ierr = VecGetArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x,
                                   &x_mem_type); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(user->Y_loc, &y, &y_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(user->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);
  CeedVectorSetArray(user->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);

  // Apply libCEED operator
  CeedOperatorApply(user->op_apply, user->x_ceed, user->y_ceed,
                    CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->x_ceed, MemTypeP2C(x_mem_type), NULL);
  CeedVectorTakeArray(user->y_ceed, MemTypeP2C(y_mem_type), NULL);
  ierr = VecRestoreArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(user->Y_loc, &y); CHKERRQ(ierr);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, user->Y_loc, ADD_VALUES, Y); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function wraps the libCEED operator for a MatShell
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  User user;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // libCEED for local action of residual evaluator
  ierr = ApplyLocal_Ceed(user, X, Y); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function calculates the error in the final solution
// -----------------------------------------------------------------------------
PetscErrorCode ComputeError(User user, Vec X, CeedVector target,
                            CeedScalar *l2_error_u,
                            CeedScalar *l2_error_p) {
  PetscErrorCode ierr;
  PetscScalar *x;
  PetscMemType mem_type;
  CeedVector collocated_error;
  CeedSize length;

  PetscFunctionBeginUser;
  CeedVectorGetLength(target, &length);
  CeedVectorCreate(user->ceed, length, &collocated_error);

  // Global-to-local
  ierr = DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->X_loc); CHKERRQ(ierr);

  // Setup CEED vector
  ierr = VecGetArrayAndMemType(user->X_loc, &x, &mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(user->x_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, x);

  // Apply CEED operator
  CeedOperatorApply(user->op_error, user->x_ceed, collocated_error,
                    CEED_REQUEST_IMMEDIATE);
  // Restore PETSc vector
  CeedVectorTakeArray(user->x_ceed, MemTypeP2C(mem_type), NULL);
  ierr = VecRestoreArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x);
  CHKERRQ(ierr);

  // Compute L2 error for each field
  CeedInt c_start, c_end, dim, num_elem, num_qpts;
  ierr = DMGetDimension(user->dm, &dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(user->dm, 0, &c_start, &c_end); CHKERRQ(ierr);
  num_elem = c_end -c_start;
  num_qpts = length / (num_elem*(dim+1));
  CeedInt cent_qpts = num_qpts / 2;
  CeedVector collocated_error_u, collocated_error_p;
  const CeedScalar *E_U; // to store total error
  CeedInt length_u, length_p;
  length_p = num_elem;
  length_u = num_elem*num_qpts*dim;
  CeedScalar e_u[length_u], e_p[length_p];
  CeedVectorCreate(user->ceed, length_p, &collocated_error_p);
  CeedVectorCreate(user->ceed, length_u, &collocated_error_u);
  // E_U is ordered as [p_0,u_0/.../p_n,u_n] for 0 to n num_elem
  // For each element p_0 size is num_qpts, and u_0 is dim*num_qpts
  CeedVectorGetArrayRead(collocated_error, CEED_MEM_HOST, &E_U);
  for (CeedInt n=0; n < num_elem; n++) {
    for (CeedInt i=0; i < 1; i++) {
      CeedInt j = i + n*1;
      CeedInt k = cent_qpts + n*num_qpts*(dim+1);
      e_p[j] = E_U[k];
    }
  }

  for (CeedInt n=0; n < num_elem; n++) {
    for (CeedInt i=0; i < dim*num_qpts; i++) {
      CeedInt j = i + n*num_qpts*dim;
      CeedInt k = num_qpts + i + n*num_qpts*(dim+1);
      e_u[j] = E_U[k];
    }
  }

  CeedVectorSetArray(collocated_error_p, CEED_MEM_HOST, CEED_USE_POINTER, e_p);
  CeedVectorSetArray(collocated_error_u, CEED_MEM_HOST, CEED_USE_POINTER, e_u);
  CeedVectorRestoreArrayRead(collocated_error, &E_U);

  CeedScalar error_u, error_p;
  CeedVectorNorm(collocated_error_u, CEED_NORM_1, &error_u);
  CeedVectorNorm(collocated_error_p, CEED_NORM_1, &error_p);
  *l2_error_u = sqrt(error_u);
  *l2_error_p = sqrt(error_p);
  // Cleanup
  CeedVectorDestroy(&collocated_error);
  CeedVectorDestroy(&collocated_error_u);
  CeedVectorDestroy(&collocated_error_p);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
