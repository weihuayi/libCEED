#include "../include/matops.h"
#include "../include/petscutils.h"

// -----------------------------------------------------------------------------
// This function returns the computed diagonal of the operator
// -----------------------------------------------------------------------------
PetscErrorCode MatGetDiag(Mat A, Vec D) {
  PetscErrorCode ierr;
  UserO user;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Compute Diagonal via libCEED
  PetscScalar *x;
  PetscMemType mem_type;

  // -- Place PETSc vector in libCEED vector
  ierr = VecGetArrayAndMemType(user->X_loc, &x, &mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(user->x_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, x);

  // -- Compute Diagonal
  CeedOperatorLinearAssembleDiagonal(user->op, user->x_ceed,
                                     CEED_REQUEST_IMMEDIATE);

  // -- Local-to-Global
  CeedVectorTakeArray(user->x_ceed, MemTypeP2C(mem_type), &x);
  ierr = VecRestoreArrayAndMemType(user->X_loc, &x); CHKERRQ(ierr);
  ierr = VecZeroEntries(D); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, user->X_loc, ADD_VALUES, D); CHKERRQ(ierr);

  // Cleanup
  ierr = VecZeroEntries(user->X_loc); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the Laplacian with
// Dirichlet boundary conditions
// -----------------------------------------------------------------------------
PetscErrorCode ApplyLocal_Ceed(Vec X, Vec Y, UserO user) {
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
  CeedOperatorApply(user->op, user->x_ceed, user->y_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->x_ceed, MemTypeP2C(x_mem_type), &x);
  CeedVectorTakeArray(user->y_ceed, MemTypeP2C(y_mem_type), &y);
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
  UserO user;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // libCEED for local action of residual evaluator
  ierr = ApplyLocal_Ceed(X, Y, user); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function wraps the libCEED operator for a SNES residual evaluation
// -----------------------------------------------------------------------------
PetscErrorCode FormResidual_Ceed(SNES snes, Vec X, Vec Y, void *ctx) {
  PetscErrorCode ierr;
  UserO user = (UserO)ctx;

  PetscFunctionBeginUser;

  // libCEED for local action of residual evaluator
  ierr = ApplyLocal_Ceed(X, Y, user); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the prolongation operator
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Prolong(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  UserProlongRestr user;
  PetscScalar *c, *f;
  PetscMemType c_mem_type, f_mem_type;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Global-to-local
  ierr = VecZeroEntries(user->loc_vec_c); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dmc, X, INSERT_VALUES, user->loc_vec_c);
  CHKERRQ(ierr);

  // Setup libCEED vectors
  ierr = VecGetArrayReadAndMemType(user->loc_vec_c, (const PetscScalar **)&c,
                                   &c_mem_type); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(user->loc_vec_f, &f, &f_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(user->ceed_vec_c, MemTypeP2C(c_mem_type), CEED_USE_POINTER,
                     c);
  CeedVectorSetArray(user->ceed_vec_f, MemTypeP2C(f_mem_type), CEED_USE_POINTER,
                     f);

  // Apply libCEED operator
  CeedOperatorApply(user->op_prolong, user->ceed_vec_c, user->ceed_vec_f,
                    CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->ceed_vec_c, MemTypeP2C(c_mem_type), &c);
  CeedVectorTakeArray(user->ceed_vec_f, MemTypeP2C(f_mem_type), &f);
  ierr = VecRestoreArrayReadAndMemType(user->loc_vec_c, (const PetscScalar **)&c);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(user->loc_vec_f, &f); CHKERRQ(ierr);

  // Multiplicity
  ierr = VecPointwiseMult(user->loc_vec_f, user->loc_vec_f, user->mult_vec);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dmf, user->loc_vec_f, ADD_VALUES, Y);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the restriction operator
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_Restrict(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  UserProlongRestr user;
  PetscScalar *c, *f;
  PetscMemType c_mem_type, f_mem_type;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Global-to-local
  ierr = VecZeroEntries(user->loc_vec_f); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dmf, X, INSERT_VALUES, user->loc_vec_f);
  CHKERRQ(ierr);

  // Multiplicity
  ierr = VecPointwiseMult(user->loc_vec_f, user->loc_vec_f, user->mult_vec);
  CHKERRQ(ierr);

  // Setup libCEED vectors
  ierr = VecGetArrayReadAndMemType(user->loc_vec_f, (const PetscScalar **)&f,
                                   &f_mem_type); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(user->loc_vec_c, &c, &c_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(user->ceed_vec_f, MemTypeP2C(f_mem_type), CEED_USE_POINTER,
                     f);
  CeedVectorSetArray(user->ceed_vec_c, MemTypeP2C(c_mem_type), CEED_USE_POINTER,
                     c);

  // Apply CEED operator
  CeedOperatorApply(user->op_restrict, user->ceed_vec_f, user->ceed_vec_c,
                    CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->ceed_vec_c, MemTypeP2C(c_mem_type), &c);
  CeedVectorTakeArray(user->ceed_vec_f, MemTypeP2C(f_mem_type), &f);
  ierr = VecRestoreArrayReadAndMemType(user->loc_vec_f, (const PetscScalar **)&f);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(user->loc_vec_c, &c); CHKERRQ(ierr);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dmc, user->loc_vec_c, ADD_VALUES, Y);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function sets up the BDDC preconditioner
// -----------------------------------------------------------------------------
PetscErrorCode PCShellSetup_BDDC(PC pc) {
  int ierr;
  UserBDDC user;

  PetscFunctionBeginUser;

  ierr = PCShellGetContext(pc, (void *)&user); CHKERRQ(ierr);

  // Assemble mat for element Schur AMG
  {
    ierr = VecZeroEntries(user->X_Pi_r_loc); CHKERRQ(ierr);
    ierr = SNESComputeJacobianDefaultColor(user->snes_Pi_r, user->X_Pi_r_loc,
                                           user->mat_S_Pi_r, user->mat_S_Pi_r, NULL);
    CHKERRQ(ierr);
    ierr = MatAssemblyBegin(user->mat_S_Pi_r, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(user->mat_S_Pi_r, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }

  // Assemble mat for Schur AMG
  {
    ierr = VecZeroEntries(user->X_Pi); CHKERRQ(ierr);
    ierr = SNESComputeJacobianDefaultColor(user->snes_Pi, user->X_Pi,
                                           user->mat_S_Pi, user->mat_S_Pi, NULL);
    CHKERRQ(ierr);
    ierr = MatAssemblyBegin(user->mat_S_Pi, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(user->mat_S_Pi, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function provides the action of the element Schur compliment
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_BDDCElementSchur(UserBDDC user, Vec X_Pi_r_loc,
                                        Vec Y_Pi_r_loc) {
  PetscErrorCode ierr;
  CeedDataBDDC data = user->ceed_data_bddc;
  PetscScalar *x, *y;
  PetscMemType x_mem_type, y_mem_type;

  PetscFunctionBeginUser;

  // Set arrays in libCEED
  ierr = VecGetArrayReadAndMemType(X_Pi_r_loc, (const PetscScalar **)&x,
                                   &x_mem_type); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(Y_Pi_r_loc, &y, &y_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(data->x_Pi_r_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER,
                     x);
  CeedVectorSetArray(data->y_Pi_r_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER,
                     y);

  // Apply action on Schur compliment
  // Y_Pi_r = -B A_r,r^-1 B^T X_Pi_r
  // -- X_r = B^T X_Pi_r
  CeedOperatorApply(data->op_inject_Pi_r, data->x_Pi_r_ceed, data->x_r_ceed,
                    CEED_REQUEST_IMMEDIATE);
  // -- Y_r = A_r,r^-1 X_r
  CeedOperatorApply(data->op_r_r_inv, data->x_r_ceed, data->y_r_ceed,
                    CEED_REQUEST_IMMEDIATE);
  // -- Y_Pi_r = -B Y_r
  CeedOperatorApply(data->op_restrict_Pi_r, data->y_r_ceed, data->y_Pi_r_ceed,
                    CEED_REQUEST_IMMEDIATE);
  CeedVectorScale(data->y_Pi_r_ceed, -1.0);

  // Restore arrays
  CeedVectorTakeArray(data->x_Pi_r_ceed, MemTypeP2C(x_mem_type), &x);
  CeedVectorTakeArray(data->y_Pi_r_ceed, MemTypeP2C(y_mem_type), &y);
  ierr = VecRestoreArrayReadAndMemType(X_Pi_r_loc, (const PetscScalar **)&x);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(Y_Pi_r_loc, &y); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function assembles the element Schur compliment for the dummy SNES
// -----------------------------------------------------------------------------
PetscErrorCode FormResidual_BDDCElementSchur(SNES snes, Vec X_Pi_r_loc,
    Vec Y_Pi_r_loc, void *ctx) {
  PetscErrorCode ierr;
  UserBDDC user = (UserBDDC)ctx;

  PetscFunctionBeginUser;

  ierr = MatMult_BDDCElementSchur(user, X_Pi_r_loc, Y_Pi_r_loc); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function provides the action of the element inverse
// -----------------------------------------------------------------------------
PetscErrorCode BDDCArrInv(UserBDDC user, CeedVector x_r_ceed,
                          CeedVector y_r_ceed) {
  PetscErrorCode ierr;
  CeedDataBDDC data = user->ceed_data_bddc;
  PetscScalar *x, *y;
  PetscMemType x_mem_type, y_mem_type;

  PetscFunctionBeginUser;

  // Y_r = A_r,r^-1 (I + B S^-1 B^T A_r,r^-1) X_r
  // -- X_r = (I + B S^-1 B^T A_r,r^-1) X_r
  // ---- Y_r = A_r,r^-1 X_r
  CeedVectorPointwiseMult(x_r_ceed, x_r_ceed, data->mask_r_ceed);
  CeedOperatorApply(data->op_r_r_inv, x_r_ceed, y_r_ceed, CEED_REQUEST_IMMEDIATE);
  // ---- Y_Pi_r = B^T Y_r
  ierr = VecGetArrayAndMemType(user->Y_Pi_r_loc, &y, &y_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(data->y_Pi_r_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER,
                     y);
  CeedOperatorApply(data->op_restrict_Pi_r, y_r_ceed, data->y_Pi_r_ceed,
                    CEED_REQUEST_IMMEDIATE);
  CeedVectorTakeArray(data->y_Pi_r_ceed, MemTypeP2C(y_mem_type), &y);
  ierr = VecRestoreArrayAndMemType(user->Y_Pi_r_loc, &y); CHKERRQ(ierr);
  // ---- X_Pi_r = S^-1 Y_Pi_r
  ierr = KSPSolve(user->ksp_S_Pi_r, user->Y_Pi_r_loc, user->X_Pi_r_loc);
  CHKERRQ(ierr);
  // ---- X_r += B X_Pi_r
  ierr = VecGetArrayReadAndMemType(user->X_Pi_r_loc, (const PetscScalar **)&x,
                                   &x_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(data->x_Pi_r_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER,
                     x);
  CeedOperatorApplyAdd(data->op_inject_Pi_r, data->x_Pi_r_ceed, x_r_ceed,
                       CEED_REQUEST_IMMEDIATE);
  CeedVectorTakeArray(data->x_Pi_r_ceed, MemTypeP2C(x_mem_type), &x);
  ierr = VecRestoreArrayReadAndMemType(user->X_Pi_r_loc,
                                       (const PetscScalar **)&x);
  CHKERRQ(ierr);
  // -- Y_r = A_r,r^-1 X_r
  CeedOperatorApply(data->op_r_r_inv, x_r_ceed, y_r_ceed, CEED_REQUEST_IMMEDIATE);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function provides the action of the Schur compliment
// -----------------------------------------------------------------------------
PetscErrorCode MatMult_BDDCSchur(UserBDDC user, Vec X_Pi, Vec Y_Pi) {
  PetscErrorCode ierr;
  CeedDataBDDC data = user->ceed_data_bddc;
  PetscScalar *x, *y;
  PetscMemType x_mem_type, y_mem_type;

  PetscFunctionBeginUser;

  // Global-to-Local
  ierr = VecZeroEntries(user->X_Pi_loc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm_Pi, X_Pi, INSERT_VALUES, user->X_Pi_loc);
  CHKERRQ(ierr);
  // Set arrays in libCEED
  ierr = VecGetArrayReadAndMemType(user->X_Pi_loc, (const PetscScalar **)&x,
                                   &x_mem_type); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(user->Y_Pi_loc, &y, &y_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(data->x_Pi_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER,
                     x);
  CeedVectorSetArray(data->y_Pi_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER,
                     y);

  // Apply action on Schur compliment
  // Y_Pi  = (A_Pi,Pi - A_Pi,r A_r,r^-1 A_r,Pi) X_Pi
  // -- X_r = A_r,Pi X_Pi
  CeedOperatorApply(data->op_r_Pi, data->x_Pi_ceed, data->x_r_ceed,
                    CEED_REQUEST_IMMEDIATE);
  // -- Y_r = A_r,r^-1 X_r
  ierr = BDDCArrInv(user, data->x_r_ceed, data->y_r_ceed); CHKERRQ(ierr);
  // -- Y_Pi = -A_Pi,r Y_r
  CeedVectorPointwiseMult(data->y_r_ceed, data->y_r_ceed, data->mask_r_ceed);
  CeedOperatorApply(data->op_Pi_r, data->y_r_ceed, data->y_Pi_ceed,
                    CEED_REQUEST_IMMEDIATE);
  CeedVectorScale(data->y_Pi_ceed, -1.0);
  // -- Y_Pi += A_Pi,Pi X_Pi
  CeedOperatorApplyAdd(data->op_Pi_Pi, data->x_Pi_ceed, data->y_Pi_ceed,
                       CEED_REQUEST_IMMEDIATE);

  // Restore arrays
  CeedVectorTakeArray(data->x_Pi_ceed, MemTypeP2C(x_mem_type), &x);
  CeedVectorTakeArray(data->y_Pi_ceed, MemTypeP2C(y_mem_type), &y);
  ierr = VecRestoreArrayReadAndMemType(user->X_Pi_loc, (const PetscScalar **)&x);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(user->Y_Pi_loc, &y); CHKERRQ(ierr);
  // Local-to-Global
  ierr = VecZeroEntries(Y_Pi); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm_Pi, user->Y_Pi_loc, ADD_VALUES, Y_Pi);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function assembles the Schur compliment for the dummy SNES
// -----------------------------------------------------------------------------
PetscErrorCode FormResidual_BDDCSchur(SNES snes, Vec X_Pi, Vec Y_Pi,
                                      void *ctx) {
  PetscErrorCode ierr;
  UserBDDC user = (UserBDDC)ctx;

  PetscFunctionBeginUser;

  ierr = MatMult_BDDCSchur(user, X_Pi, Y_Pi); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the action of the BDDC preconditioner
// -----------------------------------------------------------------------------
PetscErrorCode PCShellApply_BDDC(PC pc, Vec X, Vec Y) {
  PetscErrorCode ierr;
  UserBDDC user;
  CeedDataBDDC data;
  PetscScalar *x, *y;
  PetscMemType x_mem_type, y_mem_type;

  PetscFunctionBeginUser;

  ierr = PCShellGetContext(pc, (void *)&user); CHKERRQ(ierr);
  data = user->ceed_data_bddc;

  // Inject to broken space
  // -- Scaled injection, point multiply by 1/multiplicity
  // ---- Global-to-Local
  ierr = VecZeroEntries(user->X_loc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->X_loc);
  CHKERRQ(ierr);
  // ---- Inject to Y_r
  ierr = VecGetArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x,
                                   &x_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(data->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER,
                     x);
  CeedOperatorApply(data->op_inject_r, data->x_ceed, data->y_r_ceed,
                    CEED_REQUEST_IMMEDIATE);
  CeedVectorTakeArray(data->x_ceed, MemTypeP2C(x_mem_type), &x);
  ierr = VecRestoreArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x);
  CHKERRQ(ierr);
  // -- Harmonic injection, scaled with jump map
  if (user->is_harmonic) {
    CeedVectorPointwiseMult(data->x_r_ceed, data->y_r_ceed, data->mask_I_ceed);
    // ---- Z_r = A_I,I^-1 X_r
    ierr = BDDCArrInv(user, data->x_r_ceed, data->z_r_ceed); CHKERRQ(ierr);
    // ---- X_r = - A_Gamma,I Z_r
    CeedVectorPointwiseMult(data->z_r_ceed, data->z_r_ceed, data->mask_I_ceed);
    CeedOperatorApply(data->op_r_r, data->z_r_ceed, data->x_r_ceed,
                      CEED_REQUEST_IMMEDIATE);
    CeedVectorPointwiseMult(data->x_r_ceed, data->x_r_ceed, data->mask_Gamma_ceed);
    // ---- J^T (jump map)
    CeedVectorPointwiseMult(data->z_r_ceed, data->x_r_ceed, data->mult_ceed);
    // ------ Local-to-Global
    ierr = VecGetArrayAndMemType(user->Y_loc, &y, &y_mem_type); CHKERRQ(ierr);
    CeedVectorSetArray(data->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER,
                       y);
    CeedOperatorApply(data->op_restrict_r, data->z_r_ceed, data->y_ceed,
                      CEED_REQUEST_IMMEDIATE);
    CeedVectorTakeArray(data->y_ceed, MemTypeP2C(y_mem_type), &y);
    ierr = VecRestoreArrayAndMemType(user->Y_loc, &y); CHKERRQ(ierr);
    ierr = VecZeroEntries(Y); CHKERRQ(ierr);
    ierr = DMLocalToGlobal(user->dm, user->Y_loc, ADD_VALUES, Y);
    CHKERRQ(ierr);
    // ------ Global-to-Local
    ierr = VecZeroEntries(user->Y_loc); CHKERRQ(ierr);
    ierr = DMGlobalToLocal(user->dm, Y, INSERT_VALUES, user->Y_loc);
    CHKERRQ(ierr);
    ierr = VecGetArrayReadAndMemType(user->Y_loc, (const PetscScalar **)&y,
                                     &y_mem_type); CHKERRQ(ierr);
    CeedVectorSetArray(data->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER,
                       y);
    CeedOperatorApply(data->op_inject_r, data->y_ceed, data->z_r_ceed,
                      CEED_REQUEST_IMMEDIATE);
    CeedVectorAXPY(data->z_r_ceed, -1.0, data->x_r_ceed);
    // ---- Y_r -=  J^T (- A_Gamma,I A_I,I^-1) Y_r
    CeedVectorPointwiseMult(data->y_r_ceed, data->y_r_ceed, data->mult_ceed);
    CeedVectorAXPY(data->y_r_ceed, -1.0, data->z_r_ceed);
  } else {
    CeedVectorPointwiseMult(data->y_r_ceed, data->y_r_ceed, data->mult_ceed);
  }
  // ---- Inject to Y_Pi
  ierr = VecGetArrayAndMemType(user->Y_Pi_loc, &y, &y_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(data->y_Pi_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER,
                     y);
  CeedOperatorApply(data->op_inject_Pi, data->y_r_ceed, data->y_Pi_ceed,
                    CEED_REQUEST_IMMEDIATE);
  CeedVectorTakeArray(data->y_Pi_ceed, MemTypeP2C(y_mem_type), &y);
  ierr = VecRestoreArrayAndMemType(user->Y_Pi_loc, &y); CHKERRQ(ierr);
  // ---- Global-To-Local
  ierr = VecZeroEntries(user->Y_Pi); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm_Pi, user->Y_Pi_loc, ADD_VALUES, user->Y_Pi);
  CHKERRQ(ierr);
  // Note: current values in Y_Pi, Y_r

  // K_u^-T - update nodal values from subdomain
  // -- X_r = A_r,r^-1 Y_r
  ierr = BDDCArrInv(user, data->y_r_ceed, data->x_r_ceed); CHKERRQ(ierr);
  // -- X_Pi = A_Pi,r X_r
  ierr = VecGetArrayAndMemType(user->X_Pi_loc, &x, &x_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(data->x_Pi_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER,
                     x);
  CeedVectorPointwiseMult(data->x_r_ceed, data->x_r_ceed, data->mask_r_ceed);
  CeedOperatorApply(data->op_Pi_r, data->x_r_ceed, data->x_Pi_ceed,
                    CEED_REQUEST_IMMEDIATE);
  CeedVectorTakeArray(data->x_Pi_ceed, MemTypeP2C(x_mem_type), &x);
  ierr = VecRestoreArrayAndMemType(user->X_Pi_loc, &x); CHKERRQ(ierr);
  ierr = VecZeroEntries(user->X_Pi); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm_Pi, user->X_Pi_loc, ADD_VALUES, user->X_Pi);
  CHKERRQ(ierr);
  // -- Y_Pi -= A_Pi_r A_r,r^-1 Y_r == X_Pi
  ierr = VecAXPY(user->Y_Pi, -1.0, user->X_Pi);
  // Note: current values in Y_Pi, Y_r

  // P^-1 - subdomain and Schur compliment solve
  // -- X_r = A_r,r^-1 Y_r
  ierr = BDDCArrInv(user, data->y_r_ceed, data->x_r_ceed); CHKERRQ(ierr);
  // -- X_Pi = S_Pi^-1 Y_Pi
  ierr = KSPSolve(user->ksp_S_Pi, user->Y_Pi, user->X_Pi); CHKERRQ(ierr);
  // Note: current values in X_Pi, X_r

  // K_u^-1 - update subdomain values from nodes
  // -- Y_r = A_r,Pi X_Pi
  ierr = VecZeroEntries(user->X_Pi_loc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm_Pi, user->X_Pi, INSERT_VALUES, user->X_Pi_loc);
  CHKERRQ(ierr);
  ierr = VecGetArrayReadAndMemType(user->X_Pi_loc, (const PetscScalar **)&x,
                                   &x_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(data->x_Pi_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER,
                     x);
  CeedOperatorApply(data->op_r_Pi, data->x_Pi_ceed, data->y_r_ceed,
                    CEED_REQUEST_IMMEDIATE);
  CeedVectorTakeArray(data->x_Pi_ceed, MemTypeP2C(x_mem_type), &x);
  ierr = VecRestoreArrayReadAndMemType(user->X_Pi_loc, (const PetscScalar **)&x);
  CHKERRQ(ierr);
  // -- Z_r = A_r,r^-1 Y_r
  ierr = BDDCArrInv(user, data->y_r_ceed, data->z_r_ceed); CHKERRQ(ierr);
  // -- X_r -= A_r,r^-1 A_r,Pi X_Pi == Z_r
  CeedVectorAXPY(data->x_r_ceed, -1.0, data->z_r_ceed);
  // Note: current values in X_Pi, X_r

  // Restrict to fine space
  // -- Scaled restriction, point multiply by 1/multiplicity
  // ---- Copy X_Pi to X_r
  ierr = VecGetArrayReadAndMemType(user->X_Pi_loc, (const PetscScalar **)&x,
                                   &x_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(data->x_Pi_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER,
                     x);
  CeedVectorPointwiseMult(data->x_r_ceed, data->x_r_ceed, data->mask_r_ceed);
  CeedOperatorApplyAdd(data->op_restrict_Pi, data->x_Pi_ceed, data->x_r_ceed,
                       CEED_REQUEST_IMMEDIATE);
  CeedVectorTakeArray(data->x_Pi_ceed, MemTypeP2C(x_mem_type), &x);
  ierr = VecRestoreArrayReadAndMemType(user->X_Pi_loc, (const PetscScalar **)&x);
  CHKERRQ(ierr);
  // -- Harmonic injection, scaled with jump map
  if (user->is_harmonic) {
    // ---- J^T (jump map)
    CeedVectorPointwiseMult(data->z_r_ceed, data->x_r_ceed, data->mult_ceed);
    // ------ Local-to-Global
    ierr = VecGetArrayAndMemType(user->Y_loc, &y, &y_mem_type); CHKERRQ(ierr);
    CeedVectorSetArray(data->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER,
                       y);
    CeedOperatorApply(data->op_restrict_r, data->z_r_ceed, data->y_ceed,
                      CEED_REQUEST_IMMEDIATE);
    CeedVectorTakeArray(data->y_ceed, MemTypeP2C(y_mem_type), &y);
    ierr = VecRestoreArrayAndMemType(user->Y_loc, &y); CHKERRQ(ierr);
    ierr = VecZeroEntries(Y); CHKERRQ(ierr);
    ierr = DMLocalToGlobal(user->dm, user->Y_loc, ADD_VALUES, Y);
    CHKERRQ(ierr);
    // ------ Global-to-Local
    ierr = VecZeroEntries(user->Y_loc); CHKERRQ(ierr);
    ierr = DMGlobalToLocal(user->dm, Y, INSERT_VALUES, user->Y_loc);
    CHKERRQ(ierr);
    ierr = VecGetArrayReadAndMemType(user->Y_loc, (const PetscScalar **)&y,
                                     &y_mem_type); CHKERRQ(ierr);
    CeedVectorSetArray(data->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER,
                       y);
    CeedOperatorApply(data->op_inject_r, data->y_ceed, data->z_r_ceed,
                      CEED_REQUEST_IMMEDIATE);
    CeedVectorAXPY(data->z_r_ceed, -1.0, data->x_r_ceed);
    CeedVectorTakeArray(data->y_ceed, MemTypeP2C(y_mem_type), &y);
    ierr = VecRestoreArrayAndMemType(user->Y_loc, &y); CHKERRQ(ierr);
    // ---- Y_r = A_I,Gamma Z_r
    CeedVectorPointwiseMult(data->z_r_ceed, data->z_r_ceed, data->mask_Gamma_ceed);
    CeedOperatorApply(data->op_r_r, data->z_r_ceed, data->y_r_ceed,
                      CEED_REQUEST_IMMEDIATE);
    // ---- Z_r = A_I,I^-1 Y_r
    ierr = BDDCArrInv(user, data->y_r_ceed, data->z_r_ceed); CHKERRQ(ierr);
    // ---- X_r += A_I,I^-1 A_I,Gamma J X_r
    CeedVectorPointwiseMult(data->z_r_ceed, data->z_r_ceed, data->mask_I_ceed);
    CeedVectorPointwiseMult(data->x_r_ceed, data->x_r_ceed, data->mult_ceed);
    CeedVectorAXPY(data->x_r_ceed, -1.0, data->z_r_ceed);
  } else {
    CeedVectorPointwiseMult(data->x_r_ceed, data->x_r_ceed, data->mult_ceed);
  }
  // ---- Restrict to Y
  ierr = VecGetArrayAndMemType(user->Y_loc, &y, &y_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(data->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER,
                     y);
  CeedOperatorApply(data->op_restrict_r, data->x_r_ceed, data->y_ceed,
                    CEED_REQUEST_IMMEDIATE);
  CeedVectorTakeArray(data->y_ceed, MemTypeP2C(y_mem_type), &y);
  ierr = VecRestoreArrayAndMemType(user->Y_loc, &y); CHKERRQ(ierr);
  // ---- Local-to-Global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, user->Y_loc, ADD_VALUES, Y);
  CHKERRQ(ierr);
  // Note: current values in Y

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function calculates the error in the final solution
// -----------------------------------------------------------------------------
PetscErrorCode ComputeErrorMax(UserO user, CeedOperator op_error,
                               Vec X, CeedVector target,
                               PetscScalar *max_error) {
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
  CeedOperatorApply(op_error, user->x_ceed, collocated_error,
                    CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vector
  CeedVectorTakeArray(user->x_ceed, MemTypeP2C(mem_type), NULL);
  ierr = VecRestoreArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x);
  CHKERRQ(ierr);

  // Reduce max error
  *max_error = 0;
  const CeedScalar *e;
  CeedVectorGetArrayRead(collocated_error, CEED_MEM_HOST, &e);
  for (CeedInt i=0; i<length; i++) {
    *max_error = PetscMax(*max_error, PetscAbsScalar(e[i]));
  }
  CeedVectorRestoreArrayRead(collocated_error, &e);
  ierr = MPI_Allreduce(MPI_IN_PLACE, max_error, 1, MPIU_REAL, MPIU_MAX,
                       user->comm);
  CHKERRQ(ierr);

  // Cleanup
  CeedVectorDestroy(&collocated_error);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
