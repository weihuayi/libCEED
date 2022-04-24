#include "../include/setup-solvers.h"
#include "../include/setup-matops.h"
#include "../include/setup-libceed.h"

// -----------------------------------------------------------------------------
// Setup operator context data
// -----------------------------------------------------------------------------
PetscErrorCode SetupCommonCtx(MPI_Comm comm, DM dm, Ceed ceed,
                              CeedData ceed_data,
                              OperatorApplyContext op_apply_ctx) {
  PetscFunctionBeginUser;

  op_apply_ctx->comm = comm;
  op_apply_ctx->dm = dm;
  PetscCall( DMCreateLocalVector(dm, &op_apply_ctx->X_loc) );
  PetscCall( VecDuplicate(op_apply_ctx->X_loc, &op_apply_ctx->Y_loc) );
  op_apply_ctx->x_ceed = ceed_data->x_ceed;
  op_apply_ctx->y_ceed = ceed_data->y_ceed;
  op_apply_ctx->ceed = ceed;

  PetscFunctionReturn(0);
}

PetscErrorCode SetupJacobianOperatorCtx(CeedData ceed_data,
                                        OperatorApplyContext op_apply_ctx) {
  PetscFunctionBeginUser;

  op_apply_ctx->op_apply = ceed_data->op_jacobian;

  PetscFunctionReturn(0);
}

PetscErrorCode SetupResidualOperatorCtx(CeedData ceed_data,
                                        OperatorApplyContext op_apply_ctx) {
  PetscFunctionBeginUser;

  op_apply_ctx->op_apply = ceed_data->op_residual;

  PetscFunctionReturn(0);
}

PetscErrorCode SetupMMSOperatorCtx(CeedData ceed_data,
                                   OperatorApplyContext op_apply_ctx) {
  PetscFunctionBeginUser;

  op_apply_ctx->op_apply = ceed_data->op_error;

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// This function wraps the libCEED operator for a MatShell
// -----------------------------------------------------------------------------
PetscErrorCode ApplyJacobian(Mat A, Vec X, Vec Y) {
  OperatorApplyContext op_apply_ctx;

  PetscFunctionBeginUser;

  PetscCall( MatShellGetContext(A, &op_apply_ctx) );

  // libCEED for local action of residual evaluator
  PetscCall( ApplyLocalCeedOp(X, Y, op_apply_ctx) );

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the non-linear residual
// -----------------------------------------------------------------------------
PetscErrorCode SNESFormResidual(SNES snes, Vec X, Vec Y, void *ctx) {
  OperatorApplyContext op_apply_ctx = (OperatorApplyContext)ctx;

  PetscFunctionBeginUser;

  // Use computed BCs
  //PetscCall( VecZeroEntries(op_apply_ctx->X_loc) );
  //PetscCall( DMPlexInsertBoundaryValues(op_apply_ctx->dm, PETSC_TRUE,
  //                                      op_apply_ctx->X_loc,
  //                                      1.0, NULL, NULL, NULL) );

  // libCEED for local action of residual evaluator
  PetscCall( ApplyLocalCeedOp(X, Y, op_apply_ctx) );

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Jacobian setup
// -----------------------------------------------------------------------------
PetscErrorCode SNESFormJacobian(SNES snes, Vec U, Mat J, Mat J_pre, void *ctx) {
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Context data
  //FormJacobCtx  form_jacob_ctx = (FormJacobCtx)ctx;

  // J_pre might be AIJ (e.g., when using coloring), so we need to assemble it
  ierr = MatAssemblyBegin(J_pre, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J_pre, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  if (J != J_pre) {
    ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
};

// ---------------------------------------------------------------------------
// Setup Solver
// ---------------------------------------------------------------------------
PetscErrorCode PDESolver(CeedData ceed_data, VecType vec_type, SNES snes,
                         KSP ksp,
                         Vec F, Vec *U_g, OperatorApplyContext op_apply_ctx) {

  PetscInt       U_l_size, U_g_size;

  PetscFunctionBeginUser;

  // Create global unknown solution U_g
  PetscCall( DMCreateGlobalVector(op_apply_ctx->dm, U_g) );
  PetscCall( VecGetSize(*U_g, &U_g_size) );
  // Local size for matShell
  PetscCall( VecGetLocalSize(*U_g, &U_l_size) );
  Vec R;
  PetscCall( VecDuplicate(*U_g, &R) );

  // ---------------------------------------------------------------------------
  // Setup SNES
  // ---------------------------------------------------------------------------
  // Operator
  Mat mat_jacobian;
  SetupJacobianOperatorCtx(ceed_data, op_apply_ctx);
  PetscCall( SNESSetDM(snes, op_apply_ctx->dm) );
  // -- Form Action of Jacobian on delta_u
  PetscCall( MatCreateShell(op_apply_ctx->comm, U_l_size, U_l_size, U_g_size,
                            U_g_size, op_apply_ctx, &mat_jacobian) );
  PetscCall( MatShellSetOperation(mat_jacobian, MATOP_MULT,
                                  (void (*)(void))ApplyJacobian) );
  PetscCall( MatShellSetVecType(mat_jacobian, vec_type) );

  // Set SNES residual evaluation function
  SetupResidualOperatorCtx(ceed_data, op_apply_ctx);
  PetscCall( SNESSetFunction(snes, R, SNESFormResidual, op_apply_ctx) );
  // -- SNES Jacobian
  SetupJacobianOperatorCtx(ceed_data, op_apply_ctx);
  PetscCall( SNESSetJacobian(snes, mat_jacobian, mat_jacobian,
                             SNESFormJacobian, op_apply_ctx) );

  // Setup KSP
  PetscCall( KSPSetFromOptions(ksp) );

  // Default to critical-point (CP) line search (related to Wolfe's curvature condition)
  SNESLineSearch line_search;

  PetscCall( SNESGetLineSearch(snes, &line_search) );
  PetscCall( SNESLineSearchSetType(line_search, SNESLINESEARCHCP) );
  PetscCall( SNESSetFromOptions(snes) );

  // Solve
  PetscCall( VecSet(*U_g, 0.0));
  PetscCall( SNESSolve(snes, F, *U_g));

  // Free PETSc objects
  PetscCall( MatDestroy(&mat_jacobian) );
  PetscCall( VecDestroy(&R) );
  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function calculates the L2 error in the final solution
// -----------------------------------------------------------------------------
PetscErrorCode ComputeL2Error(CeedData ceed_data, Vec X, CeedVector target,
                              CeedScalar *l2_error_u,
                              CeedScalar *l2_error_p,
                              OperatorApplyContext op_apply_ctx) {
  PetscScalar *x;
  PetscMemType mem_type;
  CeedVector collocated_error;
  CeedSize length;

  PetscFunctionBeginUser;

  CeedVectorGetLength(target, &length);
  CeedVectorCreate(op_apply_ctx->ceed, length, &collocated_error);
  SetupMMSOperatorCtx(ceed_data, op_apply_ctx);
  // Global-to-local
  PetscCall( DMGlobalToLocal(op_apply_ctx->dm, X, INSERT_VALUES,
                             op_apply_ctx->X_loc) );

  // Setup CEED vector
  PetscCall( VecGetArrayAndMemType(op_apply_ctx->X_loc, &x, &mem_type) );
  CeedVectorSetArray(op_apply_ctx->x_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER,
                     x);

  // Apply CEED operator
  CeedOperatorApply(op_apply_ctx->op_apply, op_apply_ctx->x_ceed,
                    collocated_error,
                    CEED_REQUEST_IMMEDIATE);
  // Restore PETSc vector
  CeedVectorTakeArray(op_apply_ctx->x_ceed, MemTypeP2C(mem_type), NULL);
  PetscCall( VecRestoreArrayReadAndMemType(op_apply_ctx->X_loc,
             (const PetscScalar **)&x) );
  // Compute L2 error for each field
  CeedInt c_start, c_end, dim, num_elem, num_qpts;
  PetscCall( DMGetDimension(op_apply_ctx->dm, &dim) );
  PetscCall( DMPlexGetHeightStratum(op_apply_ctx->dm, 0, &c_start, &c_end) );
  num_elem = c_end -c_start;
  num_qpts = length / (num_elem*(dim+1));
  CeedInt cent_qpts = num_qpts / 2;
  CeedVector collocated_error_u, collocated_error_p;
  const CeedScalar *E_U; // to store total error
  CeedInt length_u, length_p;
  length_p = num_elem;
  length_u = num_elem*num_qpts*dim;
  CeedScalar e_u[length_u], e_p[length_p];
  CeedVectorCreate(op_apply_ctx->ceed, length_p, &collocated_error_p);
  CeedVectorCreate(op_apply_ctx->ceed, length_u, &collocated_error_u);
  // E_U is ordered as [p_0,u_0/.../p_n,u_n] for 0 to n elements
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
// This function print the output
// -----------------------------------------------------------------------------
PetscErrorCode PrintOutput(MPI_Comm comm, Ceed ceed,
                           CeedMemType mem_type_backend,
                           SNES snes, KSP ksp,
                           Vec U, CeedScalar l2_error_u,
                           CeedScalar l2_error_p, AppCtx app_ctx) {

  PetscFunctionBeginUser;

  const char *used_resource;
  CeedGetResource(ceed, &used_resource);
  char hostname[PETSC_MAX_PATH_LEN];
  PetscCall( PetscGetHostName(hostname, sizeof hostname) );
  PetscInt comm_size;
  PetscCall( MPI_Comm_size(comm, &comm_size) );
  PetscCall( PetscPrintf(comm,
                         "\n-- Mixed H(div) Example - libCEED + PETSc --\n"
                         "  MPI:\n"
                         "    Hostname                           : %s\n"
                         "    Total ranks                        : %d\n"
                         "  libCEED:\n"
                         "    libCEED Backend                    : %s\n"
                         "    libCEED Backend MemType            : %s\n",
                         hostname, comm_size, used_resource, CeedMemTypes[mem_type_backend]) );

  VecType vecType;
  PetscCall( VecGetType(U, &vecType) );
  PetscCall( PetscPrintf(comm,
                         "  PETSc:\n"
                         "    PETSc Vec Type                     : %s\n",
                         vecType) );

  PetscInt       U_l_size, U_g_size;
  PetscCall( VecGetSize(U, &U_g_size) );
  PetscCall( VecGetLocalSize(U, &U_l_size) );
  PetscCall( PetscPrintf(comm,
                         "  Problem:\n"
                         "    Problem Name                       : %s\n"
                         "    Global nodes (u + p)               : %" PetscInt_FMT "\n"
                         "    Owned nodes (u + p)                : %" PetscInt_FMT "\n",
                         app_ctx->problem_name, U_g_size, U_l_size
                        ) );
  // -- SNES
  PetscInt its, snes_its = 0;
  PetscCall( SNESGetIterationNumber(snes, &its) );
  snes_its += its;
  SNESType snes_type;
  SNESConvergedReason snes_reason;
  PetscReal snes_rnorm;
  PetscCall( SNESGetType(snes, &snes_type) );
  PetscCall( SNESGetConvergedReason(snes, &snes_reason) );
  PetscCall( SNESGetFunctionNorm(snes, &snes_rnorm) );
  PetscCall( PetscPrintf(comm,
                         "  SNES:\n"
                         "    SNES Type                          : %s\n"
                         "    SNES Convergence                   : %s\n"
                         "    Total SNES Iterations              : %" PetscInt_FMT "\n"
                         "    Final rnorm                        : %e\n",
                         snes_type, SNESConvergedReasons[snes_reason],
                         snes_its, (double)snes_rnorm) );

  PetscInt ksp_its = 0;
  PetscCall( SNESGetLinearSolveIterations(snes, &its) );
  ksp_its += its;
  KSPType ksp_type;
  KSPConvergedReason ksp_reason;
  PetscReal ksp_rnorm;
  PC pc;
  PCType pc_type;
  PetscCall( KSPGetPC(ksp, &pc) );
  PetscCall( PCGetType(pc, &pc_type) );
  PetscCall( KSPGetType(ksp, &ksp_type) );
  PetscCall( KSPGetConvergedReason(ksp, &ksp_reason) );
  PetscCall( KSPGetIterationNumber(ksp, &ksp_its) );
  PetscCall( KSPGetResidualNorm(ksp, &ksp_rnorm) );
  PetscCall( PetscPrintf(comm,
                         "  KSP:\n"
                         "    KSP Type                           : %s\n"
                         "    PC Type                            : %s\n"
                         "    KSP Convergence                    : %s\n"
                         "    Total KSP Iterations               : %" PetscInt_FMT "\n"
                         "    Final rnorm                        : %e\n",
                         ksp_type, pc_type, KSPConvergedReasons[ksp_reason], ksp_its,
                         (double)ksp_rnorm ) );

  PetscCall( PetscPrintf(comm,
                         "  L2 Error (MMS):\n"
                         "    L2 error of u and p                : %e, %e\n",
                         (double)l2_error_u,
                         (double)l2_error_p) );
  PetscFunctionReturn(0);
};
// -----------------------------------------------------------------------------
