#ifndef setup_solvers_h
#define setup_solvers_h

#include <ceed.h>
#include <petsc.h>

#include "structs.h"

PetscErrorCode SetupCommonCtx(MPI_Comm comm, DM dm, Ceed ceed,
                              CeedData ceed_data,
                              OperatorApplyContext op_apply_ctx);
PetscErrorCode SetupJacobianOperatorCtx(CeedData ceed_data,
                                        OperatorApplyContext op_apply_ctx);
PetscErrorCode SetupResidualOperatorCtx(CeedData ceed_data,
                                        OperatorApplyContext op_apply_ctx);
PetscErrorCode SetupMMSOperatorCtx(CeedData ceed_data,
                                   OperatorApplyContext op_apply_ctx);
PetscErrorCode ApplyJacobian(Mat A, Vec X, Vec Y);
PetscErrorCode SNESFormResidual(SNES snes, Vec X, Vec Y, void *ctx);
PetscErrorCode SNESFormJacobian(SNES snes, Vec U, Mat J, Mat J_pre, void *ctx);
PetscErrorCode PDESolver(CeedData ceed_data, VecType vec_type, SNES snes,
                         KSP ksp,
                         Vec F, Vec *U_g, OperatorApplyContext op_apply_ctx);
PetscErrorCode ComputeL2Error(CeedData ceed_data, Vec U, CeedVector target,
                              CeedScalar *l2_error_u, CeedScalar *l2_error_p,
                              OperatorApplyContext op_apply_ctx);
PetscErrorCode PrintOutput(MPI_Comm comm, Ceed ceed,
                           CeedMemType mem_type_backend,
                           SNES snes, KSP ksp,
                           Vec U, CeedScalar l2_error_u,
                           CeedScalar l2_error_p, AppCtx app_ctx);

#endif // setup_solvers_h
