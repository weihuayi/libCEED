#ifndef matops_h
#define matops_h

#include <ceed.h>
#include <petsc.h>

#include "structs.h"

PetscErrorCode ApplyLocal_Ceed(User user, Vec X, Vec Y);
PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y);
PetscErrorCode ComputeError(User user, Vec X, CeedVector target,
                            CeedScalar *l2_error_u, CeedScalar *l2_error_p);

#endif // matops_h
