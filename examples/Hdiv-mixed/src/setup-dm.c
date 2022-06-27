#include "../include/setup-dm.h"

// ---------------------------------------------------------------------------
// Setup DM
// ---------------------------------------------------------------------------
PetscErrorCode CreateDM(MPI_Comm comm, VecType vec_type, DM *dm) {

  PetscFunctionBeginUser;

  // Create DMPLEX
  PetscCall( DMCreate(comm, dm) );
  PetscCall( DMSetType(*dm, DMPLEX) );
  PetscCall( DMSetVecType(*dm, vec_type) );
  // Set Tensor elements
  PetscCall( PetscOptionsSetValue(NULL, "-dm_plex_simplex", "0") );
  // Set CL options
  PetscCall( DMSetFromOptions(*dm) );
  PetscCall( DMViewFromOptions(*dm, NULL, "-dm_view") );

  PetscFunctionReturn(0);
};