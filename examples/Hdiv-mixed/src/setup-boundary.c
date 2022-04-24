#include "../include/setup-boundary.h"

// ---------------------------------------------------------------------------
// Create boundary label
// ---------------------------------------------------------------------------
PetscErrorCode CreateBCLabel(DM dm, const char name[]) {
  DMLabel        label;

  PetscFunctionBeginUser;

  PetscCall( DMCreateLabel(dm, name) );
  PetscCall( DMGetLabel(dm, name, &label) );
  PetscCall( DMPlexMarkBoundaryFaces(dm, PETSC_DETERMINE, label) );
  PetscCall( DMPlexLabelComplete(dm, label) );

  PetscFunctionReturn(0);
};

// ---------------------------------------------------------------------------
// Add Dirichlet boundaries to DM
// ---------------------------------------------------------------------------
PetscErrorCode DMAddBoundariesDirichlet(DM dm) {

  PetscFunctionBeginUser;

  // BCs given by manufactured solution
  PetscBool  has_label;
  const char *name = "MMS Face Sets";
  PetscInt   face_ids[1] = {1};
  PetscCall( DMHasLabel(dm, name, &has_label) );
  if (!has_label) {
    PetscCall( CreateBCLabel(dm, name) );
  }
  DMLabel label;
  PetscCall( DMGetLabel(dm, name, &label) );
  PetscCall( DMAddBoundary(dm, DM_BC_ESSENTIAL, "mms", label, 1, face_ids, 0, 0,
                           NULL,
                           (void(*)(void))BoundaryDirichletMMS, NULL, NULL, NULL) );


  PetscFunctionReturn(0);
}

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif
// ---------------------------------------------------------------------------
// Boundary function for manufactured solution
// ---------------------------------------------------------------------------
PetscErrorCode BoundaryDirichletMMS(PetscInt dim, PetscReal t,
                                    const PetscReal coords[],
                                    PetscInt num_comp_u, PetscScalar *u, void *ctx) {
  PetscScalar x = coords[0];
  PetscScalar y = coords[1];
  PetscScalar z = coords[1];

  PetscFunctionBeginUser;

  if (dim == 2) {
    u[0] = -M_PI*cos(M_PI*x) *sin(M_PI*y);
    u[1] = -M_PI*sin(M_PI*x) *cos(M_PI*y);
  } else {
    u[0] = -M_PI*cos(M_PI*x) *sin(M_PI*y) *sin(M_PI*z);
    u[1] = -M_PI*sin(M_PI*x) *cos(M_PI*y) *sin(M_PI*z);
    u[2] = -M_PI*sin(M_PI*x) *sin(M_PI*y) *cos(M_PI*z);
  }


  PetscFunctionReturn(0);
}
