#ifndef register_boundary_h
#define register_boundary_h

#include <petsc.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <ceed.h>
#include "../include/structs.h"

// ---------------------------------------------------------------------------
// Create boundary label
// ---------------------------------------------------------------------------
PetscErrorCode CreateBCLabel(DM dm, const char name[]);

// ---------------------------------------------------------------------------
// Add Dirichlet boundaries to DM
// ---------------------------------------------------------------------------
PetscErrorCode DMAddBoundariesDirichlet(DM dm);
PetscErrorCode BoundaryDirichletMMS(PetscInt dim, PetscReal t,
                                    const PetscReal coords[],
                                    PetscInt num_comp_u, PetscScalar *u, void *ctx);
#endif // register_boundary_h
