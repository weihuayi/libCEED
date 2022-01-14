#include "../include/setup-dm.h"

// ---------------------------------------------------------------------------
// Set-up DM
// ---------------------------------------------------------------------------
PetscErrorCode CreateDistributedDM(MPI_Comm comm, DM *dm) {
  PetscErrorCode  ierr;
  PetscSection   sec;
  PetscBool      interpolate = PETSC_TRUE;
  PetscInt       nx = 2, ny = 2;
  PetscInt       faces[2] = {nx, ny};
  PetscInt       dim, dofs_per_face;
  PetscInt       p_start, p_end;
  PetscInt       c_start, c_end; // cells
  PetscInt       f_start, f_end; // faces
  PetscInt       v_start, v_end; // vertices

  PetscFunctionBeginUser;

  ierr = PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces",
                                 faces, &dim, NULL); CHKERRQ(ierr);

  if (!dim) dim = 2;
  ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, faces, NULL,
                             NULL, NULL, interpolate, dm); CHKERRQ(ierr);
  // Get plex limits
  ierr = DMPlexGetChart(*dm, &p_start, &p_end); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dm, 0, &c_start, &c_end); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dm, 1, &f_start, &f_end); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(*dm, 0, &v_start, &v_end); CHKERRQ(ierr);
  // Create section
  ierr = PetscSectionCreate(comm, &sec); CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(sec, 2); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec, 0, "Velocity"); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec, 0, 1); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec, 1, "Pressure"); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec, 1, 1); CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec, p_start, p_end); CHKERRQ(ierr);
  // Setup dofs per face for velocity field
  for (PetscInt f = f_start; f < f_end; f++) {
    ierr = DMPlexGetConeSize(*dm, f, &dofs_per_face); CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(sec, f, 0, dofs_per_face); CHKERRQ(ierr);
    ierr = PetscSectionSetDof     (sec, f, dofs_per_face); CHKERRQ(ierr);
  }
  // Setup 1 dof per cell for pressure field
  for(PetscInt c = c_start; c < c_end; c++) {
    ierr = PetscSectionSetFieldDof(sec, c, 1, 1); CHKERRQ(ierr);
    ierr = PetscSectionSetDof     (sec, c, 1); CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sec); CHKERRQ(ierr);
  ierr = DMSetSection(*dm,sec); CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view"); CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};