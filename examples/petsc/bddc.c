// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

//                        libCEED + PETSc Example: CEED BPs 3-6 with BDDC
//
// This example demonstrates a simple usage of libCEED with PETSc to solve the
// CEED BP benchmark problems, see http://ceed.exascaleproject.org/bps.
//
// The code uses higher level communication protocols in DMPlex.
//
// Build with:
//
//     make bddc [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     bddc -problem bp3
//     bddc -problem bp4
//     bddc -problem bp5 -ceed /cpu/self
//     bddc -problem bp6 -ceed /gpu/cuda
//
//TESTARGS -ceed {ceed_resource} -test -problem bp3 -degree 3

/// @file
/// CEED BPs 1-6 BDDC example using PETSc
const char help[] = "Solve CEED BPs using BDDC with PETSc and DMPlex\n";

#include "bddc.h"

// The BDDC example uses vectors in three spaces
//
//  Fine mesh:       Broken mesh:      Vertex mesh:    Broken vertex mesh:
// x----x----x      x----x x----x       x    x    x      x    x x    x
// |    |    |      |    | |    |
// |    |    |      |    | |    |
// x----x----x      x----x x----x       x    x    x      x    x x    x
//
// Vectors are organized as follows
//  - *_Pi    : vector on the vertex mesh
//  - *_Pi_r  : vector on the broken vertex mesh
//  - *_r     : vector on the broken mesh, all points but vertices
//  - *_Gamma : vector on the broken mesh, face/vertex/edge points
//  - *_I     : vector on the broken mesh, interior points
//  - *       : all other vectors are on the fine mesh

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  char filename[PETSC_MAX_PATH_LEN],
       ceed_resource[PETSC_MAX_PATH_LEN] = "/cpu/self";
  double my_rt_start, my_rt, rt_min, rt_max;
  PetscInt degree = 3, q_extra, l_size, xl_size, g_size, l_Pi_size,
           xl_Pi_size, g_Pi_size, dim = 3, mesh_elem[3] = {3, 3, 3}, num_comp_u = 1;
  PetscScalar *r;
  PetscScalar eps = 1.0;
  PetscBool test_mode, benchmark_mode, read_mesh, write_solution;
  PetscLogStage solve_stage;
  DM  dm, dm_Pi;
  SNES snes_Pi, snes_Pi_r;
  KSP ksp, ksp_S_Pi, ksp_S_Pi_r;
  PC pc;
  Mat mat_O, mat_S_Pi, mat_S_Pi_r;
  Vec X, X_loc, X_Pi, X_Pi_loc, X_Pi_r_loc, rhs, rhs_loc;
  PetscMemType mem_type;
  UserO user_O;
  UserBDDC user_bddc;
  Ceed ceed;
  CeedData ceed_data;
  CeedDataBDDC ceed_data_bddc;
  CeedVector rhs_ceed, target;
  CeedQFunction qf_error, qf_restrict, qf_prolong;
  CeedOperator op_error;
  BPType bp_choice;
  InjectionType injection;

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr)  return ierr;
  comm = PETSC_COMM_WORLD;

  // Parse command line options
  PetscOptionsBegin(comm, NULL, "CEED BPs in PETSc", NULL);
  bp_choice = CEED_BP3;
  ierr = PetscOptionsEnum("-problem",
                          "CEED benchmark problem to solve", NULL,
                          bp_types, (PetscEnum)bp_choice, (PetscEnum *)&bp_choice,
                          NULL); CHKERRQ(ierr);
  num_comp_u = bp_options[bp_choice].num_comp_u;
  test_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-test",
                          "Testing mode (do not print unless error is large)",
                          NULL, test_mode, &test_mode, NULL); CHKERRQ(ierr);
  benchmark_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-benchmark",
                          "Benchmarking mode (prints benchmark statistics)",
                          NULL, benchmark_mode, &benchmark_mode, NULL);
  CHKERRQ(ierr);
  write_solution = PETSC_FALSE;
  ierr = PetscOptionsBool("-write_solution",
                          "Write solution for visualization",
                          NULL, write_solution, &write_solution, NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-eps",
                            "Epsilon parameter for Kershaw mesh transformation",
                            NULL, eps, &eps, NULL);
  if (eps > 1 || eps <= 0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                                     "-eps %D must be (0,1]", eps);
  degree = test_mode ? 3 : 2;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of tensor product basis",
                         NULL, degree, &degree, NULL); CHKERRQ(ierr);
  if (degree < 2) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                            "-degree %D must be at least 2", degree);
  q_extra = bp_options[bp_choice].q_extra;
  ierr = PetscOptionsInt("-q_extra", "Number of extra quadrature points",
                         NULL, q_extra, &q_extra, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceed_resource, ceed_resource,
                            sizeof(ceed_resource), NULL); CHKERRQ(ierr);
  injection = INJECTION_SCALED;
  ierr = PetscOptionsEnum("-injection",
                          "Injection strategy to use", NULL,
                          injection_types, (PetscEnum)injection,
                          (PetscEnum *)&injection, NULL); CHKERRQ(ierr);
  read_mesh = PETSC_FALSE;
  ierr = PetscOptionsString("-mesh", "Read mesh from file", NULL,
                            filename, filename, sizeof(filename), &read_mesh);
  CHKERRQ(ierr);
  if (!read_mesh) {
    PetscInt tmp = dim;
    ierr = PetscOptionsIntArray("-cells","Number of cells per dimension", NULL,
                                mesh_elem, &tmp, NULL); CHKERRQ(ierr);
  }
  PetscOptionsEnd();

  // Set up libCEED
  CeedInit(ceed_resource, &ceed);
  CeedMemType mem_type_backend;
  CeedGetPreferredMemType(ceed, &mem_type_backend);

  // Setup DMs
  if (read_mesh) {
    ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, NULL, PETSC_TRUE, &dm);
    CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, mesh_elem, NULL,
                               NULL, NULL, PETSC_TRUE, &dm); CHKERRQ(ierr);
  }

  {
    DM dm_dist = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(dm, &part); CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part); CHKERRQ(ierr);
    ierr = DMPlexDistribute(dm, 0, NULL, &dm_dist); CHKERRQ(ierr);
    if (dm_dist) {
      ierr = DMDestroy(&dm); CHKERRQ(ierr);
      dm = dm_dist;
    }
  }

  // Apply Kershaw mesh transformation
  ierr = Kershaw(dm, eps); CHKERRQ(ierr);

  VecType vec_type;
  switch (mem_type_backend) {
  case CEED_MEM_HOST: vec_type = VECSTANDARD; break;
  case CEED_MEM_DEVICE: {
    const char *resolved;
    CeedGetResource(ceed, &resolved);
    if (strstr(resolved, "/gpu/cuda")) vec_type = VECCUDA;
    else if (strstr(resolved, "/gpu/hip/occa"))
      vec_type = VECSTANDARD; // https://github.com/CEED/libCEED/issues/678
    else if (strstr(resolved, "/gpu/hip")) vec_type = VECHIP;
    else vec_type = VECSTANDARD;
  }
  }
  ierr = DMSetVecType(dm, vec_type); CHKERRQ(ierr);

  // Setup DM
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  ierr = SetupDMByDegree(dm, degree, num_comp_u, dim,
                         bp_options[bp_choice].enforce_bc, bp_options[bp_choice].bc_func);
  CHKERRQ(ierr);

  // Set up subdomain vertex DM
  ierr = DMClone(dm, &dm_Pi); CHKERRQ(ierr);
  ierr = DMSetVecType(dm_Pi, vec_type); CHKERRQ(ierr);
  ierr = SetupVertexDMFromDM(dm, dm_Pi, num_comp_u,
                             bp_options[bp_choice].enforce_bc,
                             bp_options[bp_choice].bc_func);
  CHKERRQ(ierr);

  // Create vectors
  // -- Fine mesh
  ierr = DMCreateGlobalVector(dm, &X); CHKERRQ(ierr);
  ierr = VecGetLocalSize(X, &l_size); CHKERRQ(ierr);
  ierr = VecGetSize(X, &g_size); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, &X_loc); CHKERRQ(ierr);
  ierr = VecGetSize(X_loc, &xl_size); CHKERRQ(ierr);
  // -- Vertex mesh
  ierr = DMCreateGlobalVector(dm_Pi, &X_Pi); CHKERRQ(ierr);
  ierr = VecGetLocalSize(X_Pi, &l_Pi_size); CHKERRQ(ierr);
  ierr = VecGetSize(X_Pi, &g_Pi_size); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm_Pi, &X_Pi_loc); CHKERRQ(ierr);
  ierr = VecGetSize(X_Pi_loc, &xl_Pi_size); CHKERRQ(ierr);

  // Operator
  ierr = PetscMalloc1(1, &user_O); CHKERRQ(ierr);
  ierr = MatCreateShell(comm, l_size, l_size, g_size, g_size,
                        user_O, &mat_O); CHKERRQ(ierr);
  ierr = MatShellSetOperation(mat_O, MATOP_MULT,
                              (void(*)(void))MatMult_Ceed); CHKERRQ(ierr);
  ierr = MatShellSetOperation(mat_O, MATOP_GET_DIAGONAL,
                              (void(*)(void))MatGetDiag); CHKERRQ(ierr);
  ierr = MatShellSetVecType(mat_O, vec_type); CHKERRQ(ierr);

  // Print global grid information
  if (!test_mode) {
    PetscInt P = degree + 1, Q = P + q_extra;

    const char *used_resource;
    CeedGetResource(ceed, &used_resource);

    ierr = VecGetType(X, &vec_type); CHKERRQ(ierr);

    ierr = PetscPrintf(comm,
                       "\n-- CEED Benchmark Problem %d -- libCEED + PETSc + BDDC --\n"
                       "  PETSc:\n"
                       "    PETSc Vec Type                     : %s\n"
                       "  libCEED:\n"
                       "    libCEED Backend                    : %s\n"
                       "    libCEED Backend MemType            : %s\n"
                       "  Mesh:\n"
                       "    Number of 1D Basis Nodes (p)       : %d\n"
                       "    Number of 1D Quadrature Points (q) : %d\n"
                       "    Global Nodes                       : %D\n"
                       "    Owned Nodes                        : %D\n"
                       "    DoF per node                       : %D\n"
                       "  BDDC:\n"
                       "    Injection                          : %s\n"
                       "    Global Interface Nodes             : %D\n"
                       "    Owned Interface Nodes              : %D\n",
                       bp_choice+1, vec_type, used_resource,
                       CeedMemTypes[mem_type_backend],
                       P, Q, g_size/num_comp_u, l_size/num_comp_u,
                       num_comp_u, injection_types[injection], g_Pi_size,
                       l_Pi_size); CHKERRQ(ierr);
  }

  // Create RHS vector
  ierr = VecDuplicate(X, &rhs); CHKERRQ(ierr);
  ierr = VecDuplicate(X_loc, &rhs_loc); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhs_loc); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(rhs_loc, &r, &mem_type); CHKERRQ(ierr);
  CeedVectorCreate(ceed, xl_size, &rhs_ceed);
  CeedVectorSetArray(rhs_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, r);

  // Set up libCEED operator
  ierr = PetscCalloc1(1, &ceed_data); CHKERRQ(ierr);
  ierr = SetupLibceedByDegree(dm, ceed, degree, dim, q_extra,
                              dim, num_comp_u, g_size, xl_size,
                              bp_options[bp_choice], ceed_data,
                              true, rhs_ceed, &target);
  CHKERRQ(ierr);

  // Gather RHS
  CeedVectorTakeArray(rhs_ceed, MemTypeP2C(mem_type), NULL);
  ierr = VecRestoreArrayAndMemType(rhs_loc, &r); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhs); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm, rhs_loc, ADD_VALUES, rhs); CHKERRQ(ierr);
  CeedVectorDestroy(&rhs_ceed);

  // Set up libCEED operator on interface vertices
  ierr = PetscMalloc1(1, &ceed_data_bddc); CHKERRQ(ierr);
  ierr = SetupLibceedBDDC(dm_Pi, ceed_data, ceed_data_bddc, g_Pi_size,
                          xl_Pi_size, bp_options[bp_choice]);
  CHKERRQ(ierr);

  // Create the injection/restriction QFunction
  CeedQFunctionCreateIdentity(ceed, num_comp_u, CEED_EVAL_NONE, CEED_EVAL_INTERP,
                              &qf_restrict);
  CeedQFunctionCreateIdentity(ceed, num_comp_u, CEED_EVAL_INTERP, CEED_EVAL_NONE,
                              &qf_prolong);

  // Create the error QFunction
  CeedQFunctionCreateInterior(ceed, 1, bp_options[bp_choice].error,
                              bp_options[bp_choice].error_loc, &qf_error);
  CeedQFunctionAddInput(qf_error, "u", num_comp_u, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "true_soln", num_comp_u, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_error, "error", num_comp_u, CEED_EVAL_NONE);

  // Create the error operator
  CeedOperatorCreate(ceed, qf_error, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_error);
  CeedOperatorSetField(op_error, "u", ceed_data->elem_restr_u,
                       ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "true_soln",
                       ceed_data->elem_restr_u_i,
                       CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_error, "error", ceed_data->elem_restr_u_i,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Calculate multiplicity
  {
    PetscScalar *x;

    // CEED vector
    ierr = VecZeroEntries(X_loc); CHKERRQ(ierr);
    ierr = VecGetArray(X_loc, &x); CHKERRQ(ierr);
    CeedVectorSetArray(ceed_data->x_ceed, CEED_MEM_HOST, CEED_USE_POINTER, x);

    // Multiplicity
    CeedElemRestrictionGetMultiplicity(ceed_data->elem_restr_u,
                                       ceed_data->x_ceed);

    // Restore vector
    CeedVectorTakeArray(ceed_data->x_ceed, CEED_MEM_HOST, &x);
    ierr = VecRestoreArray(X_loc, &x); CHKERRQ(ierr);

    // Local-to-global
    ierr = VecZeroEntries(X); CHKERRQ(ierr);
    ierr = DMLocalToGlobal(dm, X_loc, ADD_VALUES, X);
    CHKERRQ(ierr);

    // Global-to-local
    ierr = DMGlobalToLocal(dm, X, INSERT_VALUES, X_loc);
    CHKERRQ(ierr);
    ierr = VecZeroEntries(X); CHKERRQ(ierr);

    // Multiplicity scaling
    ierr = VecReciprocal(X_loc);

    // CEED vector
    ierr = VecGetArray(X_loc, &x); CHKERRQ(ierr);
    CeedVectorSetArray(ceed_data->x_ceed, CEED_MEM_HOST, CEED_USE_POINTER, x);

    // Inject multiplicity
    CeedOperatorApply(ceed_data_bddc->op_inject_r, ceed_data->x_ceed,
                      ceed_data_bddc->mult_ceed, CEED_REQUEST_IMMEDIATE);
    // Restore vector
    CeedVectorTakeArray(ceed_data->x_ceed, CEED_MEM_HOST, &x);
    ierr = VecRestoreArray(X_loc, &x); CHKERRQ(ierr);
    ierr = VecZeroEntries(X_loc); CHKERRQ(ierr);
  }

  // Setup dummy SNESs
  {
    // Schur compliment operator
    // -- Jacobian Matrix
    ierr = DMSetMatType(dm_Pi, MATAIJ); CHKERRQ(ierr);
    ierr = DMCreateMatrix(dm_Pi, &mat_S_Pi); CHKERRQ(ierr);

    // -- Dummy SNES
    ierr = SNESCreate(comm, &snes_Pi); CHKERRQ(ierr);
    ierr = SNESSetDM(snes_Pi, dm_Pi); CHKERRQ(ierr);
    ierr = SNESSetSolution(snes_Pi, X_Pi); CHKERRQ(ierr);

    // -- Jacobian function
    ierr = SNESSetJacobian(snes_Pi, mat_S_Pi, mat_S_Pi, NULL,
                           NULL); CHKERRQ(ierr);

    // -- Residual evaluation function
    ierr = PetscMalloc1(1, &user_bddc); CHKERRQ(ierr);
    ierr = SNESSetFunction(snes_Pi, X_Pi, FormResidual_BDDCSchur,
                           user_bddc); CHKERRQ(ierr);
  }
  {
    // Element Schur compliment operator
    // -- Vectors
    PetscInt numesh_elem;
    CeedElemRestrictionGetNumElements(ceed_data->elem_restr_u, &numesh_elem);
    ierr = VecCreate(comm, &X_Pi_r_loc); CHKERRQ(ierr);
    ierr = VecSetSizes(X_Pi_r_loc, numesh_elem*8, PETSC_DECIDE); CHKERRQ(ierr);
    ierr = VecSetType(X_Pi_r_loc, vec_type); CHKERRQ(ierr);

    // -- Jacobian Matrix
    ierr = MatCreateSeqAIJ(comm, 8*numesh_elem, 8*numesh_elem, 8, NULL,
                           &mat_S_Pi_r);
    CHKERRQ(ierr);
    for (PetscInt e=0; e<numesh_elem; e++) {
      for (PetscInt i=0; i<8; i++) {
        for (PetscInt j=0; j<8; j++) {
          PetscInt row = e*8 + i;
          PetscInt col = e*8 + j;
          PetscScalar value = i + j;
          ierr = MatSetValues(mat_S_Pi_r, 1, &row, 1, &col, &value, INSERT_VALUES);
          CHKERRQ(ierr);
        }
      }
    }
    ierr = MatAssemblyBegin(mat_S_Pi_r, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mat_S_Pi_r, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    // -- Dummy SNES
    ierr = SNESCreate(comm, &snes_Pi_r); CHKERRQ(ierr);
    ierr = SNESSetSolution(snes_Pi_r, X_Pi_r_loc); CHKERRQ(ierr);

    // -- Jacobian function
    ierr = SNESSetJacobian(snes_Pi_r, mat_S_Pi_r, mat_S_Pi_r, NULL,
                           NULL); CHKERRQ(ierr);

    // -- Residual evaluation function
    ierr = SNESSetFunction(snes_Pi_r, X_Pi_r_loc, FormResidual_BDDCElementSchur,
                           user_bddc); CHKERRQ(ierr);
  }

  // Set up MatShell user data
  {
    user_O->comm = comm;
    user_O->dm = dm;
    user_O->X_loc = X_loc;
    ierr = VecDuplicate(X_loc, &user_O->Y_loc); CHKERRQ(ierr);
    user_O->x_ceed = ceed_data->x_ceed;
    user_O->y_ceed = ceed_data->y_ceed;
    user_O->op = ceed_data->op_apply;
    user_O->ceed = ceed;
  }

  // Set up PCShell user data (also used for Schur operators)
  {
    user_bddc->comm = comm;
    user_bddc->dm = dm;
    user_bddc->dm_Pi = dm_Pi;
    user_bddc->X_loc = X_loc;
    user_bddc->Y_loc = user_O->Y_loc;
    user_bddc->X_Pi = X_Pi;
    ierr = VecDuplicate(X_Pi, &user_bddc->Y_Pi); CHKERRQ(ierr);
    user_bddc->X_Pi_loc = X_Pi_loc;
    ierr = VecDuplicate(X_Pi_loc, &user_bddc->Y_Pi_loc); CHKERRQ(ierr);
    user_bddc->X_Pi_r_loc = X_Pi_r_loc;
    ierr = VecDuplicate(X_Pi_r_loc, &user_bddc->Y_Pi_r_loc); CHKERRQ(ierr);
    user_bddc->ceed_data_bddc = ceed_data_bddc;
    user_bddc->mat_S_Pi = mat_S_Pi;
    user_bddc->mat_S_Pi_r = mat_S_Pi_r; CHKERRQ(ierr);
    ierr = KSPCreate(comm, &ksp_S_Pi);
    ierr = KSPCreate(comm, &ksp_S_Pi_r); CHKERRQ(ierr);
    user_bddc->ksp_S_Pi = ksp_S_Pi;
    user_bddc->ksp_S_Pi_r = ksp_S_Pi_r;
    user_bddc->snes_Pi = snes_Pi;
    user_bddc->snes_Pi_r = snes_Pi_r;
    user_bddc->is_harmonic = injection == INJECTION_HARMONIC;
  }

  // Set up KSP
  ierr = KSPCreate(comm, &ksp); CHKERRQ(ierr);
  {
    ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
    ierr = KSPSetNormType(ksp, KSP_NORM_NATURAL); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                            PETSC_DEFAULT); CHKERRQ(ierr);
  }
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, mat_O, mat_O); CHKERRQ(ierr);

  // Set up PCShell
  ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
  {
    ierr = PCSetType(pc, PCSHELL); CHKERRQ(ierr);
    ierr = PCShellSetContext(pc, user_bddc); CHKERRQ(ierr);
    ierr = PCShellSetApply(pc, PCShellApply_BDDC); CHKERRQ(ierr);
    ierr = PCShellSetSetUp(pc, PCShellSetup_BDDC); CHKERRQ(ierr);

    // Set up Schur compilemnt solvers
    {
      // -- Vertex mesh AMG
      PC pc_S_Pi;
      ierr = KSPSetType(ksp_S_Pi, KSPPREONLY); CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp_S_Pi, mat_S_Pi, mat_S_Pi); CHKERRQ(ierr);

      ierr = KSPGetPC(ksp_S_Pi, &pc_S_Pi); CHKERRQ(ierr);
      ierr = PCSetType(pc_S_Pi, PCGAMG); CHKERRQ(ierr);

      ierr = KSPSetOptionsPrefix(ksp_S_Pi, "S_Pi_"); CHKERRQ(ierr);
      ierr = PCSetOptionsPrefix(pc_S_Pi, "S_Pi_"); CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp_S_Pi); CHKERRQ(ierr);
      ierr = PCSetFromOptions(pc_S_Pi); CHKERRQ(ierr);
    }
    {
      // -- Broken mesh AMG
      PC pc_S_Pi_r;
      ierr = KSPSetType(ksp_S_Pi_r, KSPPREONLY); CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp_S_Pi_r, mat_S_Pi_r, mat_S_Pi_r); CHKERRQ(ierr);

      ierr = KSPGetPC(ksp_S_Pi_r, &pc_S_Pi_r); CHKERRQ(ierr);
      ierr = PCSetType(pc_S_Pi_r, PCGAMG); CHKERRQ(ierr);

      ierr = KSPSetOptionsPrefix(ksp_S_Pi_r, "S_Pi_r_"); CHKERRQ(ierr);
      ierr = PCSetOptionsPrefix(pc_S_Pi_r, "S_Pi_r_"); CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp_S_Pi_r); CHKERRQ(ierr);
      ierr = PCSetFromOptions(pc_S_Pi_r); CHKERRQ(ierr);
    }
  }

  // First run, if benchmarking
  if (benchmark_mode) {
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1);
    CHKERRQ(ierr);
    ierr = VecZeroEntries(X); CHKERRQ(ierr);
    my_rt_start = MPI_Wtime();
    ierr = KSPSolve(ksp, rhs, X); CHKERRQ(ierr);
    my_rt = MPI_Wtime() - my_rt_start;
    ierr = MPI_Allreduce(MPI_IN_PLACE, &my_rt, 1, MPI_DOUBLE, MPI_MIN, comm);
    CHKERRQ(ierr);
    // Set maxits based on first iteration timing
    if (my_rt > 0.02) {
      ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 5);
      CHKERRQ(ierr);
    } else {
      ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 20);
      CHKERRQ(ierr);
    }
  }

  // Timed solve
  ierr = VecZeroEntries(X); CHKERRQ(ierr);
  ierr = PetscBarrier((PetscObject)ksp); CHKERRQ(ierr);

  // -- Performance logging
  ierr = PetscLogStageRegister("Solve Stage", &solve_stage); CHKERRQ(ierr);
  ierr = PetscLogStagePush(solve_stage); CHKERRQ(ierr);

  // -- Solve
  my_rt_start = MPI_Wtime();
  ierr = KSPSolve(ksp, rhs, X); CHKERRQ(ierr);
  my_rt = MPI_Wtime() - my_rt_start;

  // -- Performance logging
  ierr = PetscLogStagePop();

  // Output results
  {
    KSPType ksp_type;
    KSPConvergedReason reason;
    PCType pc_type;
    PetscReal rnorm;
    PetscInt its;
    ierr = KSPGetType(ksp, &ksp_type); CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(ksp, &reason); CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr);
    ierr = KSPGetResidualNorm(ksp, &rnorm); CHKERRQ(ierr);
    ierr = PCGetType(pc, &pc_type); CHKERRQ(ierr);
    if (!test_mode || reason < 0 || rnorm > 1e-8) {
      ierr = PetscPrintf(comm,
                         "  KSP:\n"
                         "    KSP Type                           : %s\n"
                         "    KSP Convergence                    : %s\n"
                         "    Total KSP Iterations               : %D\n"
                         "    Final rnorm                        : %e\n",
                         ksp_type, KSPConvergedReasons[reason], its,
                         (double)rnorm); CHKERRQ(ierr);
      ierr = PetscPrintf(comm,
                         "  BDDC:\n"
                         "    PC Type                            : %s\n",
                         pc_type); CHKERRQ(ierr);
    }
    if (!test_mode) {
      ierr = PetscPrintf(comm,"  Performance:\n"); CHKERRQ(ierr);
    }
    {
      PetscReal max_error;
      ierr = ComputeErrorMax(user_O, op_error, X, target,
                             &max_error); CHKERRQ(ierr);
      PetscReal tol = 5e-2;
      if (!test_mode || max_error > tol) {
        ierr = MPI_Allreduce(&my_rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, comm);
        CHKERRQ(ierr);
        ierr = MPI_Allreduce(&my_rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, comm);
        CHKERRQ(ierr);
        ierr = PetscPrintf(comm,
                           "    Pointwise Error (max)              : %e\n"
                           "    CG Solve Time                      : %g (%g) sec\n",
                           (double)max_error, rt_max, rt_min); CHKERRQ(ierr);
      }
    }
    if (benchmark_mode && (!test_mode)) {
      ierr = PetscPrintf(comm,
                         "    DoFs/Sec in CG                     : %g (%g) million\n",
                         1e-6*g_size*its/rt_max,
                         1e-6*g_size*its/rt_min);
      CHKERRQ(ierr);
    }
  }

  if (write_solution) {
    PetscViewer vtk_viewer_soln;

    ierr = PetscViewerCreate(comm, &vtk_viewer_soln); CHKERRQ(ierr);
    ierr = PetscViewerSetType(vtk_viewer_soln, PETSCVIEWERVTK); CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(vtk_viewer_soln, "solution.vtu"); CHKERRQ(ierr);
    ierr = VecView(X, vtk_viewer_soln); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vtk_viewer_soln); CHKERRQ(ierr);
  }

  // Cleanup
  ierr = VecDestroy(&X); CHKERRQ(ierr);
  ierr = VecDestroy(&X_loc); CHKERRQ(ierr);
  ierr = VecDestroy(&user_O->Y_loc); CHKERRQ(ierr);
  ierr = VecDestroy(&X_Pi); CHKERRQ(ierr);
  ierr = VecDestroy(&user_bddc->Y_Pi); CHKERRQ(ierr);
  ierr = VecDestroy(&X_Pi_loc); CHKERRQ(ierr);
  ierr = VecDestroy(&user_bddc->Y_Pi_loc); CHKERRQ(ierr);
  ierr = VecDestroy(&X_Pi_r_loc); CHKERRQ(ierr);
  ierr = VecDestroy(&user_bddc->Y_Pi_r_loc); CHKERRQ(ierr);
  ierr = VecDestroy(&rhs); CHKERRQ(ierr);
  ierr = VecDestroy(&rhs_loc); CHKERRQ(ierr);
  ierr = MatDestroy(&mat_O); CHKERRQ(ierr);
  ierr = MatDestroy(&mat_S_Pi); CHKERRQ(ierr);
  ierr = MatDestroy(&mat_S_Pi_r); CHKERRQ(ierr);
  ierr = PetscFree(user_O); CHKERRQ(ierr);
  ierr = PetscFree(user_bddc); CHKERRQ(ierr);
  ierr = CeedDataDestroy(0, ceed_data); CHKERRQ(ierr);
  ierr = CeedDataBDDCDestroy(ceed_data_bddc); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = DMDestroy(&dm_Pi); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp_S_Pi); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp_S_Pi_r); CHKERRQ(ierr);
  ierr = SNESDestroy(&snes_Pi); CHKERRQ(ierr);
  ierr = SNESDestroy(&snes_Pi_r); CHKERRQ(ierr);
  CeedVectorDestroy(&target);
  CeedQFunctionDestroy(&qf_error);
  CeedQFunctionDestroy(&qf_restrict);
  CeedQFunctionDestroy(&qf_prolong);
  CeedOperatorDestroy(&op_error);
  CeedDestroy(&ceed);
  return PetscFinalize();
}
