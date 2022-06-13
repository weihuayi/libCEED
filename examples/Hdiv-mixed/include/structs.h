#ifndef structs_h
#define structs_h

#include <ceed.h>
#include <petsc.h>

// Application context from user command line options
typedef struct AppCtx_ *AppCtx;
struct AppCtx_ {
  MPI_Comm          comm;
  // Degree of polynomial (1 only), extra quadrature pts
  PetscInt          degree;
  PetscInt          q_extra;
  // For applying traction BCs
  PetscInt          bc_pressure_count;
  PetscInt          bc_faces[16]; // face ID
  PetscScalar       bc_pressure_value[16];
  // Problem type arguments
  PetscFunctionList problems;
  char              problem_name[PETSC_MAX_PATH_LEN];

};

// libCEED data struct
typedef struct CeedData_ *CeedData;
struct CeedData_ {
  CeedBasis            basis_x, basis_u, basis_p, basis_u_face;
  CeedElemRestriction  elem_restr_x, elem_restr_u, elem_restr_U_i,
                       elem_restr_p;
  CeedQFunction        qf_residual, qf_jacobian, qf_error;
  CeedOperator         op_residual, op_jacobian, op_error;
  CeedVector           x_ceed, y_ceed, x_coord;
};

// 1) darcy2d
#ifndef PHYSICS_DARCY2D_STRUCT
#define PHYSICS_DARCY2D_STRUCT
typedef struct DARCY2DContext_ *DARCY2DContext;
struct DARCY2DContext_ {
  CeedScalar kappa;
};
#endif

// 2) darcy3d
#ifndef PHYSICS_DARCY3D_STRUCT
#define PHYSICS_DARCY3D_STRUCT
typedef struct DARCY3DContext_ *DARCY3DContext;
struct DARCY3DContext_ {
  CeedScalar kappa;
};
#endif

// 3) richard2d
#ifndef PHYSICS_RICHARD2D_STRUCT
#define PHYSICS_RICHARD2D_STRUCT
typedef struct RICHARD2DContext_ *RICHARD2DContext;
struct RICHARD2DContext_ {
  CeedScalar kappa;
};
#endif

// 4) richard3d
#ifndef PHYSICS_RICHARD3D_STRUCT
#define PHYSICS_RICHARD3D_STRUCT
typedef struct RICHARD3DContext_ *RICHARD3DContext;
struct RICHARD3DContext_ {
  CeedScalar kappa;
};
#endif

// Struct that contains all enums and structs used for the physics of all problems
typedef struct Physics_ *Physics;
struct Physics_ {
  DARCY2DContext            darcy2d_ctx;
  DARCY3DContext            darcy3d_ctx;
  RICHARD2DContext          richard2d_ctx;
  RICHARD3DContext          richard3d_ctx;
};

// PETSc operator contexts
typedef struct OperatorApplyContext_ *OperatorApplyContext;
struct OperatorApplyContext_ {
  MPI_Comm        comm;
  Vec             X_loc, Y_loc;
  CeedVector      x_ceed, y_ceed;
  CeedOperator    op_apply;
  DM              dm;
  Ceed            ceed;
};

// Problem specific data
typedef struct ProblemData_ *ProblemData;
struct ProblemData_ {
  CeedQFunctionUser force, residual, jacobian, error,
                    setup_true, bc_pressure;
  const char        *force_loc, *residual_loc, *jacobian_loc,
        *error_loc, *setup_true_loc, *bc_pressure_loc;
  CeedQuadMode      quadrature_mode;
  CeedInt           elem_node, dim, q_data_size_face;

};

#endif // structs_h