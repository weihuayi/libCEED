// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Geometric factors (2D) using PETSc

#ifndef face_geo_2d_h
#define face_geo_2d_h

#include <math.h>
// *****************************************************************************
// This QFunction sets up the geometric factor required for integration when
//   reference coordinates are in 1D and the physical coordinates are in 2D
//
// Reference (parent) 1D coordinates: X
// Physical (current) 2D coordinates: x
// Change of coordinate vector:
//           J1 = dx_1/dX
//           J2 = dx_2/dX
//
// detJb is the magnitude of (J1,J2)
//
// All quadrature data is stored in 3 field vector of quadrature data.
//
// We require the determinant of the Jacobian to properly compute integrals of
//   the form: int( u v )
//
// Stored: w detJb
//   in q_data_face[0]
//
// Normal vector is given by the cross product of (J1,J2)/detJ and ẑ
//
// Stored: (J1,J2,0) x (0,0,1) / detJb
//   in q_data_face[1:2] as
//   (detJb^-1) * [ J2 ]
//                [-J1 ]
//
// *****************************************************************************
CEED_QFUNCTION(SetupFaceGeo2D)(void *ctx, CeedInt Q,
                               const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*J)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*w) = in[1];
  // Outputs
  CeedScalar (*q_data_face)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    const CeedScalar J1 = J[0][i];
    const CeedScalar J2 = J[1][i];

    const CeedScalar detJb = sqrt(J1*J1 + J2*J2);

    q_data_face[0][i] = w[i] * detJb;
    q_data_face[1][i] = J2 / detJb;
    q_data_face[2][i] = -J1 / detJb;
  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************

#endif // face_geo_2d_h
