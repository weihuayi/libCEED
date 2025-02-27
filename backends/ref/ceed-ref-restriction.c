// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <string.h>
#include "ceed-ref.h"

//------------------------------------------------------------------------------
// Core ElemRestriction Apply Code
//------------------------------------------------------------------------------
static inline int CeedElemRestrictionApply_Ref_Core(CeedElemRestriction r,
    const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
    CeedInt start, CeedInt stop, CeedTransposeMode t_mode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  int ierr;
  CeedElemRestriction_Ref *impl;
  ierr = CeedElemRestrictionGetData(r, &impl); CeedChkBackend(ierr);
  const CeedScalar *uu;
  CeedScalar *vv;
  CeedInt num_elem, elem_size, v_offset;
  ierr = CeedElemRestrictionGetNumElements(r, &num_elem); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elem_size); CeedChkBackend(ierr);
  v_offset = start*blk_size*elem_size*num_comp;

  bool is_oriented;
  ierr = CeedElemRestrictionIsOriented(r, &is_oriented); CeedChkBackend(ierr);
  ierr = CeedVectorGetArrayRead(u, CEED_MEM_HOST, &uu); CeedChkBackend(ierr);
  if (t_mode == CEED_TRANSPOSE) {
    // Sum into for transpose mode, e-vec to l-vec
    ierr = CeedVectorGetArray(v, CEED_MEM_HOST, &vv); CeedChkBackend(ierr);
  } else {
    // Overwrite for notranspose mode, l-vec to e-vec
    ierr = CeedVectorGetArrayWrite(v, CEED_MEM_HOST, &vv); CeedChkBackend(ierr);
  }
  // Restriction from L-vector to E-vector
  // Perform: v = r * u
  if (t_mode == CEED_NOTRANSPOSE) {
    // No offsets provided, Identity Restriction
    if (!impl->offsets) {
      bool has_backend_strides;
      ierr = CeedElemRestrictionHasBackendStrides(r, &has_backend_strides);
      CeedChkBackend(ierr);
      if (has_backend_strides) {
        // CPU backend strides are {1, elem_size, elem_size*num_comp}
        // This if branch is left separate to allow better inlining
        for (CeedInt e = start*blk_size; e < stop*blk_size; e+=blk_size)
          CeedPragmaSIMD
          for (CeedInt k = 0; k < num_comp; k++)
            CeedPragmaSIMD
            for (CeedInt n = 0; n < elem_size; n++)
              CeedPragmaSIMD
              for (CeedInt j = 0; j < blk_size; j++)
                vv[e*elem_size*num_comp + (k*elem_size+n)*blk_size + j - v_offset]
                  = uu[n + k*elem_size +
                         CeedIntMin(e+j, num_elem-1)*elem_size*num_comp];
      } else {
        // User provided strides
        CeedInt strides[3];
        ierr = CeedElemRestrictionGetStrides(r, &strides); CeedChkBackend(ierr);
        for (CeedInt e = start*blk_size; e < stop*blk_size; e+=blk_size)
          CeedPragmaSIMD
          for (CeedInt k = 0; k < num_comp; k++)
            CeedPragmaSIMD
            for (CeedInt n = 0; n < elem_size; n++)
              CeedPragmaSIMD
              for (CeedInt j = 0; j < blk_size; j++)
                vv[e*elem_size*num_comp + (k*elem_size+n)*blk_size + j - v_offset]
                  = uu[n*strides[0] + k*strides[1] +
                                    CeedIntMin(e+j, num_elem-1)*strides[2]];
      }
    } else {
      // Offsets provided, standard or blocked restriction
      // vv has shape [elem_size, num_comp, num_elem], row-major
      // uu has shape [nnodes, num_comp]
      for (CeedInt e = start*blk_size; e < stop*blk_size; e+=blk_size)
        CeedPragmaSIMD
        for (CeedInt k = 0; k < num_comp; k++)
          CeedPragmaSIMD
          for (CeedInt i = 0; i < elem_size*blk_size; i++)
            vv[elem_size*(k*blk_size+num_comp*e) + i - v_offset]
              = uu[impl->offsets[i+elem_size*e] + k*comp_stride] *
                (is_oriented && impl->orient[i+elem_size*e] ? -1. : 1.);
    }
  } else {
    // Restriction from E-vector to L-vector
    // Performing v += r^T * u
    // No offsets provided, Identity Restriction
    if (!impl->offsets) {
      bool has_backend_strides;
      ierr = CeedElemRestrictionHasBackendStrides(r, &has_backend_strides);
      CeedChkBackend(ierr);
      if (has_backend_strides) {
        // CPU backend strides are {1, elem_size, elem_size*num_comp}
        // This if brach is left separate to allow better inlining
        for (CeedInt e = start*blk_size; e < stop*blk_size; e+=blk_size)
          CeedPragmaSIMD
          for (CeedInt k = 0; k < num_comp; k++)
            CeedPragmaSIMD
            for (CeedInt n = 0; n < elem_size; n++)
              CeedPragmaSIMD
              for (CeedInt j = 0; j < CeedIntMin(blk_size, num_elem-e); j++)
                vv[n + k*elem_size + (e+j)*elem_size*num_comp]
                += uu[e*elem_size*num_comp + (k*elem_size+n)*blk_size + j - v_offset];
      } else {
        // User provided strides
        CeedInt strides[3];
        ierr = CeedElemRestrictionGetStrides(r, &strides); CeedChkBackend(ierr);
        for (CeedInt e = start*blk_size; e < stop*blk_size; e+=blk_size)
          CeedPragmaSIMD
          for (CeedInt k = 0; k < num_comp; k++)
            CeedPragmaSIMD
            for (CeedInt n = 0; n < elem_size; n++)
              CeedPragmaSIMD
              for (CeedInt j = 0; j < CeedIntMin(blk_size, num_elem-e); j++)
                vv[n*strides[0] + k*strides[1] + (e+j)*strides[2]]
                += uu[e*elem_size*num_comp + (k*elem_size+n)*blk_size + j - v_offset];
      }
    } else {
      // Offsets provided, standard or blocked restriction
      // uu has shape [elem_size, num_comp, num_elem]
      // vv has shape [nnodes, num_comp]
      for (CeedInt e = start*blk_size; e < stop*blk_size; e+=blk_size)
        for (CeedInt k = 0; k < num_comp; k++)
          for (CeedInt i = 0; i < elem_size*blk_size; i+=blk_size)
            // Iteration bound set to discard padding elements
            for (CeedInt j = i; j < i+CeedIntMin(blk_size, num_elem-e); j++)
              vv[impl->offsets[j+e*elem_size] + k*comp_stride]
              += uu[elem_size*(k*blk_size+num_comp*e) + j - v_offset] *
                 (is_oriented && impl->orient[j+e*elem_size] ? -1. : 1.);
    }
  }
  ierr = CeedVectorRestoreArrayRead(u, &uu); CeedChkBackend(ierr);
  ierr = CeedVectorRestoreArray(v, &vv); CeedChkBackend(ierr);
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Apply - Common Sizes
//------------------------------------------------------------------------------
static int CeedElemRestrictionApply_Ref_110(CeedElemRestriction r,
    const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
    CeedInt start, CeedInt stop, CeedTransposeMode t_mode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 1, 1, comp_stride, start, stop,
         t_mode, u, v, request);
}

static int CeedElemRestrictionApply_Ref_111(CeedElemRestriction r,
    const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
    CeedInt start, CeedInt stop, CeedTransposeMode t_mode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 1, 1, 1, start, stop, t_mode,
         u, v, request);
}

static int CeedElemRestrictionApply_Ref_180(CeedElemRestriction r,
    const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
    CeedInt start, CeedInt stop, CeedTransposeMode t_mode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 1, 8, comp_stride, start, stop,
         t_mode, u, v, request);
}

static int CeedElemRestrictionApply_Ref_181(CeedElemRestriction r,
    const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
    CeedInt start, CeedInt stop, CeedTransposeMode t_mode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 1, 8, 1, start, stop, t_mode,
         u, v, request);
}

static int CeedElemRestrictionApply_Ref_310(CeedElemRestriction r,
    const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
    CeedInt start, CeedInt stop, CeedTransposeMode t_mode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 3, 1, comp_stride, start, stop,
         t_mode, u, v, request);
}

static int CeedElemRestrictionApply_Ref_311(CeedElemRestriction r,
    const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
    CeedInt start, CeedInt stop, CeedTransposeMode t_mode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 3, 1, 1, start, stop, t_mode,
         u, v, request);
}

static int CeedElemRestrictionApply_Ref_380(CeedElemRestriction r,
    const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
    CeedInt start, CeedInt stop, CeedTransposeMode t_mode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 3, 8, comp_stride, start, stop,
         t_mode, u, v, request);
}

static int CeedElemRestrictionApply_Ref_381(CeedElemRestriction r,
    const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
    CeedInt start, CeedInt stop, CeedTransposeMode t_mode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 3, 8, 1, start, stop, t_mode,
         u, v, request);
}

// LCOV_EXCL_START
static int CeedElemRestrictionApply_Ref_510(CeedElemRestriction r,
    const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
    CeedInt start, CeedInt stop, CeedTransposeMode t_mode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 5, 1, comp_stride, start, stop,
         t_mode, u, v, request);
}
// LCOV_EXCL_STOP

static int CeedElemRestrictionApply_Ref_511(CeedElemRestriction r,
    const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
    CeedInt start, CeedInt stop, CeedTransposeMode t_mode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 5, 1, 1, start, stop, t_mode,
         u, v, request);
}

// LCOV_EXCL_START
static int CeedElemRestrictionApply_Ref_580(CeedElemRestriction r,
    const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
    CeedInt start, CeedInt stop, CeedTransposeMode t_mode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 5, 8, comp_stride, start, stop,
         t_mode, u, v, request);
}
// LCOV_EXCL_STOP

static int CeedElemRestrictionApply_Ref_581(CeedElemRestriction r,
    const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
    CeedInt start, CeedInt stop, CeedTransposeMode t_mode, CeedVector u,
    CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 5, 8, 1, start, stop, t_mode,
         u, v, request);
}

//------------------------------------------------------------------------------
// ElemRestriction Apply
//------------------------------------------------------------------------------
static int CeedElemRestrictionApply_Ref(CeedElemRestriction r,
                                        CeedTransposeMode t_mode, CeedVector u,
                                        CeedVector v, CeedRequest *request) {
  int ierr;
  CeedInt num_blk, blk_size, num_comp, comp_stride;
  ierr = CeedElemRestrictionGetNumBlocks(r, &num_blk); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetBlockSize(r, &blk_size); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetNumComponents(r, &num_comp); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetCompStride(r, &comp_stride); CeedChkBackend(ierr);
  CeedElemRestriction_Ref *impl;
  ierr = CeedElemRestrictionGetData(r, &impl); CeedChkBackend(ierr);

  return impl->Apply(r, num_comp, blk_size, comp_stride, 0, num_blk, t_mode, u, v,
                     request);
}

//------------------------------------------------------------------------------
// ElemRestriction Apply Block
//------------------------------------------------------------------------------
static int CeedElemRestrictionApplyBlock_Ref(CeedElemRestriction r,
    CeedInt block, CeedTransposeMode t_mode, CeedVector u, CeedVector v,
    CeedRequest *request) {
  int ierr;
  CeedInt blk_size, num_comp, comp_stride;
  ierr = CeedElemRestrictionGetBlockSize(r, &blk_size); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetNumComponents(r, &num_comp); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetCompStride(r, &comp_stride); CeedChkBackend(ierr);
  CeedElemRestriction_Ref *impl;
  ierr = CeedElemRestrictionGetData(r, &impl); CeedChkBackend(ierr);

  return impl->Apply(r, num_comp, blk_size, comp_stride, block, block+1, t_mode,
                     u, v, request);
}

//------------------------------------------------------------------------------
// ElemRestriction Get Offsets
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetOffsets_Ref(CeedElemRestriction rstr,
    CeedMemType mem_type, const CeedInt **offsets) {
  int ierr;
  CeedElemRestriction_Ref *impl;
  ierr = CeedElemRestrictionGetData(rstr, &impl); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(rstr, &ceed); CeedChkBackend(ierr);

  if (mem_type != CEED_MEM_HOST)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Can only provide to HOST memory");
  // LCOV_EXCL_STOP

  *offsets = impl->offsets;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Destroy
//------------------------------------------------------------------------------
static int CeedElemRestrictionDestroy_Ref(CeedElemRestriction r) {
  int ierr;
  CeedElemRestriction_Ref *impl;
  ierr = CeedElemRestrictionGetData(r, &impl); CeedChkBackend(ierr);

  ierr = CeedFree(&impl->offsets_allocated); CeedChkBackend(ierr);
  ierr = CeedFree(&impl); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Create
//------------------------------------------------------------------------------
int CeedElemRestrictionCreate_Ref(CeedMemType mem_type, CeedCopyMode copy_mode,
                                  const CeedInt *offsets,
                                  CeedElemRestriction r) {
  int ierr;
  CeedElemRestriction_Ref *impl;
  CeedInt num_elem, elem_size, num_blk, blk_size, num_comp, comp_stride;
  ierr = CeedElemRestrictionGetNumElements(r, &num_elem); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elem_size); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetNumBlocks(r, &num_blk); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetBlockSize(r, &blk_size); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetNumComponents(r, &num_comp); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetCompStride(r, &comp_stride); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChkBackend(ierr);

  if (mem_type != CEED_MEM_HOST)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Only MemType = HOST supported");
  // LCOV_EXCL_STOP
  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);

  // Offsets data
  bool is_strided;
  ierr = CeedElemRestrictionIsStrided(r, &is_strided); CeedChkBackend(ierr);
  if (!is_strided) {
    // Check indices for ref or memcheck backends
    Ceed parent_ceed = ceed, curr_ceed = NULL;
    while (parent_ceed != curr_ceed) {
      curr_ceed = parent_ceed;
      ierr = CeedGetParent(curr_ceed, &parent_ceed); CeedChkBackend(ierr);
    }
    const char *resource;
    ierr = CeedGetResource(parent_ceed, &resource); CeedChkBackend(ierr);
    if (!strcmp(resource, "/cpu/self/ref/serial") ||
        !strcmp(resource, "/cpu/self/ref/blocked") ||
        !strcmp(resource, "/cpu/self/memcheck/serial") ||
        !strcmp(resource, "/cpu/self/memcheck/blocked")) {
      CeedSize l_size;
      ierr = CeedElemRestrictionGetLVectorSize(r, &l_size); CeedChkBackend(ierr);

      for (CeedInt i = 0; i < num_elem*elem_size; i++)
        if (offsets[i] < 0 || l_size <= offsets[i] + (num_comp - 1) * comp_stride)
          // LCOV_EXCL_START
          return CeedError(ceed, CEED_ERROR_BACKEND,
                           "Restriction offset %" CeedInt_FMT " (%" CeedInt_FMT ") out of range "
                           "[0, %" CeedInt_FMT "]", i, offsets[i], l_size);
      // LCOV_EXCL_STOP
    }

    // Copy data
    switch (copy_mode) {
    case CEED_COPY_VALUES:
      ierr = CeedMalloc(num_elem*elem_size, &impl->offsets_allocated);
      CeedChkBackend(ierr);
      memcpy(impl->offsets_allocated, offsets,
             num_elem * elem_size * sizeof(offsets[0]));
      impl->offsets = impl->offsets_allocated;
      break;
    case CEED_OWN_POINTER:
      impl->offsets_allocated = (CeedInt *)offsets;
      impl->offsets = impl->offsets_allocated;
      break;
    case CEED_USE_POINTER:
      impl->offsets = offsets;
    }
  }

  ierr = CeedElemRestrictionSetData(r, impl); CeedChkBackend(ierr);
  CeedInt layout[3] = {1, elem_size, elem_size*num_comp};
  ierr = CeedElemRestrictionSetELayout(r, layout); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply",
                                CeedElemRestrictionApply_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "ApplyBlock",
                                CeedElemRestrictionApplyBlock_Ref);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "GetOffsets",
                                CeedElemRestrictionGetOffsets_Ref);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy",
                                CeedElemRestrictionDestroy_Ref); CeedChkBackend(ierr);

  // Set apply function based upon num_comp, blk_size, and comp_stride
  CeedInt idx = -1;
  if (blk_size < 10)
    idx = 100*num_comp + 10*blk_size + (comp_stride == 1);
  switch (idx) {
  case 110:
    impl->Apply = CeedElemRestrictionApply_Ref_110;
    break;
  case 111:
    impl->Apply = CeedElemRestrictionApply_Ref_111;
    break;
  case 180:
    impl->Apply = CeedElemRestrictionApply_Ref_180;
    break;
  case 181:
    impl->Apply = CeedElemRestrictionApply_Ref_181;
    break;
  case 310:
    impl->Apply = CeedElemRestrictionApply_Ref_310;
    break;
  case 311:
    impl->Apply = CeedElemRestrictionApply_Ref_311;
    break;
  case 380:
    impl->Apply = CeedElemRestrictionApply_Ref_380;
    break;
  case 381:
    impl->Apply = CeedElemRestrictionApply_Ref_381;
    break;
  // LCOV_EXCL_START
  case 510:
    impl->Apply = CeedElemRestrictionApply_Ref_510;
    break;
  // LCOV_EXCL_STOP
  case 511:
    impl->Apply = CeedElemRestrictionApply_Ref_511;
    break;
  // LCOV_EXCL_START
  case 580:
    impl->Apply = CeedElemRestrictionApply_Ref_580;
    break;
  // LCOV_EXCL_STOP
  case 581:
    impl->Apply = CeedElemRestrictionApply_Ref_581;
    break;
  default:
    impl->Apply = CeedElemRestrictionApply_Ref_Core;
    break;
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Create Oriented
//------------------------------------------------------------------------------
int CeedElemRestrictionCreateOriented_Ref(CeedMemType mem_type,
    CeedCopyMode copy_mode,
    const CeedInt *offsets, const bool *orient,
    CeedElemRestriction r) {
  int ierr;
  CeedElemRestriction_Ref *impl;
  CeedInt num_elem, elem_size;
  // Set up for normal restriction with explicit offsets. This sets up dispatch to
  // CeedElemRestrictionApply_Ref_* and manages the impl->offsets array copy/allocation.
  ierr = CeedElemRestrictionCreate_Ref(mem_type, copy_mode, offsets, r);
  CeedChkBackend(ierr);

  ierr = CeedElemRestrictionGetData(r, &impl); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetNumElements(r, &num_elem); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elem_size); CeedChkBackend(ierr);
  switch (copy_mode) {
  case CEED_COPY_VALUES:
    ierr = CeedMalloc(num_elem * elem_size, &impl->orient_allocated);
    CeedChkBackend(ierr);
    memcpy(impl->orient_allocated, orient,
           num_elem * elem_size * sizeof(orient[0]));
    impl->orient = impl->orient_allocated;
    break;
  case CEED_OWN_POINTER:
    impl->orient_allocated = (bool *)orient;
    impl->orient = impl->orient_allocated;
    break;
  case CEED_USE_POINTER:
    impl->orient = orient;
  }
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
