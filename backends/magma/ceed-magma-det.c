// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <string.h>
#include <stdlib.h>
#include "ceed-magma.h"

CEED_INTERN int CeedInit_Magma_Det(const char *resource, Ceed ceed) {
  int ierr;
  const int nrc = 18; // number of characters in resource
  if (strncmp(resource, "/gpu/cuda/magma/det", nrc)
      && strncmp(resource, "/gpu/hip/magma/det", nrc))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Magma backend cannot use resource: %s", resource);
  // LCOV_EXCL_STOP
  ierr = CeedSetDeterministic(ceed, true); CeedChkBackend(ierr);

  Ceed_Magma *data;
  ierr = CeedCalloc(sizeof(Ceed_Magma), &data); CeedChkBackend(ierr);
  ierr = CeedSetData(ceed, data); CeedChkBackend(ierr);

  // get/set device ID
  const char *device_spec = strstr(resource, ":device_id=");
  const int deviceID = (device_spec) ? atoi(device_spec+11) : -1;

  int currentDeviceID;
  magma_getdevice(&currentDeviceID);
  if (deviceID >= 0 && currentDeviceID != deviceID) {
    magma_setdevice(deviceID);
    currentDeviceID = deviceID;
  }
  // create a queue that uses the null stream
  data->device = currentDeviceID;

  // Create reference CEED that implementation will be dispatched
  //   through unless overridden
  Ceed ceedref;
  #ifdef CEED_MAGMA_USE_HIP
  CeedInit("/gpu/hip/magma", &ceedref);
  #else
  CeedInit("/gpu/cuda/magma", &ceedref);
  #endif
  ierr = CeedSetDelegate(ceed, ceedref); CeedChkBackend(ierr);

  // Create reference CEED for restriction
  Ceed restrictionceedref;
  #ifdef CEED_MAGMA_USE_HIP
  CeedInit("/gpu/hip/ref", &restrictionceedref);
  #else
  CeedInit("/gpu/cuda/ref", &restrictionceedref);
  #endif
  ierr = CeedSetObjectDelegate(ceed, restrictionceedref, "ElemRestriction");
  CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

CEED_INTERN int CeedRegister_Magma_Det(void) {
  #ifdef CEED_MAGMA_USE_HIP
  return CeedRegister("/gpu/hip/magma/det", CeedInit_Magma_Det, 125);
  #else
  return CeedRegister("/gpu/cuda/magma/det", CeedInit_Magma_Det, 125);
  #endif
}
