// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
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
#include <ceed-backend.h>
#include "ceed-cuda-gen.h"
#include <iostream>
#include <sstream>
#include "../cuda/ceed-cuda.h"
#include "../cuda-reg/ceed-cuda-reg.h"
#include "../cuda-shared/ceed-cuda-shared.h"
#include <string.h>

static const char *deviceFunctions = QUOTE(

typedef struct { const CeedScalar* in[16]; CeedScalar* out[16]; } CudaFields;
typedef struct { CeedInt* in[16]; CeedInt* out[16]; } CudaFieldsInt;

typedef struct {
  CeedInt tidx;
  CeedInt tidy;
  CeedInt tidz;
  CeedInt tid;
  CeedScalar* slice;
} BackendData;

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old =
      atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val +
                                     __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN
    // (since NaN != NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif // __CUDA_ARCH__ < 600

template <int P, int Q>
inline __device__ void loadMatrix(BackendData& data, const CeedScalar* d_B, CeedScalar* B) {
  for(int i=data.tid; i<P*Q; i+=blockDim.x*blockDim.y*blockDim.z) {
    B[i] = d_B[i];
  }
  __syncthreads;
}

//****
// 1D
template <int NCOMP, int P1d>
inline __device__ void readDofs1d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* d_u, CeedScalar* r_u) {
  if (data.tidx<P1d)
  {
    const CeedInt dof = data.tidx;
    const CeedInt ind = indices ? indices[dof + elem * P1d] : dof + elem * P1d;
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      r_u[comp] = d_u[ind + ndofs * comp];
    }
  }
}

template <int NCOMP, int P1d>
inline __device__ void readDofsTranspose1d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* d_u, CeedScalar* r_u) {
  if (data.tidx<P1d)
  {
    const CeedInt dof = data.tidx;
    const CeedInt ind = indices ? indices[dof + elem * P1d] : dof + elem * P1d;
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      r_u[comp] = d_u[ind * NCOMP + comp];
    }
  }
}

template <int NCOMP, int Q1d>
inline __device__ void readQuads1d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
  const CeedInt dof = data.tidx;
  const CeedInt ind = dof + elem * Q1d;
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    r_u[comp] = d_u[ind + nquads * comp];
  }
}

template <int NCOMP, int Q1d>
inline __device__ void readQuadsTranspose1d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
  const CeedInt dof = data.tidx;
  const CeedInt ind = dof + elem * Q1d;
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    r_u[comp] = d_u[ind * NCOMP + comp];
  }
}

template <int NCOMP, int P1d>
inline __device__ void writeDofs1d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* r_v, CeedScalar* d_v) {
  if (data.tidx<P1d)
  {
    const CeedInt dof = data.tidx;
    const CeedInt ind = indices ? indices[dof + elem * P1d] : dof + elem * P1d;
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      atomicAdd(&d_v[ind + ndofs * comp], r_v[comp]);
    }
  }
}

template <int NCOMP, int P1d>
inline __device__ void writeDofsTranspose1d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* r_v, CeedScalar* d_v) {
  if (data.tidx<P1d)
  {
    const CeedInt dof = data.tidx;
    const CeedInt ind = indices ? indices[dof + elem * P1d] : dof + elem * P1d;
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      atomicAdd(&d_v[ind * NCOMP + comp], r_v[comp]);
    }
  }
}

template <int NCOMP, int Q1d>
inline __device__ void writeQuads1d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* r_v, CeedScalar* d_v) {
  const CeedInt dof = data.tidx;
  const CeedInt ind = dof + elem * Q1d;
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    d_v[ind + nquads * comp] = r_v[comp];
  }
}

template <int NCOMP, int Q1d>
inline __device__ void writeQuadsTranspose1d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* r_v, CeedScalar* d_v) {
  const CeedInt dof = data.tidx;
  const CeedInt ind = dof + elem * Q1d;
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    d_v[ind * NCOMP + comp] = r_v[comp];
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractX1d(BackendData& data,
                                   const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.tidx] = *U;
  __syncthreads();
  *V = 0.0;
  for (int i = 0; i < P1d; ++i) {
    *V += B[i + data.tidx*P1d] * data.slice[i];//contract x direction
  }
  __syncthreads();
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractTransposeX1d(BackendData& data,
                                            const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.tidx] = *U;
  __syncthreads();
  *V = 0.0;
  for (int i = 0; i < Q1d; ++i) {
    *V += B[data.tidx + i*P1d] * data.slice[i];//contract x direction
  }
  __syncthreads();
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void interp1d(BackendData& data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                CeedScalar *__restrict__ r_V) {
  for(int comp=0; comp<NCOMP; comp++) {
    ContractX1d<NCOMP,P1d,Q1d>(data, r_U+comp, c_B, r_V+comp);
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void interpTranspose1d(BackendData& data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                CeedScalar *__restrict__ r_V) {
  for(int comp=0; comp<NCOMP; comp++) {
    ContractTransposeX1d<NCOMP,P1d,Q1d>(data, r_U+comp, c_B, r_V+comp);
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void grad1d(BackendData& data, const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
  for(int comp=0; comp<NCOMP; comp++) {
    ContractX1d<NCOMP,P1d,Q1d>(data, r_U+comp, c_G, r_V+comp);
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void gradTranspose1d(BackendData& data, const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
  for(int comp=0; comp<NCOMP; comp++) {
    ContractTransposeX1d<NCOMP,P1d,Q1d>(data, r_U+comp, c_G, r_V+comp);
  }
}

//****
// 2D
template <int NCOMP, int P1d>
inline __device__ void readDofs2d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* d_u, CeedScalar* r_u) {
  if (data.tidx<P1d && data.tidy<P1d)
  {
    const CeedInt dof = data.tidx + data.tidy*P1d;
    const CeedInt ind = indices ? indices[dof + elem * P1d*P1d] : dof + elem * P1d*P1d;
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      r_u[comp] = d_u[ind + ndofs * comp];
    }
  }
}

template <int NCOMP, int P1d>
inline __device__ void readDofsTranspose2d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* d_u, CeedScalar* r_u) {
  if (data.tidx<P1d && data.tidy<P1d)
  {
    const CeedInt dof = data.tidx + data.tidy*P1d;
    const CeedInt ind = indices ? indices[dof + elem * P1d*P1d] : dof + elem * P1d*P1d;
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      r_u[comp] = d_u[ind * NCOMP + comp];
    }
  }
}

template <int NCOMP, int Q1d>
inline __device__ void readQuads2d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
  const CeedInt dof = data.tidx + data.tidy*Q1d;
  const CeedInt ind = dof + elem * Q1d*Q1d;
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    r_u[comp] = d_u[ind + nquads * comp];
  }
}

template <int NCOMP, int Q1d>
inline __device__ void readQuadsTranspose2d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
  const CeedInt dof = data.tidx + data.tidy*Q1d;
  const CeedInt ind = dof + elem * Q1d*Q1d;
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    r_u[comp] = d_u[ind * NCOMP + comp];
  }
}

template <int NCOMP, int P1d>
inline __device__ void writeDofs2d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* r_v, CeedScalar* d_v) {
  if (data.tidx<P1d && data.tidy<P1d)
  {
    const CeedInt dof = data.tidx + data.tidy*P1d;
    const CeedInt ind = indices ? indices[dof + elem * P1d*P1d] : dof + elem * P1d*P1d;
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      atomicAdd(&d_v[ind + ndofs * comp], r_v[comp]);
    }
  }
}

template <int NCOMP, int P1d>
inline __device__ void writeDofsTranspose2d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* r_v, CeedScalar* d_v) {
  if (data.tidx<P1d && data.tidy<P1d)
  {
    const CeedInt dof = data.tidx + data.tidy*P1d;
    const CeedInt ind = indices ? indices[dof + elem * P1d*P1d] : dof + elem * P1d*P1d;
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      atomicAdd(&d_v[ind * NCOMP + comp], r_v[comp]);
    }
  }
}

template <int NCOMP, int Q1d>
inline __device__ void writeQuads2d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* r_v, CeedScalar* d_v) {
  const CeedInt dof = data.tidx + data.tidy*Q1d;
  const CeedInt ind = dof + elem * Q1d*Q1d;
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    d_v[ind + nquads * comp] = r_v[comp];
  }
}

template <int NCOMP, int Q1d>
inline __device__ void writeQuadsTranspose2d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* r_v, CeedScalar* d_v) {
  const CeedInt dof = data.tidx + data.tidy*Q1d;
  const CeedInt ind = dof + elem * Q1d*Q1d;
  for(CeedInt comp = 0; comp < NCOMP; ++comp) {
    d_v[ind * NCOMP + comp] = r_v[comp];
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractX2d(BackendData& data,
                                   const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.tidx+data.tidy*Q1d] = *U;
  __syncthreads();
  *V = 0.0;
  for (int i = 0; i < P1d; ++i) {
    *V += B[i + data.tidx*P1d] * data.slice[i + data.tidy*Q1d];//contract x direction
  }
  __syncthreads();
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractY2d(BackendData& data,
                                   const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.tidx+data.tidy*Q1d] = *U;
  __syncthreads();
  *V = 0.0;
  for (int i = 0; i < P1d; ++i) {
    *V += B[i + data.tidy*P1d] * data.slice[data.tidx + i*Q1d];//contract y direction
  }
  __syncthreads();
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractYTranspose2d(BackendData& data,
                                            const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.tidx+data.tidy*Q1d] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.tidy<P1d) {
    for (int i = 0; i < Q1d; ++i) {
      *V += B[data.tidy + i*P1d] * data.slice[data.tidx + i*Q1d];//contract y direction
    }
  }
  __syncthreads();
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractXTranspose2d(BackendData& data,
                                            const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.tidx+data.tidy*Q1d] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.tidx<P1d) {
    for (int i = 0; i < Q1d; ++i) {
      *V += B[data.tidx + i*P1d] * data.slice[i + data.tidy*Q1d];//contract x direction
    }
  }
  __syncthreads();
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractXTransposeAdd2d(BackendData& data,
                                            const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.tidx+data.tidy*Q1d] = *U;
  __syncthreads();
  if (data.tidx<P1d) {
    for (int i = 0; i < Q1d; ++i) {
      *V += B[data.tidx + i*P1d] * data.slice[i + data.tidy*Q1d];//contract x direction
    }
  }
  __syncthreads();
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void interp2d(BackendData& data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                CeedScalar *__restrict__ r_V) {
  CeedScalar r_t[1];
  for(int comp=0; comp<NCOMP; comp++) {
    ContractX2d<NCOMP,P1d,Q1d>(data, r_U+comp, c_B, r_t);
    ContractY2d<NCOMP,P1d,Q1d>(data, r_t, c_B, r_V+comp);
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void interpTranspose2d(BackendData& data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                CeedScalar *__restrict__ r_V) {
  CeedScalar r_t[1];
  for(int comp=0; comp<NCOMP; comp++) {
    ContractYTranspose2d<NCOMP,P1d,Q1d>(data, r_U+comp, c_B, r_t);
    ContractXTranspose2d<NCOMP,P1d,Q1d>(data, r_t, c_B, r_V+comp);
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void grad2d(BackendData& data, const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
  CeedScalar r_t[1];
  for(int comp=0; comp<NCOMP; comp++) {
    ContractX2d<NCOMP,P1d,Q1d>(data, r_U+comp, c_G, r_t);
    ContractY2d<NCOMP,P1d,Q1d>(data, r_t, c_B, r_V+comp+0*NCOMP);
    ContractX2d<NCOMP,P1d,Q1d>(data, r_U+comp, c_B, r_t);
    ContractY2d<NCOMP,P1d,Q1d>(data, r_t, c_G, r_V+comp+1*NCOMP);
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void gradTranspose2d(BackendData& data, const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
  CeedScalar r_t[1];
  for(int comp=0; comp<NCOMP; comp++) {
    ContractYTranspose2d<NCOMP,P1d,Q1d>(data, r_U+comp+0*NCOMP, c_B, r_t);
    ContractXTranspose2d<NCOMP,P1d,Q1d>(data, r_t, c_G, r_V+comp);
    ContractYTranspose2d<NCOMP,P1d,Q1d>(data, r_U+comp+1*NCOMP, c_G, r_t);
    ContractXTransposeAdd2d<NCOMP,P1d,Q1d>(data, r_t, c_B, r_V+comp);
  }
}

//****
// 3D
template <int NCOMP, int P1d>
inline __device__ void readDofs3d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* d_u, CeedScalar* r_u) {
  if (data.tidx<P1d && data.tidy<P1d) {
    for (CeedInt z = 0; z < P1d; ++z) {
      const CeedInt dof = data.tidx + data.tidy*P1d + z*P1d*P1d;
      const CeedInt ind = indices ? indices[dof + elem * P1d*P1d*P1d] : dof + elem * P1d*P1d*P1d;
      for(CeedInt comp = 0; comp < NCOMP; ++comp) {
        r_u[z+comp*P1d] = d_u[ind + ndofs * comp];
      }
    }
  }
}

template <int NCOMP, int P1d>
inline __device__ void readDofsTranspose3d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* d_u, CeedScalar* r_u) {
  if (data.tidx<P1d && data.tidy<P1d) {
    for (CeedInt z = 0; z < P1d; ++z) {
      const CeedInt dof = data.tidx + data.tidy*P1d + z*P1d*P1d;
      const CeedInt ind = indices ? indices[dof + elem * P1d*P1d*P1d] : dof + elem * P1d*P1d*P1d;
      for(CeedInt comp = 0; comp < NCOMP; ++comp) {
        r_u[z+comp*P1d] = d_u[ind * NCOMP + comp];
      }
    }
  }
}

template <int NCOMP, int Q1d>
inline __device__ void readQuads3d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
  for(CeedInt z=0; z < Q1d; ++z) {
    const CeedInt dof = data.tidx + data.tidy*Q1d + z*Q1d*Q1d;
    const CeedInt ind = dof + elem * Q1d*Q1d*Q1d;
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      r_u[z+comp*Q1d] = d_u[ind + nquads * comp];
    }
  }
}

template <int NCOMP, int Q1d>
inline __device__ void readQuadsTranspose3d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* d_u, CeedScalar* r_u) {
  for(CeedInt z=0; z < Q1d; ++z) {
    const CeedInt dof = data.tidx + data.tidy*Q1d + z*Q1d*Q1d;
    const CeedInt ind = dof + elem * Q1d*Q1d*Q1d;
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      r_u[z+comp*Q1d] = d_u[ind * NCOMP + comp];
    }
  }
}

template <int NCOMP, int P1d>
inline __device__ void writeDofs3d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* r_v, CeedScalar* d_v) {
  if (data.tidx<P1d && data.tidy<P1d) {
    for (CeedInt z = 0; z < P1d; ++z) {
      const CeedInt dof = data.tidx + data.tidy*P1d + z*P1d*P1d;
      const CeedInt ind = indices ? indices[dof + elem * P1d*P1d*P1d] : dof + elem * P1d*P1d*P1d;
      for(CeedInt comp = 0; comp < NCOMP; ++comp) {
        atomicAdd(&d_v[ind + ndofs * comp], r_v[z+comp*P1d]);
      }
    }
  }
}

template <int NCOMP, int P1d>
inline __device__ void writeDofsTranspose3d(BackendData& data, const CeedInt ndofs, const CeedInt elem, const CeedInt* indices, const CeedScalar* r_v, CeedScalar* d_v) {
  if (data.tidx<P1d && data.tidy<P1d) {
    for (CeedInt z = 0; z < P1d; ++z) {
      const CeedInt dof = data.tidx + data.tidy*P1d + z*P1d*P1d;
      const CeedInt ind = indices ? indices[dof + elem * P1d*P1d*P1d] : dof + elem * P1d*P1d*P1d;
      for(CeedInt comp = 0; comp < NCOMP; ++comp) {
        atomicAdd(&d_v[ind * NCOMP + comp], r_v[z+comp*P1d]);
      }
    }
  }
}

template <int NCOMP, int Q1d>
inline __device__ void writeQuads3d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* r_v, CeedScalar* d_v) {
  for(CeedInt z=0; z < Q1d; ++z) {
    const CeedInt dof = data.tidx + data.tidy*Q1d + z*Q1d*Q1d;
    const CeedInt ind = dof + elem * Q1d*Q1d*Q1d;
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      d_v[ind + nquads * comp] = r_v[z+comp*Q1d];
    }
  }
}

template <int NCOMP, int Q1d>
inline __device__ void writeQuadsTranspose3d(BackendData& data, const CeedInt nquads, const CeedInt elem, const CeedScalar* r_v, CeedScalar* d_v) {
  for(CeedInt z=0; z < Q1d; ++z) {
    const CeedInt dof = data.tidx + data.tidy*Q1d + z*Q1d*Q1d;
    const CeedInt ind = dof + elem * Q1d*Q1d*Q1d;
    for(CeedInt comp = 0; comp < NCOMP; ++comp) {
      d_v[ind * NCOMP + comp] = r_v[z+comp*Q1d];
    }
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractX3d(BackendData& data,
                                   const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (int k = 0; k < P1d; ++k) {
    data.slice[data.tidx+data.tidy*Q1d] = U[k];
    __syncthreads();
    V[k] = 0.0;
    for (int i = 0; i < P1d; ++i) {
      V[k] += B[i + data.tidx*P1d] * data.slice[i + data.tidy*Q1d];//contract x direction
    }
    __syncthreads();
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractY3d(BackendData& data,
                                   const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (int k = 0; k < P1d; ++k) {
    data.slice[data.tidx+data.tidy*Q1d] = U[k];
    __syncthreads();
    V[k] = 0.0;
    for (int i = 0; i < P1d; ++i) {
      V[k] += B[i + data.tidy*P1d] * data.slice[data.tidx + i*Q1d];//contract y direction
    }
    __syncthreads();
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractZ3d(BackendData& data,
                                   const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (int k = 0; k < Q1d; ++k) {
    V[k] = 0.0;
    for (int i = 0; i < P1d; ++i) {
      V[k] += B[i + k*P1d] * U[i];//contract z direction
    }
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractTransposeZ3d(BackendData& data,
    const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (int k = 0; k < Q1d; ++k) {
    V[k] = 0.0;
    if (k<P1d) {
      for (int i = 0; i < Q1d; ++i) {
        V[k] += B[k + i*P1d] * U[i];//contract z direction
      }
    }
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractTransposeY3d(BackendData& data,
    const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (int k = 0; k < P1d; ++k) {
    data.slice[data.tidx+data.tidy*Q1d] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (data.tidy<P1d) {
      for (int i = 0; i < Q1d; ++i) {
        V[k] += B[data.tidy + i*P1d] * data.slice[data.tidx + i*Q1d];//contract y direction
      }
    }
    __syncthreads();
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractTransposeX3d(BackendData& data,
    const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (int k = 0; k < P1d; ++k) {
    data.slice[data.tidx+data.tidy*Q1d] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (data.tidx<P1d) {
      for (int i = 0; i < Q1d; ++i) {
        V[k] += B[data.tidx + i*P1d] * data.slice[i + data.tidy*Q1d];//contract x direction
      }
    }
    __syncthreads();
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void ContractTransposeAddX3d(BackendData& data,
    const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (int k = 0; k < P1d; ++k) {
    data.slice[data.tidx+data.tidy*Q1d] = U[k];
    __syncthreads();
    if (data.tidx<P1d) {
      for (int i = 0; i < Q1d; ++i) {
        V[k] += B[data.tidx + i*P1d] * data.slice[i + data.tidy*Q1d];//contract x direction
      }
    }
    __syncthreads();
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void interp3d(BackendData& data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                CeedScalar *__restrict__ r_V) {
  CeedScalar r_t1[Q1d];
  CeedScalar r_t2[Q1d];
  for(int comp=0; comp<NCOMP; comp++) {
    ContractX3d<NCOMP,P1d,Q1d>(data, r_U+comp*P1d, c_B, r_t1);
    ContractY3d<NCOMP,P1d,Q1d>(data, r_t1, c_B, r_t2);
    ContractZ3d<NCOMP,P1d,Q1d>(data, r_t2, c_B, r_V+comp*Q1d);
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void interpTranspose3d(BackendData& data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                CeedScalar *__restrict__ r_V) {
  CeedScalar r_t1[Q1d];
  CeedScalar r_t2[Q1d];
  for(int comp=0; comp<NCOMP; comp++) {
    ContractTransposeZ3d<NCOMP,P1d,Q1d>(data, r_U+comp*Q1d, c_B, r_t1);
    ContractTransposeY3d<NCOMP,P1d,Q1d>(data, r_t1, c_B, r_t2);
    ContractTransposeX3d<NCOMP,P1d,Q1d>(data, r_t2, c_B, r_V+comp*P1d);
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void grad3d(BackendData& data, const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
  CeedScalar r_t1[Q1d];
  CeedScalar r_t2[Q1d];
  for(int comp=0; comp<NCOMP; comp++) {
    ContractX3d<NCOMP,P1d,Q1d>(data, r_U+comp*P1d, c_G, r_t1);
    ContractY3d<NCOMP,P1d,Q1d>(data, r_t1, c_B, r_t2);
    ContractZ3d<NCOMP,P1d,Q1d>(data, r_t2, c_B, r_V+comp*Q1d+0*NCOMP*Q1d);
    ContractX3d<NCOMP,P1d,Q1d>(data, r_U+comp*P1d, c_B, r_t1);
    ContractY3d<NCOMP,P1d,Q1d>(data, r_t1, c_G, r_t2);
    ContractZ3d<NCOMP,P1d,Q1d>(data, r_t2, c_B, r_V+comp*Q1d+1*NCOMP*Q1d);
    ContractX3d<NCOMP,P1d,Q1d>(data, r_U+comp*P1d, c_B, r_t1);
    ContractY3d<NCOMP,P1d,Q1d>(data, r_t1, c_B, r_t2);
    ContractZ3d<NCOMP,P1d,Q1d>(data, r_t2, c_G, r_V+comp*Q1d+2*NCOMP*Q1d);
  }
}

template <int NCOMP, int P1d, int Q1d>
inline __device__ void gradTranspose3d(BackendData& data, const CeedScalar *__restrict__ r_U,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
  CeedScalar r_t1[Q1d];
  CeedScalar r_t2[Q1d];
  for(int comp=0; comp<NCOMP; comp++) {
    ContractTransposeZ3d<NCOMP,P1d,Q1d>(data, r_U+comp*Q1d+0*NCOMP*Q1d, c_B, r_t1);
    ContractTransposeY3d<NCOMP,P1d,Q1d>(data, r_t1, c_B, r_t2);
    ContractTransposeX3d<NCOMP,P1d,Q1d>(data, r_t2, c_G, r_V+comp*P1d);
    ContractTransposeZ3d<NCOMP,P1d,Q1d>(data, r_U+comp*Q1d+1*NCOMP*Q1d, c_B, r_t1);
    ContractTransposeY3d<NCOMP,P1d,Q1d>(data, r_t1, c_G, r_t2);
    ContractTransposeAddX3d<NCOMP,P1d,Q1d>(data, r_t2, c_B, r_V+comp*P1d);
    ContractTransposeZ3d<NCOMP,P1d,Q1d>(data, r_U+comp*Q1d+2*NCOMP*Q1d, c_G, r_t1);
    ContractTransposeY3d<NCOMP,P1d,Q1d>(data, r_t1, c_B, r_t2);
    ContractTransposeAddX3d<NCOMP,P1d,Q1d>(data, r_t2, c_B, r_V+comp*P1d);
  }
}

template <int Q1d>
inline __device__ void weight1d(BackendData& data, const CeedScalar *qweight1d, CeedScalar *w) {
  *w = qweight1d[data.tidx];
}

template <int Q1d>
inline __device__ void weight2d(BackendData& data, const CeedScalar *qweight1d, CeedScalar *w) {
  *w = qweight1d[data.tidx]*qweight1d[data.tidy];
}

template <int Q1d>
inline __device__ void weight3d(BackendData& data, const CeedScalar *qweight1d, CeedScalar *w) {
  const CeedScalar pw = qweight1d[data.tidx]*qweight1d[data.tidy];
  for (int z = 0; z < Q1d; ++z)
  {
    w[z] = pw*qweight1d[z];
  }
}

);

extern "C" int CeedCudaGenOperatorBuild(CeedOperator op) {

	using std::ostringstream;
  using std::string;
  int ierr;
  bool setupdone;
  ierr = CeedOperatorGetSetupStatus(op, &setupdone); CeedChk(ierr);
  if (setupdone) return 0;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedOperator_Cuda_gen *data;
  ierr = CeedOperatorGetData(op, (void**)&data); CeedChk(ierr);
  CeedQFunction qf;
  CeedQFunction_Cuda_gen *qf_data;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  ierr = CeedQFunctionGetData(qf, (void **)&qf_data); CeedChk(ierr);
  CeedInt Q, P1d, Q1d = -1, numelements, elemsize, numinputfields, numoutputfields, ncomp, dim, ndof;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChk(ierr);
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChk(ierr);
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChk(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChk(ierr);
  CeedEvalMode emode;
  CeedTransposeMode lmode;
  CeedBasis basis;
  CeedBasis_Cuda_shared *basis_data;
  CeedElemRestriction Erestrict;
  CeedElemRestriction_Cuda_reg *restr_data;

  ostringstream code;
  string devFunctions(deviceFunctions);

  code << devFunctions;

  string qFunction(qf_data->qFunctionSource);
  code << qFunction;

  // Setup
  code << "\nextern \"C\" __global__ void oper(CeedInt nelem, void* ctx, CudaFieldsInt indices, CudaFields fields, CudaFields B, CudaFields G, CeedScalar* W) {\n";
  // Input Evecs and Restriction
  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      code << "const CeedScalar* d_u" <<i<<" = fields.in["<<i<<"];\n";
      if (emode != CEED_EVAL_NONE)
      {
        ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
        bool isTensor;
        ierr = CeedBasisGetTensorStatus(basis, &isTensor); CeedChk(ierr);
        //TODO check that all are the same
        ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
        if (isTensor)
        {
          //TODO check that all are the same
          ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChk(ierr);
        } else {
          return CeedError(ceed, 1, "Backend does not implement operators with non-tensor basis");
        }
      }
    }
  }
  data->dim = dim;
  data->Q1d = Q1d;

  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "CeedScalar* d_v"<<i<<" = fields.out["<<i<<"];\n";
  }
  code << "const CeedInt Dim = "<<dim<<";\n";
  code << "const CeedInt Q1d = "<<Q1d<<";\n";
  // code << "const CeedInt Q   = "<<Q<<";\n";
  code << "extern __shared__ CeedScalar slice[];\n";
  code << "BackendData data;\n";
  code << "data.tidx = threadIdx.x;\n";
  code << "data.tidy = threadIdx.y;\n";
  code << "data.tidz = threadIdx.z;\n";
  code << "data.tid  = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;\n";
  code << "data.slice = slice+data.tidz*Q1d"<<(dim>1?"*Q1d":"")<<";\n";
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "// Input field "<<i<<"\n";
    // Get elemsize, emode, ncomp
    ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
    CeedChk(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetNumComponents(qfinputfields[i], &ncomp);
    CeedChk(ierr);
    // Basis action
    switch (emode) {
    case CEED_EVAL_NONE:
      ierr = CeedElemRestrictionGetNumDoF(Erestrict, &ndof); CeedChk(ierr);
      code << "  const CeedInt ncomp_in_"<<i<<" = "<<ncomp<<";\n";
      code << "  const CeedInt nquads_in_"<<i<<" = "<<ndof/ncomp<<";\n";
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
      ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
      ierr = CeedElemRestrictionGetNumDoF(Erestrict, &ndof); CeedChk(ierr);
      code << "  const CeedInt P_in_"<<i<<" = "<<P1d<<";\n";
      code << "  const CeedInt ncomp_in_"<<i<<" = "<<ncomp<<";\n";
      code << "  const CeedInt ndofs_in_"<<i<<" = "<<ndof<<";\n";
      ierr = CeedBasisGetData(basis, (void **)&basis_data); CeedChk(ierr);
      data->B.in[i] = basis_data->d_interp1d;
      code << "  __shared__ double s_B_in_"<<i<<"["<<P1d*Q1d<<"];\n";
      code << "  loadMatrix<P_in_"<<i<<",Q1d>(data, B.in["<<i<<"], s_B_in_"<<i<<");\n";
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
      ierr = CeedElemRestrictionGetNumDoF(Erestrict, &ndof); CeedChk(ierr);
      code << "  const CeedInt ncomp_in_"<<i<<" = "<<ncomp<<";\n";
      code << "  const CeedInt ndofs_in_"<<i<<" = "<<ndof<<";\n";
      ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
      code << "  const CeedInt P_in_"<<i<<" = "<<P1d<<";\n";
      ierr = CeedBasisGetData(basis, (void **)&basis_data); CeedChk(ierr);
      data->B.in[i] = basis_data->d_interp1d;
      data->G.in[i] = basis_data->d_grad1d;
      code << "  __shared__ double s_B_in_"<<i<<"["<<P1d*Q1d<<"];\n";
      code << "  __shared__ double s_G_in_"<<i<<"["<<P1d*Q1d<<"];\n";
      code << "  loadMatrix<P_in_"<<i<<",Q1d>(data, B.in["<<i<<"], s_B_in_"<<i<<");\n";
      code << "  loadMatrix<P_in_"<<i<<",Q1d>(data, G.in["<<i<<"], s_G_in_"<<i<<");\n";
      break;
    case CEED_EVAL_WEIGHT:
      break; // No action
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "// Output field "<<i<<"\n";
    // Get elemsize, emode, ncomp
    ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    CeedChk(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetNumComponents(qfoutputfields[i], &ncomp);
    CeedChk(ierr);
    // Basis action
    switch (emode) {
    case CEED_EVAL_NONE:
      code << "  const CeedInt ncomp_out_"<<i<<" = "<<ncomp<<";\n";
      ierr = CeedElemRestrictionGetNumDoF(Erestrict, &ndof); CeedChk(ierr);
      ierr = CeedOperatorFieldGetLMode(opoutputfields[i], &lmode); CeedChk(ierr);
      code << "  const CeedInt nquads_out_"<<i<<" = "<<ndof<<"/ncomp_out_"<<i<<";\n";
      break; // No action
    case CEED_EVAL_INTERP:
      code << "  const CeedInt ncomp_out_"<<i<<" = "<<ncomp<<";\n";
      ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis); CeedChk(ierr);
      ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
      code << "  const CeedInt P_out_"<<i<<" = "<<P1d<<";\n";
      ierr = CeedBasisGetData(basis, (void **)&basis_data); CeedChk(ierr);
      data->B.out[i] = basis_data->d_interp1d;
      code << "  __shared__ double s_B_out_"<<i<<"["<<P1d*Q1d<<"];\n";
      code << "  loadMatrix<P_out_"<<i<<",Q1d>(data, B.out["<<i<<"], s_B_out_"<<i<<");\n";
      ierr = CeedElemRestrictionGetNumDoF(Erestrict, &ndof); CeedChk(ierr);
      code << "  const CeedInt ndofs_out_"<<i<<" = "<<ndof<<";\n";
      break;
    case CEED_EVAL_GRAD:
      code << "  const CeedInt ncomp_out_"<<i<<" = "<<ncomp<<";\n";
      ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis); CeedChk(ierr);
      ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
      code << "  const CeedInt P_out_"<<i<<" = "<<P1d<<";\n";
      ierr = CeedBasisGetData(basis, (void **)&basis_data); CeedChk(ierr);
      data->B.out[i] = basis_data->d_interp1d;
      data->G.out[i] = basis_data->d_grad1d;
      code << "  __shared__ double s_B_out_"<<i<<"["<<P1d*Q1d<<"];\n";
      code << "  __shared__ double s_G_out_"<<i<<"["<<P1d*Q1d<<"];\n";
      code << "  loadMatrix<P_out_"<<i<<",Q1d>(data, B.out["<<i<<"], s_B_out_"<<i<<");\n";
      code << "  loadMatrix<P_out_"<<i<<",Q1d>(data, G.out["<<i<<"], s_G_out_"<<i<<");\n";
      ierr = CeedElemRestrictionGetNumDoF(Erestrict, &ndof); CeedChk(ierr);
      code << "  const CeedInt ndofs_out_"<<i<<" = "<<ndof<<";\n";
      break;
    case CEED_EVAL_WEIGHT: {
      Ceed ceed;
      ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
      return CeedError(ceed, 1,
                       "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
      break; // Should not occur
    }
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }
  code << "for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < nelem; elem += gridDim.x*blockDim.z) {\n";
  // Input basis apply if needed
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "// Input field "<<i<<"\n";
    // Get elemsize, emode, ncomp
    ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
    CeedChk(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetNumComponents(qfinputfields[i], &ncomp);
    CeedChk(ierr);
    // Basis action
    switch (emode) {
    case CEED_EVAL_NONE:
      code << "  CeedScalar r_t"<<i<<"[ncomp_in_"<<i<<"*Q1d];\n";
      ierr = CeedOperatorFieldGetLMode(opinputfields[i], &lmode); CeedChk(ierr);
      code << "  readQuads"<<(lmode==CEED_NOTRANSPOSE?"":"Transpose")<<dim<<"d<ncomp_in_"<<i<<",Q1d>(data, nquads_in_"<<i<<", elem, d_u"<<i<<", r_t"<<i<<");\n";
      break;
    case CEED_EVAL_INTERP:
      code << "  CeedScalar r_u"<<i<<"[ncomp_in_"<<i<<"*P_in_"<<i<<"];\n";
      ierr = CeedOperatorFieldGetLMode(opinputfields[i], &lmode); CeedChk(ierr);
      ierr = CeedElemRestrictionGetData(Erestrict, (void **)&restr_data); CeedChk(ierr);
      data->indices.in[i] = restr_data->d_ind;
      code << "  readDofs"<<(lmode==CEED_NOTRANSPOSE?"":"Transpose")<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<">(data, ndofs_in_"<<i<<", elem, indices.in["<<i<<"], d_u"<<i<<", r_u"<<i<<");\n";
      code << "  CeedScalar r_t"<<i<<"[ncomp_in_"<<i<<"*Q1d];\n";
      code << "  interp"<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<",Q1d>(data, r_u"<<i<<", s_B_in_"<<i<<", r_t"<<i<<");\n";
      break;
    case CEED_EVAL_GRAD:
      code << "  CeedScalar r_u"<<i<<"[ncomp_in_"<<i<<"*P_in_"<<i<<"];\n";
      ierr = CeedOperatorFieldGetLMode(opinputfields[i], &lmode); CeedChk(ierr);
      ierr = CeedElemRestrictionGetData(Erestrict, (void **)&restr_data); CeedChk(ierr);
      data->indices.in[i] = restr_data->d_ind;
      code << "  readDofs"<<(lmode==CEED_NOTRANSPOSE?"":"Transpose")<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<">(data, ndofs_in_"<<i<<", elem, indices.in["<<i<<"], d_u"<<i<<", r_u"<<i<<");\n";
      code << "  CeedScalar r_t"<<i<<"[ncomp_in_"<<i<<"*Dim*Q1d];\n";
      code << "  grad"<<dim<<"d<ncomp_in_"<<i<<",P_in_"<<i<<",Q1d>(data, r_u"<<i<<", s_B_in_"<<i<<", s_G_in_"<<i<<", r_t"<<i<<");\n";
      break;
    case CEED_EVAL_WEIGHT:
      code << "  CeedScalar r_t"<<i<<"[Q1d];\n";
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
      ierr = CeedBasisGetData(basis, (void **)&basis_data); CeedChk(ierr);
      data->W = basis_data->d_qweight1d;
      code << "  weight"<<dim<<"d<Q1d>(data, W, r_t"<<i<<");\n";
      break; // No action
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }
  // Q function
  code << "// QFunction\n";
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChk(ierr);
    if (emode==CEED_EVAL_GRAD)
    {
      code << "  CeedScalar r_tt"<<i<<"[ncomp_out_"<<i<<"*Dim*Q1d];\n";
    }
    if (emode==CEED_EVAL_NONE || emode==CEED_EVAL_INTERP)
    {
      code << "  CeedScalar r_tt"<<i<<"[ncomp_out_"<<i<<"*Q1d];\n";
    }
  }
  //TODO write qfunction load for this backend
  string qFunctionName(qf_data->qFunctionName);
  code << "  "<<qFunctionName<<"(ctx, "<<(dim==3?"Q1d":"1")<<", ";
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "r_t"<<i<<", ";
  }
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "r_tt"<<i;
    if (i<numoutputfields-1)
    {
      code << ", ";
    }
  }
  code << ");\n";

  // Output basis apply if needed
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "// Output field "<<i<<"\n";
    // Get elemsize, emode, ncomp
    ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    CeedChk(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetNumComponents(qfoutputfields[i], &ncomp);
    CeedChk(ierr);
    // Basis action
    switch (emode) {
    case CEED_EVAL_NONE:
      ierr = CeedOperatorFieldGetLMode(opoutputfields[i], &lmode); CeedChk(ierr);
      code << "  writeQuads"<<(lmode==CEED_NOTRANSPOSE?"":"Transpose")<<dim<<"d<ncomp_out_"<<i<<",Q1d>(data, nquads_out_"<<i<<", elem, r_tt"<<i<<", d_v"<<i<<");\n";
      break; // No action
    case CEED_EVAL_INTERP:
      code << "  CeedScalar r_v"<<i<<"[ncomp_out_"<<i<<"*P_out_"<<i<<"];\n";
      code << "  interpTranspose"<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<",Q1d>(data, r_tt"<<i<<", s_B_out_"<<i<<", r_v"<<i<<");\n";
      ierr = CeedOperatorFieldGetLMode(opoutputfields[i], &lmode); CeedChk(ierr);
      ierr = CeedElemRestrictionGetData(Erestrict, (void **)&restr_data); CeedChk(ierr);
      data->indices.out[i] = restr_data->d_ind;
      code << "  writeDofs"<<(lmode==CEED_NOTRANSPOSE?"":"Transpose")<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<">(data, ndofs_out_"<<i<<", elem, indices.out["<<i<<"], r_v"<<i<<", d_v"<<i<<");\n";
      break;
    case CEED_EVAL_GRAD:
      code << "  CeedScalar r_v"<<i<<"[ncomp_out_"<<i<<"*P_out_"<<i<<"];\n";
      code << "  gradTranspose"<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<",Q1d>(data, r_tt"<<i<<", s_B_out_"<<i<<", s_G_out_"<<i<<", r_v"<<i<<");\n";
      ierr = CeedOperatorFieldGetLMode(opoutputfields[i], &lmode); CeedChk(ierr);
      ierr = CeedElemRestrictionGetData(Erestrict, (void **)&restr_data); CeedChk(ierr);
      data->indices.out[i] = restr_data->d_ind;
      code << "  writeDofs"<<(lmode==CEED_NOTRANSPOSE?"":"Transpose")<<dim<<"d<ncomp_out_"<<i<<",P_out_"<<i<<">(data, ndofs_out_"<<i<<", elem, indices.out["<<i<<"], r_v"<<i<<", d_v"<<i<<");\n";
      break;
    case CEED_EVAL_WEIGHT: {
      Ceed ceed;
      ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
      return CeedError(ceed, 1,
                       "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
      break; // Should not occur
    }
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }

  code << "  }\n";
  code << "}\n\n";

  // std::cout << code.str();

  static const char *libPBP3 = QUOTE(
    typedef struct { const CeedScalar* in[16]; CeedScalar* out[16]; } CudaFields;
    typedef struct { CeedInt* in[16]; CeedInt* out[16]; } CudaFieldsInt;

    extern "C" __global__ void oper(CeedInt nelem, void* ctx, CudaFieldsInt indices, CudaFields fields, CudaFields B, CudaFields G, CeedScalar* W)
    {
    const int Nelements = nelem;
    const int* localizedIds = indices.in[0];
    const double* q = fields.in[0];
    const double* ggeo = fields.in[1];
    const double* I = B.in[0];
    const double* D = G.in[0];
    double* Aq = fields.out[0];


    const int p_G00ID = 0;
    const int p_G01ID = 1;
    const int p_G02ID = 2;
    const int p_G11ID = 3;
    const int p_G12ID = 4;
    const int p_G22ID = 5;


    const int p_Nq = (p_N+1);
    const int p_cubNq = (p_N+2);
    const int p_Np = ( p_Nq*p_Nq*p_Nq );
    const int p_cubNp = ( p_cubNq*p_cubNq*p_cubNq );

    int e = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ double s_Iq[p_cubNq][p_cubNq][p_cubNq];  

    __shared__ double s_D[p_cubNq][p_cubNq];
    __shared__ double s_I[p_cubNq][p_Nq];

    __shared__ double s_Gqr[p_cubNq][p_cubNq];
    __shared__ double s_Gqs[p_cubNq][p_cubNq];

    double r_qt, r_q[p_cubNq], r_Aq[p_cubNq];

    // array of threads
    s_D[ty][tx] = D[p_cubNq*ty+tx];

    if(tx<p_Nq){
	    s_I[ty][tx] = I[p_Nq*ty+tx];
    }

    // load pencil of u into register
    if(tx<p_Nq && ty<p_Nq){
	    for(int k = 0; k < p_Nq; k++) {
		    const int id = e*p_Np + k*p_Nq*p_Nq+ ty*p_Nq + tx;
		    int localId = localizedIds[id]-1;
		    r_q[k] = q[localId];
	    }
    }

    __syncthreads();

    {
	    int b = ty, a = tx;
	    if(a<p_Nq && b<p_Nq){       
		    for(int k=0;k<p_cubNq;++k){     
			    double res = 0;       
			    for(int c=0;c<p_Nq;++c){      
				    res += s_I[k][c]*r_q[c];      
			    }           
			    s_Iq[k][b][a] = res;      
		    }           
	    }
    }

    __syncthreads();

    {
	    int k = ty, a = tx;
	    if(a<p_Nq){         
		    for(int b=0;b<p_Nq;++b){      
			    r_Aq[b] = s_Iq[k][b][a];      
		    }           

		    for(int j=0;j<p_cubNq;++j){     
			    double res = 0;       
			    for(int b=0;b<p_Nq;++b){      
				    res += s_I[j][b]*r_Aq[b];     
			    }           
			    s_Iq[k][j][a] = res;      
		    }           
	    }           
    }

    __syncthreads();

    {
	    int k = ty, j = tx;
	    for(int a=0;a<p_Nq;++a){      
		    r_Aq[a] = s_Iq[k][j][a];      
	    }           

	    for(int i=0;i<p_cubNq;++i){     
		    double res = 0;       
		    for(int a=0;a<p_Nq;++a){      
			    res += s_I[i][a]*r_Aq[a];     
		    }           
		    s_Iq[k][j][i] = res;        
	    }           
    }             

    {
	    for(int k = 0; k < p_cubNq; k++) {
		    r_Aq[k] = 0.f; // zero the accumulator
	    }
    }

    // Layer by layer
    for(int k = 0;k < p_cubNq; k++){

	    __syncthreads();

	    int j = ty, i = tx;

	    // share u(:,:,k)
	    double qr = 0, qs = 0;

	    r_qt = 0;

	    for(int m = 0; m < p_cubNq; m++) {
		    double Dim = s_D[i][m];
		    double Djm = s_D[j][m];
		    double Dkm = s_D[k][m];

		    qr += Dim*s_Iq[k][j][m];
		    qs += Djm*s_Iq[k][m][i];
		    r_qt += Dkm*s_Iq[m][j][i];      
	    }

	    // prefetch geometric factors
	    //    const int gbase = e*p_Nggeo*p_cubNp + k*p_cubNq*p_cubNq + j*p_cubNq + i;
	    const int gbase = e*p_cubNp + k*p_cubNq*p_cubNq + j*p_cubNq + i;
	    const int stride = p_cubNp*Nelements;

	    const double G00 = ggeo[gbase+p_G00ID*stride];
	    const double G01 = ggeo[gbase+p_G01ID*stride];
	    const double G02 = ggeo[gbase+p_G02ID*stride];
	    const double G11 = ggeo[gbase+p_G11ID*stride];
	    const double G12 = ggeo[gbase+p_G12ID*stride];
	    const double G22 = ggeo[gbase+p_G22ID*stride];

	    s_Gqr[j][i] = (G00*qr + G01*qs + G02*r_qt);
	    s_Gqs[j][i] = (G01*qr + G11*qs + G12*r_qt);

	    r_qt = G02*qr + G12*qs + G22*r_qt;

	    __syncthreads();

	    double Aqtmp = 0;

	    for(int m = 0; m < p_cubNq; m++){
		    double Dmi = s_D[m][i];
		    double Dmj = s_D[m][j];
		    double Dkm = s_D[k][m];

		    Aqtmp += Dmi*s_Gqr[j][m];
		    Aqtmp += Dmj*s_Gqs[m][i];
		    r_Aq[m] += Dkm*r_qt;
	    }

	    r_Aq[k] += Aqtmp;
    }

    __syncthreads();

    {             
	    /* lower 'k' */
	    {
		    int j = ty, i = tx;

		    for(int c=0;c<p_Nq;++c){      
			    double res = 0;       
			    for(int k=0;k<p_cubNq;++k){     
				    res += s_I[k][c]*r_q[k];      
			    }           
			    s_Iq[c][j][i] = res;        
		    }           
	    }

	    __syncthreads();

	    {
		    int c = ty, i = tx;

		    if(c<p_Nq){         
			    for(int j=0;j<p_cubNq;++j){     
				    r_q[j] = s_Iq[c][j][i];     
			    }           

			    for(int b=0;b<p_Nq;++b){      
				    double res = 0;       
				    for(int j=0;j<p_cubNq;++j){     
					    res += s_I[j][b]*r_q[j];      
				    }           

				    s_Iq[c][b][i] = res;      
			    }           
		    }           
	    }

	    __syncthreads();

	    {
		    int c = ty, b = tx;

		    if(b<p_Nq && c<p_Nq){       
			    for(int i=0;i<p_cubNq;++i){     
				    r_q[i] = s_Iq[c][b][i];     
			    }           

			    for(int a=0;a<p_Nq;++a){      
				    double res = 0;       
				    for(int i=0;i<p_cubNq;++i){     
					    res += s_I[i][a]*r_q[i];      
				    }           

				    s_Iq[c][b][a] = res;      
			    }           
		    }           
	    }             
    }

    // write out

    {
	    int j = ty, i = tx;
	    if(i<p_Nq && j<p_Nq){
		    for(int k = 0; k < p_Nq; k++){
			    const int id = e*p_Np +k*p_Nq*p_Nq+ j*p_Nq + i;
			    int localId = localizedIds[id]-1;
			    double res = s_Iq[k][j][i];
			    atomicAdd(Aq+localId, res); // atomic assumes Aq zerod
		    }
	    }
    }
    }
  );

  // std::cout << libPBP3 << std::endl;

  if(!strcmp(qFunctionName.c_str(),"f_apply_diff_3d")){
    std::cout << "libPBP3Op called with:" << qFunctionName << std::endl;
    ierr = CeedCompileCuda(ceed, libPBP3, &data->module, 1, "p_N", P1d-1); CeedChk(ierr);
  }else{
    std::cout << "StandardOp called with:" << qFunctionName << std::endl;
    ierr = CeedCompileCuda(ceed, code.str().c_str(), &data->module, 0); CeedChk(ierr);
  }
  ierr = CeedGetKernelCuda(ceed, data->module, "oper", &data->op);
  CeedChk(ierr);

  ierr = CeedOperatorSetSetupDone(op); CeedChk(ierr);

  return 0;
}
