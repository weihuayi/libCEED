// Copyright (c) 2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#ifndef CEED_OCCA_BASIS_HEADER
#define CEED_OCCA_BASIS_HEADER

#include "ceed-occa-ceed-object.hpp"
#include "ceed-occa-vector.hpp"

namespace ceed {
  namespace occa {
    class Basis : public CeedObject {
     public:
      // Ceed object information
      CeedInt ceedComponentCount;

      // Owned information
      CeedInt dim;
      CeedInt P;
      CeedInt Q;

      Basis();

      virtual ~Basis();

      static Basis* getBasis(CeedBasis basis,
                             const bool assertValid = true);

      static Basis* from(CeedBasis basis);
      static Basis* from(CeedOperatorField operatorField);

      int setCeedFields(CeedBasis basis);

      virtual bool isTensorBasis() const = 0;

      virtual const char* getFunctionSource() const = 0;

      virtual int apply(const CeedInt elementCount,
                        CeedTransposeMode tmode,
                        CeedEvalMode emode,
                        Vector *u,
                        Vector *v) = 0;

      //---[ Ceed Callbacks ]-----------
      static int registerCeedFunction(Ceed ceed, CeedBasis basis,
                                      const char *fname, ceed::occa::ceedFunction f);

      static int ceedApply(CeedBasis basis, const CeedInt nelem,
                           CeedTransposeMode tmode,
                           CeedEvalMode emode, CeedVector u, CeedVector v);

      static int ceedDestroy(CeedBasis basis);
    };
  }
}

#endif
