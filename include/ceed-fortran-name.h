// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_fortran_name_h
#define _ceed_fortran_name_h

/* establishes some macros to establish
   * the FORTRAN naming convention
     default      gs_setup, etc.
     -DUPCASE     GS_SETUP, etc.
     -DUNDERSCORE gs_setup_, etc.
   * a prefix for all external (non-FORTRAN) function names
     for example, -DPREFIX=jl_   transforms fail -> jl_fail
   * a prefix for all external FORTRAN function names
     for example, -DFPREFIX=jlf_ transforms gs_setup_ -> jlf_gs_setup_
*/

/* the following macro functions like a##b,
   but will expand a and/or b if they are themselves macros */
#define TOKEN_PASTE_(a,b) a##b
#define TOKEN_PASTE(a,b) TOKEN_PASTE_(a,b)

#ifdef PREFIX
#  define PREFIXED_NAME(x) TOKEN_PASTE(PREFIX,x)
#else
#  define PREFIXED_NAME(x) x
#endif

#ifdef FPREFIX
#  define FPREFIXED_NAME(x) TOKEN_PASTE(FPREFIX,x)
#else
#  define FPREFIXED_NAME(x) x
#endif

#if defined(UPCASE)
#  define FORTRAN_NAME(low,up) FPREFIXED_NAME(up)
#  define FORTRAN_UNPREFIXED(low,up) up
#elif defined(UNDERSCORE)
#  define FORTRAN_NAME(low,up) FPREFIXED_NAME(TOKEN_PASTE(low,_))
#  define FORTRAN_UNPREFIXED(low,up) TOKEN_PASTE(low,_)
#else
#  define FORTRAN_NAME(low,up) FPREFIXED_NAME(low)
#  define FORTRAN_UNPREFIXED(low,up) low
#endif

#endif
