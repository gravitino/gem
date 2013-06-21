#ifndef __GEM_H__
#define __GEM_H__

#include "types.hpp"                  // type definitions
#include <iostream>                   // input/output
#include <cmath>                      // basic math support
#include <limits>                     // appropriate usage of limits
#include <omp.h>                      // openmp support

#define SGN(x) ((x) < 0 ? (-1) : (1))
#define ABS(x) ((x)*SGN(x))

#define INFTY std::numeric_limits<ftype>::infinity()
#define NIL   std::numeric_limits<itype>::max()

/*******************************************************************************
* exposed methods
*******************************************************************************/

// matching methods

int match(std::vector<float>* N, std::vector<float>* H, std::vector<single_match>* R, unsigned int St0, unsigned int St1, float E, bool squared, bool omp);

// backtrace methods

int backtrace(std::vector<float>* N, std::vector<float>* H, single_match* M, std::vector<float>* L, std::vector<unsigned int>* X, std::vector<float>* Y, unsigned int St0, unsigned int St1, float E, bool omp, bool squared);

#endif
