/* vim: set ft=cpp: */
#ifndef __GEM_CUDA_H__
#define __GEM_CUDA_H__


#include "types.hpp"
#include <vector>

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>

#define p(s) thrust::raw_pointer_cast(&s[0])
#define CUDA_DEVICE 0

#define SGN(x) ((x) < 0 ? (-1) : (1))
#define ABS(x) ((x)*SGN(x))
#define SQR(x) ((x)*(x))

#define NIL -1
#define INFTY INFINITY
#define MAX_ST0 8
#define MAX_ST1 8
 
#define BLOCK_SIZE 128
#define dF(a,b) ((a)+(b)-1)/(b)
#define CUERR { cudaError_t err;            \
    if ((err = cudaGetLastError()) != cudaSuccess) {        \
        printf("CUDA error: %s : %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1);}}
        
/*******************************************************************************
* exposed methods
*******************************************************************************/

// matching methods

int cuda_match(std::vector<float> *N, std::vector<float> *H, std::vector<single_match> *R, unsigned int St0, unsigned int St1, float E, bool squared);

float cuda_gem(std::vector<float> *N, std::vector<float> *H, unsigned int St0, unsigned int St1, float E, bool squared);

#endif
