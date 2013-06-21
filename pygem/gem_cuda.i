%module libgemcuda
%{
/* Includes the header in the wrapper code */
#include "gem_cuda.cuh"
%}

%include "typemaps.i"
%include "std_vector.i"

namespace std {
    %template(TimeSeries) vector<float>;
    %template(TimeCoords) vector<unsigned int>;
    %template(Result) vector<single_match>; 
}

/* Parse the header file to generate wrappers */
%include "gem_cuda.cuh"
%include "types.hpp"
