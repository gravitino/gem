#include "gem_cuda.cuh"
#include "sort.hpp"

__device__ __constant__ itype Hsize, Nsize;
__device__ __constant__ int St0, St1; 
__device__ __constant__ ftype E_k[MAX_ST0], E_l[MAX_ST1];

__device__ __inline__ 
void update (ftype dst, itype sou_src, itype len_src, ftype pen_src, ftype a0, 
             itype &sou_trg, itype &len_trg, ftype &pen_trg, ftype &a_0_trg) {
    
    const itype l1 = len_src+1;
    const ftype pd = pen_src+dst;
    
    if (pd*len_trg < pen_trg*l1) {
        
        sou_trg = sou_src;
        len_trg = l1;
        pen_trg = pd;
        a_0_trg = a0;
    }
}

template <bool squared> __global__ 
void relax_kernel(itype *len, itype *sou, ftype *pen, ftype *a_0,
                  ftype *H, ftype Ni, int i) {
    
    const int j = blockIdx.x*blockDim.x + threadIdx.x + 1;
    if (j>=Hsize) return;
    
    // target node in registers
    const itype trg = (i%(St0+1))*Hsize+j;
    itype sou_trg = NIL;
    itype len_trg = 0;
    ftype pen_trg = INFTY;
    ftype a_0_trg = 0;
    
    
    // calculate delta
    const ftype delta = H[j]-Ni;
    
    for (int l = 0; j-l > 0 && l < St1; ++l) {

        const itype src = ((i-1)%(St0+1))*Hsize+j-l-1;
    
        const itype sou_src = sou[src];
        const itype len_src = len[src];
        const ftype pen_src = pen[src];

        const ftype EL = E_l[l];
        const ftype a0 = (1-EL)*a_0[src] + EL*delta;
        
        if (squared)
            update(SQR(delta-a0), sou_src, len_src, pen_src, a0,
                                  sou_trg, len_trg, pen_trg, a_0_trg);
        else
            update(ABS(delta-a0), sou_src, len_src, pen_src, a0,
                                  sou_trg, len_trg, pen_trg, a_0_trg);
    }
    
    for (int k = 1; i-k > 0 && k < St0; ++k) {

        const itype src = ((i-k-1)%(St0+1))*Hsize+j-1;
    
        const itype sou_src = sou[src];
        const itype len_src = len[src];
        const ftype pen_src = pen[src];
    
        const ftype EK = E_k[k];
        const ftype a0 = (1-EK)*a_0[src] + EK*delta;
        
        if (squared)
            update(SQR(delta-a0), sou_src, len_src, pen_src, a0,
                                  sou_trg, len_trg, pen_trg, a_0_trg);
        else
            update(ABS(delta-a0), sou_src, len_src, pen_src, a0,
                                  sou_trg, len_trg, pen_trg, a_0_trg);
    }
    
    // finally write to cell
    sou[trg] = sou_trg;
    len[trg] = len_trg;
    pen[trg] = pen_trg;
    a_0[trg] = a_0_trg;
}

__global__ 
void init(itype *len, itype *sou, ftype *pen, ftype *a_0, ftype *H, ftype N0) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x; 

    // initialize row zero
    for (int j = tid; j < Hsize; j += nthreads) {
        
        sou[j] = j;
        len[j] = 1;
        pen[j] = 0;
        a_0[j] = H[j] - N0;
    }

    // reset column zero except upper left cell (0,0)
    if (tid < St0) {
        const itype dest = (tid+1)*Hsize;
            
        sou[dest] = NIL;
        len[dest] = 0;
        pen[dest] = INFTY;    
    }
}

template <bool fullResult, bool squared>
int cuda_meta_match(series *N, series *H, result *R, 
                      itype St0, itype St1, ftype E) {
    
    
    // initialize CUDA device
    cudaSetDevice(CUDA_DEVICE);
    cudaDeviceReset();
    
    // needle(query) and haystack(subject) sizes
    itype Hsize = H->size();
    itype Nsize = N->size();

    // dynamic averaging coefficients
    std::vector<ftype> E_k(St0), E_l(St1);
    for (itype k=0; k<St0; k++) E_k[k] = 1 - pow(1-E, 1.0/(k+1));
    for (itype l=0; l<St1; l++) E_l[l] = 1 - pow(1-E, l+1);

    // copy the constants to CUDA device
    cudaMemcpyToSymbol(::E_k, &E_k[0], sizeof(ftype)*St0);                CUERR
    cudaMemcpyToSymbol(::E_l, &E_l[0], sizeof(ftype)*St1);                CUERR
    cudaMemcpyToSymbol(::Hsize, &Hsize, sizeof(itype));                   CUERR
    cudaMemcpyToSymbol(::Nsize, &Nsize, sizeof(itype));                   CUERR
    cudaMemcpyToSymbol(::St0, &St0, sizeof(itype));                       CUERR
    cudaMemcpyToSymbol(::St1, &St1, sizeof(itype));                       CUERR

    // configure cache
    cudaFuncSetCacheConfig(relax_kernel<true>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(relax_kernel<false>, cudaFuncCachePreferL1);

    // copy the main array
    thrust::device_vector<ftype> d_H(H->begin(), H->end());               CUERR
    thrust::device_vector<itype> d_len(Hsize*(St0+1));                    CUERR
    thrust::device_vector<itype> d_sou(Hsize*(St0+1));                    CUERR
    thrust::device_vector<ftype> d_pen(Hsize*(St0+1));                    CUERR
    thrust::device_vector<ftype> d_a_0(Hsize*(St0+1));                    CUERR

    // initialize data structures
    init<<<dF(Hsize,BLOCK_SIZE), BLOCK_SIZE>>>
        (p(d_len), p(d_sou), p(d_pen), p(d_a_0), p(d_H), N->at(0));       CUERR

    // relax each row from the beginning
    relax_kernel<squared><<<dF(Hsize, BLOCK_SIZE), BLOCK_SIZE>>>
           (p(d_len), p(d_sou), p(d_pen), p(d_a_0), p(d_H), N->at(1), 1); CUERR
    d_pen[0] = INFTY;                                                     CUERR
    
    for (itype i = 2; i < Nsize; ++i) 
        relax_kernel<squared><<<dF(Hsize, BLOCK_SIZE), BLOCK_SIZE>>>
           (p(d_len), p(d_sou), p(d_pen), p(d_a_0), p(d_H), N->at(i), i); CUERR
    
    // finally synchronize device and return best match
    cudaDeviceSynchronize();                                              CUERR
    
    // calculate offset for last row
    itype last = Hsize*((Nsize-1)%(St0+1));
    
    // calculate quotient of penalty and length
    thrust::device_vector<ftype> d_rst(Hsize);                            CUERR
    thrust::transform(d_pen.begin()+last, d_pen.begin()+last+Hsize,
            d_len.begin()+last, d_rst.begin(), thrust::divides<ftype>()); CUERR
    
    if (fullResult) {
        
        // sorted list of indices
        thrust::device_vector<itype> d_ind(Hsize);                        CUERR
        thrust::sequence(d_ind.begin(), d_ind.end(), 0);                  CUERR
        thrust::sort_by_key(d_rst.begin(), d_rst.end(), d_ind.begin());   CUERR
        
        // copy back to device
        thrust::host_vector<itype> ind (d_ind.begin(), d_ind.end());      CUERR
        thrust::host_vector<itype> sou (d_sou.begin(), d_sou.end());      CUERR
        thrust::host_vector<itype> len (d_len.begin(), d_len.end());      CUERR
        thrust::host_vector<ftype> rst (d_rst.begin(), d_rst.end());      CUERR

        // finally write to result vector
        for (itype i = 0; i < Hsize; ++i) {
            
            single_match entry;
            entry.left = sou[ind[i]];
            entry.right = ind[i];
            entry.length = len[ind[i]];
            entry.penalty = rst[i];

            R->push_back(entry);
        }
        
        // remove non-overlapping matches
        non_overlap(R);
    
    } else {

        // take the minimum
        single_match entry;
        entry.penalty = thrust::reduce(d_rst.begin(), d_rst.end(), INFTY,
                                       thrust::minimum<ftype>());         CUERR
        R->push_back(entry);
    }
    
    return 0;
}

int cuda_match(series *N, series *H, result *R, 
               itype St0, itype St1, ftype E, bool squared) {
    std::cout << "Running on GPU id: " << CUDA_DEVICE << std::endl;
 
    if (squared)
        return cuda_meta_match<true, true> (N, H, R, St0, St1, E);
    else
        return cuda_meta_match<true, false> (N, H, R, St0, St1, E);
}

ftype cuda_gem(series *N, series *H, 
               itype St0, itype St1, ftype E, bool squared) {
    result *R = new result();
    
    if (squared)
        cuda_meta_match<false, true> (N, H, R, St0, St1, E);
    else
        cuda_meta_match<false, false> (N, H, R, St0, St1, E);

    ftype best = R->at(0).penalty;
    
    delete R;
    
    return best;
}


int main(int argc, char **args) {

    series *N = new series();
    series *H = new series();
    result *R = new result();
   
    for (itype i = 0; i < 1000; ++i)
        N->push_back(sin(i/100.0));
    
    for (itype i = 0; i < 1000000; ++i)
        H->push_back(0);
        
    for (itype i = 701234; i < 702234; ++i)
        H->at(i)=sin(i/100.0);
   
    
    cuda_match(N, H, R, 2, 2, 0.01, true);
 
    std::cout << R->at(0).left << " " << R->at(0).right << " " << R->at(0).penalty << std::endl; 
    std::cout << cuda_gem(N, H, 2, 2, 0.01, true) << std::endl;
 
 
    delete N;
    delete H;
    delete R;  
   
    return 0;
}

