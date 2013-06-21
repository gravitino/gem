CC        = g++
NVCC      = nvcc

CFLAGS    = -march=native -O3 -fPIC 
NVCFLAGS  = -arch=sm_30 -O3

OBJ       = gem.o gem_wrap.o
NVOBJ     = gem_cuda.o gem_cuda_wrap.o 

PYPATH    = /usr/include/python2.7
CUDA_PATH  = /usr/local/cuda

CUDAFLAGS = -L $(CUDA_PATH)/lib64/ -lcudart


all: _libgem.so 
cuda: _libgemcuda.so

gem_wrap.cxx: gem.i gem.cpp gem.hpp sort.hpp types.hpp
	swig -python -c++ gem.i
gem_cuda_wrap.cxx: gem_cuda.i gem_cuda.cu gem_cuda.cuh sort.hpp types.hpp
	swig -python -c++ gem_cuda.i

gem.o: gem.cpp gem.hpp sort.hpp
	$(CC) $(CFLAGS) -c gem.cpp -fopenmp
gem_cuda.o: gem.cpp gem.hpp sort.hpp
	$(NVCC) $(NVCFLAGS) -Xcompiler="$(CFLAGS)" -c gem_cuda.cu 


gem_wrap.o: gem_wrap.cxx
	$(CC) $(CFLAGS) -c gem_wrap.cxx -I $(PYPATH) -fopenmp
gem_cuda_wrap.o: gem_cuda_wrap.cxx
	$(CC) $(CFLAGS) -c gem_cuda_wrap.cxx -I $(PYPATH) -fopenmp -I $(CUDA_PATH)/include

_libgem.so: gem.o gem_wrap.o 
	$(CC) $(CFLAGS) $(OBJ) -shared -o _libgem.so -lgomp -fopenmp
	rm -f $(OBJ)
	rm -f gem_wrap.cxx
_libgemcuda.so: gem_cuda.o gem_cuda_wrap.o 
	$(CC) $(CFLAGS) $(NVOBJ) $(CUDAFLAGS) -shared -o _libgemcuda.so 
	rm -f $(NVOBJ)
	rm -f gem_cuda_wrap.cxx

benchmarks: gem.cpp gem.hpp gem_cuda.cu gem_cuda.cuh sort.hpp types.hpp
	$(CC) $(CFLAGS) gem.cpp -o benchmark -fopenmp
	$(NVCC) $(NVCFLAGS) gem_cuda.cu -o benchmark_cuda

clean:
	rm -f benchmark
	rm -f benchmark_cuda
	rm -f libgem.py
	rm -f libgemcuda.py
	rm -f _libgem.so
	rm -f _libgemcuda.so
	rm -f *~
	rm -f *.pyc

