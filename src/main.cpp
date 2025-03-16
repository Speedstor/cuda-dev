#include <cuda_helper.cuh>
#include <my_kernels.cuh>

#include <exception>
#include <string>
#include <iostream>

int main() {
    /* MatmulKernels myKernels(2047, 3, 2047); */
    {
        MatmulKernels myKernels(2, 3, 2);
        myKernels.eval(MatmulKernels::CUDA_MATMUL_NAIVE, true);
    }
     {
        MatmulKernels myKernels(256, 512, 256);
        myKernels.eval(MatmulKernels::CUDA_MATMUL_NAIVE, true);
    }


    if (false) {
        MatmulKernels myKernels(256, 512, 256);
        myKernels.eval(MatmulKernels::CUDA_MATMUL_NAIVE, true);
        myKernels.eval(MatmulKernels::CUDA_MATMUL_NAIVE);
        myKernels.eval(MatmulKernels::CUDA_MATMUL_NAIVE);
        myKernels.eval(MatmulKernels::CUDA_MATMUL_NAIVE);
        myKernels.eval(MatmulKernels::CUDA_MATMUL_NAIVE);
        myKernels.eval(MatmulKernels::CUDA_MATMUL_NAIVE);
        std::cout << myKernels.report() << std::endl;
    }

    return exitCuda();
}

