#include <cuda_helper.cuh>
#include <my_kernels.cuh>

#include <exception>
#include <string>
#include <iostream>

int main() {
    MatmulKernels myKernels = MatmulKernels();

    myKernels.eval(MatmulKernels::CUDA_MATMUL_NAIVE, true);
    myKernels.eval(MatmulKernels::CUDA_MATMUL_NAIVE);

    std::cout << myKernels.report() << std::endl;

    return exitCuda();

}
