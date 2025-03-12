#include <cuda_helper.cuh>
#include <my_kernels.cuh>

#include <exception>
#include <string>
#include <iostream>

int main() {
    MyKernels myKernels = MyKernels();

    myKernels.eval(MyKernels::CUDA_MATMUL_NAIVE);

    std::cout << myKernels.report() << std::endl;

    return exitCuda();

}
