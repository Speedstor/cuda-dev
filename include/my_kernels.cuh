#pragma once

#include <string>

class MyKernels {
public:
    MyKernels();

    enum kernel_name {
        CPU_MATMUL,
        CUDA_MATMUL_NAIVE
    };


    void eval(enum kernel_name);
    std::string report();

private:

};
