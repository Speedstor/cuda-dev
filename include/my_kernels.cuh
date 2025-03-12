#pragma once

#include <string>

/* #define N 1024 */
/* #define K 2048 */
/* #define M 1024 */

#define N 2
#define K 3
#define M 2

class MatmulKernels {
public:
    MatmulKernels();

    enum e_kernel_name {
        CPU_MATMUL,
        CUDA_MATMUL_NAIVE
    };

    void eval(enum e_kernel_name kernel_name);
    void eval(enum e_kernel_name kernel_name, bool check_result);
    std::string report();

private:
    float *a, *b, *c, *ans;

    // Initialize matrix with random values
    void fill_arr_rand(float *arr, int len);
    void calculate_result_cpu();
};
