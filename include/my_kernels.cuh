#pragma once

#include <string>

/* #define N 1024 */
/* #define K 2048 */
/* #define M 1024 */

/* #define M 2048 */
/* #define K 3 */
/* #define N 2048 */

class MatmulKernels {
public:
    MatmulKernels(int M, int K, int N);

    enum e_kernel_name {
        CPU_MATMUL,
        CUDA_MATMUL_NAIVE,
        CUDA_MATMUL_COALESCING
    };

    void eval(enum e_kernel_name kernel_name);
    void eval(enum e_kernel_name kernel_name, bool check_result);
    std::string report();

    int M, K, N;

private:
    float *a, *b, *c, *ans;

    // Initialize matrix with random values
    void fill_arr_rand(float *arr, int len);
    void calculate_result_cpu();
};
