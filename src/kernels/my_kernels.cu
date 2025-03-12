#include <my_kernels.cuh>
#include <string>
#include <iostream>

#include <matmul_cuda.cuh>
#include <cuda_helper.cuh>

MatmulKernels::MatmulKernels() {
    // Allocate host memory
    a = (float*)malloc(N * K * sizeof(float));
    b = (float*)malloc(K * M * sizeof(float));
    c = (float*)malloc(N * M * sizeof(float));
    ans = (float*)malloc(N * M * sizeof(float));

    // Initialize host arrays
    int random_values = 0;
    if (random_values) {
        fill_arr_rand(a, N * K);
        fill_arr_rand(b, K * M);
        fill_arr_rand(c, N * M);
    } else {
        for (int i = 0; i < N*K; i++) {
            a[i] = (float) i;
        }
        for (int i = 0; i < K*M; i++) {
            b[i] = (float) N*K + i;
        }
        for (int i = 0; i < N*M; i++) {
            c[i] = (float) N*K + K*M + i;
        }
    }

    calculate_result_cpu();
}

void MatmulKernels::eval(enum e_kernel_name kernel_name) {
    eval(kernel_name, false);
}

void MatmulKernels::eval(enum e_kernel_name kernel_name, bool check_result) {
    float *cc = (float*)malloc(N * M * sizeof(float));
    float *d_a, *d_b, *d_c;
    std::string kernel_cout_name;
    std::string check_result_str;

    // Benchmarking Events
    cudaEvent_t start, stop;
    float milliseconds;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Allocate device memory
    cudaMalloc(&d_a, N * K * sizeof(float));
    cudaMalloc(&d_b, K * M * sizeof(float));
    cudaMalloc(&d_c, N * M * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_a, a, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, K * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, N * M * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel Run
    switch (kernel_name) {
    case CUDA_MATMUL_NAIVE:
        {
            // Define grid and block dimensions
            int block_size = 32;
            dim3 blockDim(block_size, block_size);
            dim3 gridDim(
                std::max((N + block_size - 1) / block_size, 1), 
                std::max((M + block_size - 1) / block_size, 1)
                );
            cuda_matmal_naive<<<gridDim, blockDim>>>(d_a, d_b, d_c, N, K, M);
            kernel_cout_name = "cuda_matmul_naive";
        }
        break;
    }

    // Copy back data from device to host
    cudaDeviceSynchronize();
    cudaMemcpy(cc, d_c, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    // stop Benchmark Events
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (check_result) {
        check_result_str = " | [Check Result: Same]";
        for (int i = 0; i < N * M; i++) {
            if (cc[i] != ans[i]) {
                check_result_str = " | [Check Result: ERROR]";
                break;
            }        
        }
    }

    // print kernel run into stdout
    std::cout << "Ran the kernel (<" << kernel_cout_name << ">) in (elapsed: " <<
        milliseconds << " ms)  " << check_result_str << std::endl;


    // Clean up
    free(cc);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

std::string MatmulKernels::report() {
    return "placeholder report::\n\n";
}

void MatmulKernels::calculate_result_cpu() {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int l = 0; l < K; l++) {
                sum += a[i * K + l] * b[l * N + j];
            }
            ans[i * N + j] = sum + c[i * N + j];
        }

    }
}

void MatmulKernels::fill_arr_rand(float *arr, int len) {
    for (int i = 0; i < len; i++) {
        arr[i] = (float)rand() / RAND_MAX;
    }
}

/*
// print out array
std::cout << "------------------------------" << std::endl;
for (int i = 0; i < N * K; ++i) {
    std::cout << a[i] << " ";
}
std::cout << std::endl << "------------------------------" << std::endl;
for (int i = 0; i < K * M; ++i) {
    std::cout << b[i] << " ";
}
std::cout << std::endl << "------------------------------" << std::endl;
for (int i = 0; i < N * M; ++i) {
    std::cout << c[i] << " ";
}
std::cout << std::endl << "------------------------------" << std::endl;
for (int i = 0; i < N * M; ++i) {
    std::cout << cc[i] << " ";
}
std::cout << std::endl << "------------------------------" << std::endl;
for (int i = 0; i < N * M; ++i) {
    std::cout << ans[i] << " ";
}
std::cout << std::endl << "------------------------------" << std::endl;
*/
