#include <my_kernels.cuh>
#include <string>
#include <iostream>

#include <matmul_cuda.cuh>
#include <cuda_helper.cuh>

// TODO:: n and m is flipped, matrixes are m x n
// n and m should be settable to allow for testing

MatmulKernels::MatmulKernels(int M, int K, int N) {
    this->M = M;
    this->K = K;
    this->N = N;

    // Allocate host memory
    a = (float*)malloc(M * K * sizeof(float));
    b = (float*)malloc(K * N * sizeof(float));
    c = (float*)malloc(M * N * sizeof(float));
    ans = (float*)malloc(M * N * sizeof(float));

    // Initialize host arrays
    int random_values = 1;
    if (random_values) {
        fill_arr_rand(a, M * K);
        fill_arr_rand(b, K * N);
        fill_arr_rand(c, M * N);
    } else {
        for (int i = 0; i < M*K; i++) {
            a[i] = (float) i;
        }
        for (int i = 0; i < K*N; i++) {
            b[i] = (float) M*K + i;
        }
        for (int i = 0; i < M*N; i++) {
            c[i] = (float) M*K + K*N + i;
        }
    }

    calculate_result_cpu();
}

void MatmulKernels::eval(enum e_kernel_name kernel_name) {
    eval(kernel_name, false);
}

void MatmulKernels::eval(enum e_kernel_name kernel_name, bool check_result) {
    float *cc = (float*)malloc(M * N * sizeof(float));
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
    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, M * N * sizeof(float)));

    // Copy input data from host to device
    CHECK_CUDA(cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, c, M * N * sizeof(float), cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();

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
            cuda_matmal_naive<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K, N);

            std::cout << gridDim.x << " | " << gridDim.y << std::endl;
            kernel_cout_name = "cuda_matmul_naive";
        }
        break;
    case CUDA_MATMUL_COALESCING:
        {
            // Define grid and block dimensions
            int block_xDim = 1024;
            dim3 blockDim(block_xDim, 1);
            dim3 gridDim(
                std::max((N*M + block_xDim - 1) / block_xDim, 1), 
                1 
                );
            cuda_matmal_naive<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K, N);
            kernel_cout_name = "cuda_matmul_coalescing";

        }
    }

    CHECK_CUDA(cudaGetLastError());

    // Copy back data from device to host
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaMemcpy(cc, d_c, N * M * sizeof(float), cudaMemcpyDeviceToHost));

    // stop Benchmark Events
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (check_result) {
        check_result_str = " | [Check Result: Same]";
        for (int i = 0; i < N * M; i++) {
            if (std::abs(cc[i] - ans[i]) > 1e-3) {
                std::cout << "error detected at idx: " << i << "; cc[i]=" << cc[i] << " | ans[i]=" << ans[i] << " | diff=" << cc[i] - ans[i] << std::endl;
                check_result_str = " | [Check Result: ERROR]";
                break;
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
            ans[i * N + j] = sum;
            /* ans[i * N + j] = sum + c[i * N + j]; */
        }

    }
}

void MatmulKernels::fill_arr_rand(float *arr, int len) {
    for (int i = 0; i < len; i++) {
        arr[i] = (float)rand() / RAND_MAX;
    }
}

