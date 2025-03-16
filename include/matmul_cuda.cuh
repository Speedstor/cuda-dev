#pragma once

__global__ void cuda_matmal_naive(float *A, float *B, float *C, int m, int k, int n);
__global__ void cuda_matmal_coalescing(float *A, float *B, float *C, int n, int k, int m);
