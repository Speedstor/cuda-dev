# cuda-dev
---

My tinkering with CUDA. Getting familiar with parallel programming.

## Benchmarks
| optimization | ..      |
| ---          | ---     |
| naive        | GFLOP/s |
| coalescing   | GFLOP/s |


## Usage
### :one: 1.Naive Matmul
```cpp
MatmulKernels myKernels = MatmulKernels();

myKernels.eval(MatmulKernels::CUDA_MATMUL_NAIVE, true);
myKernels.eval(MatmulKernels::CUDA_MATMUL_NAIVE);

std::cout << myKernels.report() << std::endl;
```
