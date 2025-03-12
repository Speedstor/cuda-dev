# cuda-dev

My tinkering with CUDA. Getting familiar with parallel programming.

![image](https://github.com/user-attachments/assets/1b1e4671-6ae4-4556-a7f0-008776f41161)

---

## Benchmarks
| optimization | ..      |
| ---          | ---     |
| naive        | GFLOP/s |
| coalescing   | GFLOP/s |


## Usage
### :one: Naive Matmul
```cpp
MatmulKernels myKernels = MatmulKernels();

myKernels.eval(MatmulKernels::CUDA_MATMUL_NAIVE, true);
myKernels.eval(MatmulKernels::CUDA_MATMUL_NAIVE);

std::cout << myKernels.report() << std::endl;
```
