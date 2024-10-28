
#ifndef DOT_PROD_CUDA_H
#define DOT_PROD_CUDA_H

#include <cuda_runtime.h>


#ifdef __cplusplus
extern "C" {
#endif

// Function to compute the sum of an integer array
int sum(int size, int *data);

// Function to calculate the binary logarithm (base 2)
int dev_my_log(int val);

// Function to compute the dot product on the CPU
int dot_product_cpu(int rank, int n, int *x, int *y);

// CUDA kernel for dot product with tree reduction
__global__ void dot_prod_tree_reduction(int *a, int *b, int *c, int my_work, int log_n);

// CUDA kernel for serial dot product
__global__ void dot_prod_serial(int *a, int *b, int *c, int n, int my_work, int log_n);

int dot_product_cuda(int my_rank, int my_work, int *h_A, int *h_B);

#ifdef __cplusplus
}
#endif

#endif // DOT_PRODUCT_CUDA_H

