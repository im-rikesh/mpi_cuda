/*
  Andrew Sohn
  10/27/2024
  CS698 MPI+CUDA Programming
  Submitted by: Rikesh Niroula
  *
  The MPI+CUDA program compiles and passes the test because they are all zeros.
  Fill the functions.

  NOTE:
  need to place nvidia Common directory two dirs above the current dir
  or 
  change the Makefile reference of Common
*/

#include <iostream>
using std::cerr;
using std::endl;

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <unistd.h>
#include <sys/time.h>

#define TILE_WIDTH 4
#define MAX_TILE_WIDTH 16
#define THREADS_PER_BLOCK 256

extern "C" {
  int matrix_multiply_cuda(int nprocs, int my_rank,int n, int my_work,int *h_A,int *h_B,int *h_C,int gx_dim,int gy_dim,int bx_dim,int by_dim );
}

int matrix_multiply_cpu(int my_rank,int *a, int *b, int *c, int n, int my_work) {
  /* 
     int i, j, k, sum=0;

     Fill function

 */
 
 //matrix_multiply for CPU

  for (int i = 0; i < my_work; i++) {
        for (int j = 0; j < n; j++) {  
            int sum = 0;
            for (int k = 0; k < n; k++) { 
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum; 
        }
    }
    return 0;
}

int compare_cpu(int my_rank, int *host, int *dev, int n, int my_work) {
  int i,j,idx;

  for (i=0; i<my_work; i++) {
    for (j=0; j<n; j++) {
      idx = i*my_work + j;
      if (dev[idx] != host[idx]) {
	printf("DIFFERENT: rank=%d: dev[%d][%d]=%d != host[%d][%d]=%d\n", \
	       my_rank,i,j,dev[idx],i,j,host[idx]);
	return 0;
      }
    }
  }

  return 1;
}

__global__ void mat_mult_cuda(int my_rank, int a_width,int my_work, int *d_a, int *d_b, int *d_c, int tile_width){
  
  __shared__ int a_shared[TILE_WIDTH][TILE_WIDTH];
  __shared__ int b_shared[TILE_WIDTH][TILE_WIDTH];
 

  /* Fill this func */
  /*Tiled approach*/
  
    int ty = threadIdx.y;
    int row = blockIdx.y * tile_width + ty;
    int col = blockIdx.x * tile_width + threadIdx.x;

    int result = 0;

    for (int t = 0; t < (a_width + tile_width - 1) / tile_width; ++t) {    
        if (row < my_work && (t * tile_width + threadIdx.x) < a_width) {
            a_shared[ty][threadIdx.x] = d_a[row * a_width + t * tile_width + threadIdx.x];
        } else {
            a_shared[ty][threadIdx.x] = 0;
        }

        if (col < a_width && (t * tile_width + ty) < a_width) {
            b_shared[ty][threadIdx.x] = d_b[(t * tile_width + ty) * a_width + col];
        } else {
            b_shared[ty][threadIdx.x] = 0;
        }

	//we are syncing all the threads
        __syncthreads(); 

        for (int k = 0; k < tile_width; ++k) {
            result += a_shared[ty][k] * b_shared[k][threadIdx.x];
        }
        //we are syncing all the threads
        __syncthreads(); 
    }

    if (row < my_work && col < a_width) {
        d_c[row * a_width + col] = result;
    }

}

void print_lst_cpu(int name,int rank,int n, int *l){
  int i=0;
  printf("CPU rank=%d: %d: ",rank,name);
  for (i=0; i<n; i++) printf("%x ",l[i]);
  printf("\n");
}

int matrix_multiply_cuda(int nprocs, int my_rank,int n, int my_work,int *h_A,int *h_B,int *h_C,int gx_dim,int gy_dim,int bx_dim,int by_dim ) {
  int cuda_prod=0;
  int *d_A, *d_B, *d_C;
  struct timeval timecheck;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop,0);
  if (my_rank == 0) {
  printf("\n**** properties: rank=%d *****\n",my_rank);
  printf("prop.name=%s\n", prop.name);
  printf("prop.multiProcessorCount=%d\n", prop.multiProcessorCount);
  printf("prop.major=%d minor=%d\n", prop.major, prop.minor);
  printf("prop.maxThreadsPerBlock=%d\n", prop.maxThreadsPerBlock);
  printf("maxThreadsDim.x=%d maxThreadsDim.y=%d maxThreadsDim.z=%d\n", prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
  printf("prop.maxGridSize.x=%d maxGridSize.y=%d maxGridSize.z=%d\n", prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
  printf("prop.maxThreadsPerMultiProcessor=%d\n", prop.maxThreadsPerMultiProcessor);
  printf("prop.totalGlobalMem=%u\n", prop.totalGlobalMem);
  printf("prop.regsPerBlock=%d\n", prop.regsPerBlock);
  printf("**** properties: rank=%d *****\n",my_rank);
  printf("\n");
  }

  unsigned int my_work_size = sizeof(int) * my_work * n;
  unsigned int mat_size = sizeof(int) * n * n;
  printf("rank=%d: my_work=%d data_size=%d bytes\n",my_rank,my_work,my_work_size);

  long dev_start, dev_end, dev_elapsed;
  gettimeofday(&timecheck, NULL);
  dev_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec/ 1000;

  int *h_C_on_cpu = (int *) malloc(my_work_size);

  cudaMalloc(reinterpret_cast<void **>(&d_A), my_work_size);
  cudaMalloc(reinterpret_cast<void **>(&d_B), mat_size);
  cudaMalloc(reinterpret_cast<void **>(&d_C), my_work_size);

  cudaMemcpy(d_A, h_A, my_work_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, mat_size, cudaMemcpyHostToDevice);

  by_dim = bx_dim;
  gx_dim = n/bx_dim;
  gy_dim = n/(bx_dim*nprocs);

  dim3 grid(gx_dim,gy_dim);
  dim3 threads(bx_dim,by_dim);

  mat_mult_cuda<<<grid,threads>>>(my_rank,n,my_work,d_A, d_B, d_C,by_dim);

  cudaMemcpy(h_C,d_C,my_work_size, cudaMemcpyDeviceToHost);

  gettimeofday(&timecheck, NULL);
  dev_end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
  dev_elapsed = dev_end - dev_start;
  
  printf("dev time: rank=%d: %d procs: %ld msecs\n",
	   my_rank, nprocs, dev_elapsed);

  fflush(stdout);

  matrix_multiply_cpu(my_rank,h_A,h_B,h_C_on_cpu,n,my_work);

  if (compare_cpu(my_rank,h_C_on_cpu,h_C,n,my_work)) /* h_C is from dev */
    printf("\nrank=%d: Test CPU: PASS: host == dev\n", my_rank);
  else
    printf("\nrank=%d: Test CPU: FAIL: host != dev\n", my_rank);

  fflush(stdout);

  return cuda_prod;

}

/*************************************************
  End of file
*************************************************/
