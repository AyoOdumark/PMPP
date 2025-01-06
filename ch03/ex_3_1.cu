#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void vecAdd2dKernel(float *A, float *B, float *C, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < n && j < n)
        A[i * n + j] = B[i * n + j] + C[i * n + j];
}

__global__ void vecAddRowKernel(float *A, float *B, float *C, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) return;

    for (int j = 0; j < n; j++)
        A[i * n + j] = B[i * n + j] + C[i * n + j];
}

__global__ void vecAddColKernel(float *A, float *B, float *C, int n)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (j >= n) return;

    for (int i = 0; i < n; i++)
        A[i * n + j] = B[i * n + j] + C[i * n + j];
}

void vec_add2d(float *A, float *B, float *C, int n)
{
    int size = n * n * sizeof(float);
    float *d_A, *d_B, *d_C;
 
    // Allocate A, B, C on the device memory. Copy B and C to the device memory

     cudaMalloc((void **)&d_A, size);
     cudaMalloc((void **)&d_B, size);
     cudaMalloc((void **)&d_C, size);


     cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
     cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

     // Row Threading
     //vecAddRowKernel<<<1, 32>>>(d_A, d_B, d_C, n);

     // Column Threading
     //vecAddColKernel<<<1, 32>>>(d_A, d_B, d_C, n);

     // Full Threading
     dim3 blockDim(32, 32);
     dim3 gridDim(ceil(n/32.0), ceil(n/32.0));

     vecAdd2dKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);

     // Copy A from device to host and free device memory
     cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);
}

void printResult(float *arr, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.1f ", arr[i * n + j]);
        }
        printf("\n");
    }
}

int main()
{
    int DIM = 32;
    float A[DIM * DIM];
    float B[DIM * DIM];
    float C[DIM * DIM];

    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            A[i * DIM + j] = 0.0;
            B[i * DIM + j] = 1.0;
            C[i * DIM + j] = 1.0;
        }
    }
    vec_add2d(A, B, C, DIM);
   // printf("YOYO!");
    printResult(A, DIM);
    return 0;
}
