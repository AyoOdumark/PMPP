#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void matVecMulKernel(float *A, float *B, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    for (int j = 0; j < n; j++) {
        A[i] += B[i * n + j] * C[j];
    }
}

void matVecMul(float *A, float *B, float *C, int n) 
{
    float *d_A, *d_B, *d_C;

    // Create memroy for A, B, and C and copy B and C to device memory
    cudaMalloc((void **)&d_B, n*n*sizeof(float));
    cudaMalloc((void **)&d_C, n*sizeof(float));
    cudaMalloc((void **)&d_A, n*sizeof(float));

    cudaMemcpy(d_B, B, n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, n*sizeof(float), cudaMemcpyHostToDevice);

    // Kernel Entry
    matVecMulKernel<<<32, 32>>>(d_A, d_B, d_C, n);

    // Copy Result from device to host
    cudaMemcpy(A, d_A, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

void printResult(float *arr, int n)
{
    for (int i = 0; i < n; i++) {
        printf("%.1f\n", arr[i]);
    }
}

int main()
{
    int m = 1024;

    float A[m], B[m * m], C[m];

    for (int i = 0; i < m*m; i++) {
        B[i] = 2.0;
    }

    for (int i = 0; i < m; i++) {
        A[i] = 0.0;
        C[i] = 3.0;
    }

    matVecMul(A, B, C, m);
    printResult(A, m);
    return 0;
}
