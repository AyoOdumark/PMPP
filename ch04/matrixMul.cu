#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define BLOCK_WIDTH 64

uint64_t nanos() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec*1000000000 + start.tv_nsec;
}
/**
__global__ void pictureKernel(float *d_Pin, float *d_Pout, int n, int m) 
{
    // Calculate the row number of the d_Pin and d_Pout element to process
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the column number of the d_Pin and d_Pout element to process
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < m) && (col < n)) {
        d_Pout[row * n + col] = 2 * d_Pin[row * n + col];
    }
}
**/

__global__ void MatrixMulKernel(float *d_M, float *d_N, float *d_P, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float val = 0.0;

        for (int k = 0; k < width; k++) {
            val += d_M[row * width + k] * d_N[k * width + col];
        }
        d_P[row * width + col] = val;
    }
}

void matmul(float *M, float *N, float *P, int width)
{
    int size = width * width * sizeof(float);

    // Declare device variables
    float *d_M, *d_N, *d_P;

    // Allocate memory for variables on devices. And Copy d_M, and d_N
    cudaMalloc((void **)&d_M, size);
    cudaMalloc((void **)&d_N, size);
    cudaMalloc((void **)&d_P, size);

    cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice);

    // Kernel Configuration
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 dimGrid(ceil(width/BLOCK_WIDTH), ceil(width/BLOCK_WIDTH));
    
    // Kernel Launch
    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);
    
    // Copy d_P from device to host and free device memory
    cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

int main() 
{
    int width = 2048;
    
    int low = 1, high = 9;

    srand(time(NULL));

    float *M = (float *)malloc(width * width * sizeof(float));
    float *N = (float *)malloc(width * width * sizeof(float));
    float *P = (float *)malloc(width * width * sizeof(float));

    for (int i = 0; i < width*width; i++) {
        M[i] = low + rand() % (high - low + 1);
        N[i] = low + rand() % (high - low + 1);
        P[i] = 0.0;
    }
    
    uint64_t start = nanos();
    matmul(M, N, P, width);
    uint64_t end = nanos();

    double gflop = (2.0 * width * width * width) * 1e-9;
    double time = (end - start) * 1e-9;

    printf("%d-dim matrix, gflops/sec %f, time-taken %f\n", width, gflop/time, time);

    free(M);
    free(N);
    free(P);

    return 0;
}



