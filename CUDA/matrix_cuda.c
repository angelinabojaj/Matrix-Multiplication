#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// CALL WITH ./parallel_cuda_mmult <matrix_size> 
// ex: ./parallel_cuda_mmult 256 

// Main multiplication work area
__global__ void kernel_mult(float* a, float* b, float* c, int blk) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < blk && col < blk) {
                float sum = 0.0f;
                for (int k = 0; k < blk; ++k) {
                        sum += a[row * blk + k] * b[k * blk + col];
                }
                c[row * blk + col] = sum;
        }
}

/* NO CHANGE FROM SERIAL, JUST INITIALIZING THE MATRICIES */
void InitBlock(float *a, float *b, float *c, int blk) {
    int len = blk * blk;
    for (int ind = 0; ind < len; ind++) {
        a[ind] = (float)(rand() % 1000) / 100.0;
        b[ind] = (float)(rand() % 1000) / 100.0;
    }
}

/* FOR PRINT STATEMENTS, DEBUGGING ONLY */
void PrintMatrix(const char* label, float* mat, int blk) {
    printf("\n%s:\n", label);
    for (int i = 0; i < blk; i++) {
        for (int j = 0; j < blk; j++) {
            printf("%6.2f ", mat[i * blk + j]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
        if (argc != 2) {
                fprintf(stderr, "usage: mmult blk\n");
                exit(1);
        }

        int blk = atoi(argv[1]);
        int size = blk * blk * sizeof(float);

        float *a = (float*)malloc(size);
        float *b = (float*)malloc(size);
        float *c = (float*)malloc(size);

        InitBlock(a, b, c, blk);

        PrintMatrix("Matrix A", a, blk);
        PrintMatrix("Matrix B", b, blk);

        // Allocate device memory
        float *d_a, *d_b, *d_c;
        cudaMalloc((void**)&d_a, size);
        cudaMalloc((void**)&d_b, size);
        cudaMalloc((void**)&d_c, size);

        // Copy input data from host to device
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

        // Set up CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Launch kernel to perform matrix multiplication
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((blk + 15) / 16, (blk + 15) / 16);

        // Times only the working portion (multiplication)
        cudaEventRecord(start);
        kernel_mult<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, blk);
        cudaEventRecord(stop);

        // Copy result from device to host
        cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

        // Wait for kernel to finish and record elapsed time
        cudaEventSynchronize(stop);
        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Print the result matrix for debugging help
        PrintMatrix("Result Matrix C", c, blk);

        printf("\nKernel duration: %3.6f ms\n", milliseconds);

        // Clean up cuda
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(a); free(b); free(c);

        return 0;
}
