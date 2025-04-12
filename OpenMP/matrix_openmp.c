#include <stdio.h>
#include <stdlib.h>     // For malloc, free, rand, atoi, exit
#include <omp.h> // Library for OpenMP

// Matrices are randomized here
void InitBlock(float *a, float *b, float *c, int blk) {
    int len = blk * blk;
    for (int ind = 0; ind < len; ind++) {
        a[ind] = (float)(rand() % 1000) / 100.0;
        b[ind] = (float)(rand() % 1000) / 100.0;
        c[ind] = 0.0;
    }
}

void BlockMult(float* c, float* a, float* b, int blk) {
    #pragma omp parallel for collapse(2) // Paralllize Outer Two Loops

    for (int i = 0; i < blk; i++){
        for (int j = 0; j < blk; j++){
            float sum = 0.0;
            for (int k = 0; k < blk; k++){
                sum += a[i * blk + k] * b[k * blk + j];
            }
        
        c[i*blk+j] = sum;
        }
    }
}

// Printing the Matrix
void PrintMatrix(const char* label, float* mat, int blk) {
    printf("\n%s:\n", label);
    for (int i = 0; i < blk; i++) {
        for (int j = 0; j < blk; j++) {
            printf("%6.2f ", mat[i * blk + j]);
        }
        printf("\n");
    }
}

// Main Solution
int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s Matrix Size\n", argv[0]);
        exit(1);
    }

    int matrixSize = atoi(argv[1]);
    float *a = (float*)malloc(sizeof(float) * matrixSize * matrixSize);
    float *b = (float*)malloc(sizeof(float) * matrixSize * matrixSize);
    float *c = (float*)malloc(sizeof(float) * matrixSize * matrixSize);

    if (!(a && b && c)) {
        fprintf(stderr, "%s: Argument is too large. Ran out of memory! Try again with smaller values.\n", argv[0]);
        free(a); free(b); free(c);
        exit(2);
    }

    srand(time(NULL)); // Seed rand
    InitBlock(a, b, c, matrixSize);

    // Print The Matrix
    PrintMatrix("Matrix A", a, matrixSize); // Matrix A Printed Out
    PrintMatrix("Matrix B", b, matrixSize); // Matrix B Printed Out

    // Time
        // This is implementing the OpenMP Time Library
        // omp_get_wtime(): Runs in seconds, is converted later on to ms using conversion methods. 
    double startTime = omp_get_wtime(); // Time Starts
    BlockMult(a,b,c,matrixSize);
    double endTime = omp_get_wtime(); // Time Ends

    double totalTime = endTime - startTime; // The time  being calculted

    PrintMatrix("Result Matrix C", c, matrixSize); // Matrix C Printed Out

    printf("\nDone. Matrix multiplication took %.6f miliseconds.\n", totalTime * 1000.0); // Convert time to ms

    free(a); free(b); free(c);
    return 0;
}
