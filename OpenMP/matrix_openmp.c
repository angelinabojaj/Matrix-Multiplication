#include <stdio.h>
#include <stdlib.h>     // For malloc, free, rand, atoi, exit
#include <time.h>       // WE ADDED! FOR TIMING
#include <omp.h> // Library for OpenMP

void InitBlock(float *a, float *b, float *c, int blk) {
    int len = blk * blk;
    for (int ind = 0; ind < len; ind++) {
        a[ind] = (float)(rand() % 1000) / 100.0;
        b[ind] = (float)(rand() % 1000) / 100.0; //WE CHANGED! TO MAKE BOTH MATRICIES RANDOM
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
    if (argc != 3) {
        fprintf(stderr, "usage: mmult m blk\n");
        exit(1);
    }

    int sidesize = atoi(argv[1]) * atoi(argv[2]);
    float *a = (float*)malloc(sizeof(float) * sidesize * sidesize);
    float *b = (float*)malloc(sizeof(float) * sidesize * sidesize);
    float *c = (float*)malloc(sizeof(float) * sidesize * sidesize);

    if (!(a && b && c)) {
        fprintf(stderr, "%s: out of memory!\n", argv[0]);
        free(a); free(b); free(c);
        exit(2);
    }

    srand(time(NULL)); // Seed rand
    InitBlock(a, b, c, sidesize);

    // Print The Matrix
    PrintMatrix("Matrix A", a, sidesize);
    PrintMatrix("Matrix B", b, sidesize);

    // Time
        // This is implementing the OpenMP Time Library
    double startTime = omp_get_wtime();
    BlockMult(a,b,c,sidesize);
    double endTime = omp_get_wtime();

    double totalTime = endTime - startTime; // The time  being calculted

    PrintMatrix("Result Matrix C", c, sidesize);

    printf("\nDone. Matrix multiplication took %.6f seconds.\n", totalTime);

    free(a); free(b); free(c);
    return 0;
}
