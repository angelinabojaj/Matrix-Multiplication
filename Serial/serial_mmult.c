  GNU nano 2.9.8                                                    serial_mmult.c

/* 
   Matrix Multiply

   Originally:
   From PVM: Parallel Virtual Machine
   A Users' Guide and Tutorial for Networked Parallel Computing

   Geist et al

   Reduced to a serial program for comparison.

*/

// Modified for simplified baseline use
// call with ./serial_mmult 1 <matrix size>
    
#include <stdio.h>
#include <stdlib.h>     // For malloc, free, rand, atoi, exit
#include <time.h>       // WE ADDED! FOR TIMING

void InitBlock(float *a, float *b, float *c, int blk) {
    int len = blk * blk;
    for (int ind = 0; ind < len; ind++) {
        a[ind] = (float)(rand() % 1000) / 100.0;
        b[ind] = (float)(rand() % 1000) / 100.0; //WE CHANGED! TO MAKE BOTH MATRICIES RANDOM
        c[ind] = 0.0;
    }
}

void BlockMult(float* c, float* a, float* b, int blk) {
    for (int i = 0; i < blk; i++)
        for (int j = 0; j < blk; j++)
            for (int k = 0; k < blk; k++)
                c[i * blk + j] += (a[i * blk + k] * b[k * blk + j]);
}

/* WE ADDED! FOR PRINT STATEMENTS */
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

    /* WE CHANGED TO PRINT */
    PrintMatrix("Matrix A", a, sidesize);
    PrintMatrix("Matrix B", b, sidesize);

    /* WE CHANGED TO TRACK THE TIMING */
    clock_t start = clock();
    BlockMult(c, a, b, sidesize);
    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    PrintMatrix("Result Matrix C", c, sidesize);

    printf("\nDone. Matrix multiplication took %.6f seconds.\n", time_spent);

    free(a); free(b); free(c);
    return 0;
}
