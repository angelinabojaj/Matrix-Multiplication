/* 
   Matrix Multiply

   Originally:
   From PVM: Parallel Virtual Machine
   A Users' Guide and Tutorial for Networked Parallel Computing

   Geist et al

   Reduced to a serial program for comparison.

*/

#include <stdio.h>

void
InitBlock(float *a, float *b, float *c, int blk)
{
        int len, ind;
        int i,j;

        len = blk*blk;
        for (ind = 0; ind < len; ind++) 
                { a[ind] = (float)(rand()%1000)/100.0; c[ind] = 0.0; }
        for (i = 0; i < blk; i++) {
                for (j = 0; j < blk; j++)
                        b[j*blk+i] = (i==j)? 1.0 : 0.0;
        }
}

void
BlockMult(float* c, float* a, float* b, int blk) 
{
        int i,j,k;

        for (i = 0; i < blk; i++)
                for (j = 0; j < blk; j ++)
                        for (k = 0; k < blk; k++)
                                c[i*blk+j] += (a[i*blk+k] * b[k*blk+j]);
}

int
main(int argc, char* argv[])
{

        float *a, *b, *c;

        /* Compute the array dim from the same parameters. */
        int sidesize;
        if (argc == 3)
                sidesize = atoi(argv[1]) * atoi(argv[2]);

        else {
                fprintf(stderr, "usage: mmult m blk\n");
                exit(1);
        }

        /* allocate the memory for the arrays. */
        a = (float*)malloc(sizeof(float)*sidesize*sidesize);
        b = (float*)malloc(sizeof(float)*sidesize*sidesize);
        c = (float*)malloc(sizeof(float)*sidesize*sidesize);
        /* check for valid pointers */
        if (!(a && b && c)) { 
                fprintf(stderr, "%s: out of memory!\n", argv[0]);
                free(a); free(b); free(c);
                exit(2);
        }

        /* initialize the arrays */
        InitBlock(a, b, c, sidesize);

        /* Multiply. */
        BlockMult(c, a, b, sidesize);

        /* check it */
        int i;
        for (i = 0 ; i < sidesize*sidesize; i++) 
                if (a[i] != c[i]) 
                        printf("Error a[%d] (%g) != c[%d] (%g) \n", i, a[i],i,c[i]);

        printf("Done.\n");
        free(a); free(b); free(c);
        return 0;
}
