#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "cuda.h"
#include "functions.c"

//compute a*b mod p safely
__device__ unsigned int modprodCuda(unsigned int a, unsigned int b, unsigned int p) {
  unsigned int za = a;
  unsigned int ab = 0;

  while (b>0) {
    if (b%2 == 1) {
      ab = (ab + za) % p;
    }
      za = (2 * za) % p;
      b /= 2;
  }
  return ab;
}

//compute a^b mod p safely
__device__ unsigned int modexpCuda(unsigned int a, unsigned int b, unsigned int p) {
  unsigned int z = a;
  unsigned int aexpb = 1;

  while (b > 0) {
    if (b%2 == 1) {
      aexpb = modprodCuda(aexpb, z, p);
    }
  z = modprodCuda(z, z, p);
  b /=2;
  }
  return aexpb;
}

__global__ void findX(volatile unsigned int *xres, unsigned int p, unsigned int h, unsigned int g) {
  
  int threadid = threadIdx.x; 
  int blockid = blockIdx.x;
  int Nblock = blockDim.x;

  unsigned int id = threadid + blockid*Nblock;
  xres[id+1]=0;
  if ((id < p) && modExpCuda(g, id + 1, p)==h) {
    xres[id+1] = id+1;
    printf("we found it. %d %d %d\n", id+1, xres[id+1], xres[id]);
    }
  
int main (int argc, char **argv) {

  /* Part 2. Start this program by first copying the contents of the main function from 
     your completed decrypt.c main function. */
   //declare storage for an ElGamal cryptosytem
  unsigned int n, p, g, h, x;
  unsigned int Nints;

  //get the secret key from the user
  printf("Enter the secret key (0 if unknown): "); fflush(stdout);
  char stat = scanf("%u",&x);

  printf("Reading file.\n");

  /* Q3 Complete this function. Read in the public key data from public_key.txt
    and the cyphertexts from messages.txt. */

  FILE *firstFile;
  if((firstFile = fopen("public_key.txt", "r")) == NULL) {
    printf("There was an error.\n");
    exit(1);
  }
}
  fscanf(firstFile, "%u\n", &n);
  fscanf(firstFile, "%u\n", &p);
  fscanf(firstFile, "%u\n", &g);
  fscanf(firstFile, "%u\n", &h);
  fclose(firstFile);

  FILE *secondFile;
  if((secondFile = fopen("message.txt", "r")) == NULL) {
    printf("There was an error.\n");
    exit(1);
  }

  fscanf(secondFile, "%u\n", &Nints);

  unsigned int cnt = 0;
  unsigned int *m = (unsigned int *) malloc(Nints*sizeof(unsigned int));
  unsigned int *a = (unsigned int *) malloc(Nints*sizeof(unsigned int));

  while (cnt < Nints) {
    fscanf(secondFile, "%u %u\n", &m[cnt], &a[cnt]);
    cnt = cnt + 1;
  }

  fclose(secondFile);

  // find the secret key
  if (x==0 || modExp(g,x,p)!=h) {
    printf("Finding the secret key...\n");
    double startTime = clock();
    for (unsigned int i=0;i<p-1;i++) {
      if (modExp(g,i+1,p)==h) {
        printf("Secret key found! x = %u \n", i+1);
        x=i+1;
      }
    }
    double endTime = clock();

    double totalTime = (endTime-startTime)/CLOCKS_PER_SEC;
    double work = (double) p;
    double throughput = work/totalTime;

    printf("Searching all keys took %g seconds, throughput was %g values tested per second.\n", totalTime, throughput);
  }

  /* Q3 After finding the secret key, decrypt the message */

  ElGamalDecrypt(m, a, Nints, p, x);

  int bSize = 1024;
  unsigned char *message = (unsigned char *) malloc(bSize*sizeof(unsigned char));
  unsigned int charsPerInt = (int) ((n - 1)/8);
  unsigned int Nchars = (Nints * charsPerInt);

  convertZToString(m, Nints, message, Nchars);
  printf("Decrypted message = \"%s\"\n", message);
  return 0;
                                                            
  /* Q4 Make the search for the secret key parallel on the GPU using CUDA. */

  double starting = clock();
  unsigned int *x_res;
  cudaMalloc(&x_res, (p-1)*sizeof(unsigned int));

  int Nthreads = 128;
  int Nblocks = (p + Nthreads - 1) / Nthreads;

  findX <<<Nblocks, Nthreads>>> (x_res, p, h, g);
  cudaDeviceSynchronize();
  
  unsigned int *result - (unsigned int*) malloc((p-1(*sizeof(unsigned int));
  cudaMemcpy(result, x_res, (p-1)*sizeof(unsigned int), cudaMemcpyDeviceToHost);

  int ctr = 0;
  while (ctr < p) {
    if (result[ctr] != 0) {
      x = result[ctr];
    }
  ctr = ctr + 1;
  }

  printf("the secret key is %d\n", x);

  double endTime = clock();
  double totalTimec = (endTime - starting)/CLOCKS_PER_SEC;
  double workc = (double) p;
  double throughputc = workc/totalTimec;
}
