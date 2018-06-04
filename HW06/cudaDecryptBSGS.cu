#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "functions.c"
#include "cuda.h"

//compute a*b mod p safely
__device__ ulong d_modprod(ulong a, ulong b, ulong p) {
  ulong za = a;
  ulong ab = 0;

  while (b > 0) {
    if (b%2==1) ab = (ab +  za) % p;
    za = (2*za) % p;
    b /= 2;
  }
  return ab;
}

//compute a^b mod p safely
__device__ ulong d_modExp(ulong a, ulong b, ulong p) {
  ulong z = a;
  ulong aExpb = 1;

  while (b > 0) {
    if (b%2==1) aExpb = d_modprod(aExpb, z, p);
    z = d_modprod(z, z, p);
    b /= 2;
  }
  return aExpb;
}

__device__ unsigned int d_binarySearch(keyValuePair* G, unsigned int M, ulong beta) {
  
  unsigned int low = 0;
  unsigned int high = M-1;

  while(1) {
    if (low>=high) return 0;

    unsigned int i = low + (high-low)/2;
    if (G[i].value == beta) return G[i].key;

    if (G[i].value < beta) 
      low = i + 1;
    else
      high = i;
  }
}

__global__ void findSecretKey(ulong p,ulong g, ulong h, ulong *x, ulong alpha, keyValuePair G) {

  int threadid = threadIdx.x;
  int blockid = blockIdx.x;
  int Nblock = blockDim.x;
  
  ulong beta = d_modprod(h, d_modExp(alpha, id, p), p);
  unsigned int id = threadid + blockid*Nblock;
  x[id] = 0;

  if (id < p) && (d_binarySearch(G, M, beta) !=0) {
    x[id] = id*M + id;
    printf("Found it: %llu\n",x[id]);  
}



int main (int argc, char **argv) {

  //declare storage for an ElGamal cryptosytem
  unsigned int n;
  ulong p, g, h, x;

  printf("Reading file.\n");
  FILE *file;
  
  file = fopen("public_key.txt", "r");

  int status;
  status = fscanf(file, "%u\n", &n);
  status = fscanf(file, "%llu\n", &p);
  status = fscanf(file, "%llu\n", &g);
  status = fscanf(file, "%llu\n", &h);
  fclose(file);


  unsigned int Nints;
  file = fopen("message.txt", "r");
  status = fscanf(file, "%u\n", &Nints);

  //storage for message as elements of Z_p
  ulong *Zmessage = 
      (ulong *) malloc(Nints*sizeof(ulong)); 
  
  //storage for extra encryption coefficient 
  ulong *a = 
      (ulong *) malloc(Nints*sizeof(ulong)); 

  for (int n=0;n<Nints;n++) {
    status = fscanf(file, "%llu %llu\n", Zmessage+n, a+n);
  }
  fclose(file);

  unsigned int charsPerInt = (n-1)/8;

  unsigned int Nchars = charsPerInt*Nints;
  unsigned char *message = (unsigned char *) malloc(Nchars*sizeof(unsigned char));

  //get the secret key from the user
  printf("Enter the secret key (0 if unknown): "); fflush(stdout);
  char stat = scanf("%llu",&x);

  //use cuda to find the secret key
  if (x==0) {
    unsigned int M = sqrt(p); //giant step

    ulong *d_x;
    cudaMalloc(&d_x,1*sizeof(ulong));

    keyValuePair *d_G;
    cudaMalloc(&d_G, (M+1)*sizeof(keyValuePair);

    ulong *d_alpha;
    cudaMalloc(&d_alpha, 1*sizeof(ulong));

    keyValuePair *G = malloc((M+1)*sizeof(keyValuePair));

    for *unsigned int i = 1; i <= M; i++) {
      G[i].key = i;
      G[i].value = d_modExp(g, i, p);

    qsort(G, M, sizeof(keyValuePair), compareValue);    

    ulong alpha = modExp(modExp(g,M,p),p-2,p);

    cudaMemcpy(&G, d_G, (M+1_*sizeof(keyValuePair), cudaMemcpyHostToDevice);
    cudaMemcpy(&alpha, d_alpha, 1*sizeof(ulong), cudaMemcpyHostToDevice);

    int Nthreads = 256;

    //block dimensions
    dim3 B(Nthreads,1,1);

    //grid dimensions
    dim3 G((p/M+Nthreads-1)/Nthreads,1,1);

    printf("Finding the secret key...\n");
    double startTime = clock();
    findSecretKey<<<G,B>>>(p,g,h,d_x, d_alpha, d_G);
    cudaMemcpy(&x,d_x,1*sizeof(ulong),cudaMemcpyDeviceToHost);
    printf("Secret key x = %llu \n", x);

    double endTime = clock();

    double totalTime = (endTime-startTime)/CLOCKS_PER_SEC;
    double work = (double) p;
    double throughput = work/totalTime;

    printf("Searching all keys took %g seconds, throughput was %g values tested per second.\n", totalTime, throughput);
    cudaFree(d_x);
    cudaFree(d_alpha);
    cudaFree(d_G);
  }

  //Decrypt the Zmessage with the ElGamal cyrptographic system
  ElGamalDecrypt(Zmessage,a,Nints,p,x);
  convertZToString(Zmessage, Nints, message, Nchars);

  printf("Decrypted Message = \"%s\"\n", message);
  printf("\n");
 
  free(message);
  free(Zmessage);
  free(a);

  return 0;
}
