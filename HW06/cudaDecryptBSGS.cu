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

__global__ void findSecretKey(ulong p,ulong g, ulong h, ulong *x) {


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
  if (x==0 || modExp(g,x,p)!=h) {
    unsigned int M; //giant step

    ulong *d_x;
    cudaMalloc(&d_x,1*sizeof(ulong));

    ulong alpha = modExp(modExp(g,M,p),p-2,p);

    int Nthreads = 256;

    //block dimensions
    dim3 B(Nthreads,1,1);

    //grid dimensions
    dim3 G((p/M+Nthreads-1)/Nthreads,1,1);

    printf("Finding the secret key...\n");
    double startTime = clock();
    findSecretKey<<<G,B>>>(p,g,h,d_x);
    cudaMemcpy(&x,d_x,1*sizeof(ulong),cudaMemcpyDeviceToHost);

    printf("Secret key x = %llu \n", x);

    double endTime = clock();

    double totalTime = (endTime-startTime)/CLOCKS_PER_SEC;
    double work = (double) p;
    double throughput = work/totalTime;

    printf("Searching all keys took %g seconds, throughput was %g values tested per second.\n", totalTime, throughput);
    cudaFree(d_x);
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
