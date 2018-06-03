#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

#include "functions.c"
#include "omp.h"


int main (int argc, char **argv) {

  int NumThreads = 24;
  omp_set_num_threads(NumThreads);

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

  ulong charsPerInt = (n-1)/8;

  ulong Nchars = charsPerInt*Nints;
  unsigned char *message = (unsigned char *) malloc(Nchars*sizeof(unsigned char));
 
  //get the secret key from the user
  printf("Enter the secret key (0 if unknown): "); fflush(stdout);
  char stat = scanf("%llu",&x);

  
  // find the secret key
    unsigned int M = sqrt(p); //giant step
    
    keyValuePair *G = malloc((M+1)*sizeof(keyValuePair)); //declaring & allocating mem for G

    for (unsigned int i = 1; i <= M; i++) {
      G[i].key = i;
      G[i].value = modExp(g, i, p);
    }
    
    qsort(G, M, sizeof(keyValuePair), compareValue);
    
    ulong a1 = modExp(g, M, p);
    ulong alpha = modExp(a1, (p-2), p);

    printf("Finding the secret key...\n"); 
    double startTime = clock();
    
    ulong beta;
    unsigned int j;
    
    volatile bool flag = false;
    #pragma omp parallel for shared(flag)
    for (unsigned int i = 0; i <= p/M; i++) {
      ulong temp = modExp(alpha, i, p);
      beta = modprod(h, temp, p);
      j = binarySearch(G, M, beta);
      if (j != 0) {
        x = modprod(i, M, p);
        x = x + j;
        flag = true;
      }
    }

    double endTime = clock();

    double totalTime = (endTime-startTime)/CLOCKS_PER_SEC;
    double work = (double) p;
    double throughput = work/totalTime;

    printf("Searching all keys took %g seconds, throughput was %g values tested per second.\n", totalTime, throughput);

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
