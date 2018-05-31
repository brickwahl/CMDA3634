#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "functions.c"
#include "mpi.h"


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

  ulong charsPerInt = (n-1)/8;

  ulong Nchars = charsPerInt*Nints;
  unsigned char *message = (unsigned char *) malloc(Nchars*sizeof(unsigned char));

  //get the secret key from the user
  printf("Enter the secret key (0 if unknown): "); fflush(stdout);
  char stat = scanf("%llu",&x);

  
  // find the secret key
  if (x==0 || modExp(g,x,p)!=h) {
    printf("Finding the secret key...\n");

    unsigned int M; //giant step
    
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
