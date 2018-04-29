#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "functions.h"


int main (int argc, char **argv) {

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

  convertZtoString(m, Nints, message, Nchars);
  printf("Decrypted message = \"%s\"\n", message);
  return 0;
}
