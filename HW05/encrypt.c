#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "functions.h"

int main (int argc, char **argv) {

	//seed value for the randomizer 
  double seed = clock(); //this will make your program run differently everytime
  //double seed = 0; //uncomment this and your program will behave the same everytime it's run

  srand(seed);

  int bufferSize = 1024;
  unsigned char *message = (unsigned char *) malloc(bufferSize*sizeof(unsigned char));

  printf("Enter a message to encrypt: ");
  int stat = scanf (" %[^\n]%*c", message); //reads in a full line from terminal, including spaces

  //declare storage for an ElGamal cryptosytem
  unsigned int n, p, g, h;

  printf("Reading file.\n");

  /* Q2 Complete this function. Read in the public key data from public_key.txt,
    convert the string to elements of Z_p, encrypt them, and write the cyphertexts to 
    message.txt */

  FILE *fpk2;
  if ((fpk2 = fopen("public_key.txt", "r")) == NULL) {
    printf("There was an error.");
    exit(1);
  }

  fscanf(fpk2, "%u\n", &n);
  fscanf(fpk2, "%u\n", &p);
  fscanf(fpk2, "%u\n", &g);
  fscanf(fpk2, "%u\n", &h);
  fclose(fpk2);

  unsigned int charsPerInt = (int) ((n-1)/8);

  padString(message, charsPerInt);

  unsigned int Nchars = strlen(message);
  unsigned int Nints = strlen(message)/charsPerInt;
  unsigned int *Zmessage = (unsigned int *) malloc(Nints*sizeof(unsigned int));
  unsigned int *a = (unsigned int *) malloc(Nints*sizeof(unsigned int));

  printf("%u\n", Nints);

  convertStringToZ(message, Nchars, Zmessage, Nints);

  ElGamalEncrypt(Zmessage, a, Nints, p, g, h);

  FILE *mes;
  if ((mes = fopen("message.txt", "w+")) == NULL) {
    printf("There was an error.");
    exit(1);
  }

  fprintf(mes, "%u\n", Nints);
  for (unsigned int i = 0; i < Nints; i++) {
    fprintf(mes, "%u %u\n", Zmessage[i], a[i]);
  }

  fclose(mes); 
 
  return 0;
}
