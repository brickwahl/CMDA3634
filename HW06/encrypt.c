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
  unsigned int n;
  ulong p, g, h;

  printf("Reading file.\n");
  FILE *file;
  
  file = fopen("public_key.txt", "r");

  int status;
  status = fscanf(file, "%u\n", &n);
  status = fscanf(file, "%llu\n", &p);
  status = fscanf(file, "%llu\n", &g);
  status = fscanf(file, "%llu\n", &h);
  fclose(file);

  unsigned int charsPerElement = (n-1)/8;

  padString(message, charsPerElement);
  printf("Padded Message = \"%s\"\n", message);

  unsigned int Nchars = mystrlen(message);
  unsigned int Nints  = mystrlen(message)/charsPerElement;

  //storage for message as elements of Z_p
  ulong *Zmessage = 
      (ulong *) malloc(Nints*sizeof(ulong)); 
  
  //storage for extra encryption coefficient 
  ulong *a = 
      (ulong *) malloc(Nints*sizeof(ulong)); 

  // cast the string into an unisigned int array
  convertStringToZ(message, Nchars, Zmessage, Nints);
  
  //Encrypt the Zmessage with the ElGamal cyrptographic system
  ElGamalEncrypt(Zmessage,a,Nints,p,g,h);

  printf("Writing to file.\n");
  
  file = fopen("message.txt", "w");
  fprintf(file, "%u\n", Nints);
  for (int n=0;n<Nints;n++) {
    fprintf(file, "%llu %llu\n", Zmessage[n], a[n]);
  }
  fclose(file);

  free(message);
  free(Zmessage);
  free(a);

  return 0;
}
