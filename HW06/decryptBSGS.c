#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "functions.h"


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
        printf("Populating array of keyValuePairs...\n");

	unsigned int M = sqrt(p); //giant step
	//printf("M equals %u\n", M);
	keyValuePair *G =  malloc((M+1)*sizeof(keyValuePair)); //declaring and allocating memory for array
	//printf("g equals %llu\n", g);

	for (unsigned int i=1; i<=M; i++) {
		// printf("i equals %u\n", i);
		G[i].key = i;
		// printf("G[%u].key equals %u\n", i, G[i].key);
		G[i].value = modExp(g, i, p);
		//G[i].value = pow(g, i);
		//printf("G[%u].value equals %llu\n", i, G[i].value); 
	}

	qsort(G, M, sizeof(keyValuePair), compareValue);
	//ulong alpha = modExp(g, M, (p-2));
	ulong a1 = modExp(g, M, p);
	ulong alpha = modExp(a1, (p-2), p);

	//printf("alpha before equals %llu\n", alpha);
	//ulong alpha = (g^M)^(-1);
	// printf("alpha equals %llu\n", alpha);
	// find the secret key
	printf("Finding the secret key...\n");
	double startTime = clock();
		// for (unsigned int i=0;i<p-1;i++) {
		// if (modExp(g,i+1,p)==h) {
		// printf("Secret key found! x = %u \n", i+1);
		// x=i+1;
		// } 
		// }
	ulong beta;
	unsigned int j;
	for (unsigned int i = 0; i <= p/M; i++) {
		ulong temp = modExp(alpha, i, p);
		beta = modprod(h, temp, p);
		// beta = pow(h*alpha, i);
	        // printf("Beta equals %llu\n", beta);
		j =  binarySearch(G, M, beta);
		if (j != 0) {
			//printf("i equals %u\n", i);
			x = modprod(i,M,p);
			x = x + j;
			//x = i*M + j;
			break;       
		}
	}
	//printf("beta equals %llu at the end\n", beta);
	//printf("j equals %u\n", j);
	//printf("x equals %llu\n", x);
	// x = x%p;
	// printf("x equals %llu after mod\n", x);  

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
	free(G);

	return 0;
}
