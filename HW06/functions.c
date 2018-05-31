#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "functions.h"

//compute a*b mod p safely
ulong modprod(ulong a, ulong b, ulong p) {
  ulong za = a;
  ulong ab = 0;

  while (b > 0) {
    if (b%2 == 1) ab = (ab +  za) % p;
    za = (2 * za) % p;
    b /= 2;
  }
  return ab;
}

//compute a^b mod p safely
ulong modExp(ulong a, ulong b, ulong p) {
  ulong z = a;
  ulong aExpb = 1;

  while (b > 0) {
    if (b%2 == 1) aExpb = modprod(aExpb, z, p);
    z = modprod(z, z, p);
    b /= 2;
  }
  return aExpb;
}

//returns either 0 or 1 randomly
ulong randomBit() {
  return rand()%2;
}

//returns a random integer which is between 2^{n-1} and 2^{n}
ulong randXbitInt(unsigned int n) {
  ulong r = 1;
  for (ulong i=0; i<n-1; i++) {
    r = r*2 + randomBit();
  }
  return r;
}

//tests for primality and return 1 if N is probably prime and 0 if N is composite
ulong isProbablyPrime(ulong N) {

  if (N%2==0) return 0; //not interested in even numbers (including 2)

  int NsmallPrimes = 168;
  int smallPrimeList[168] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 
                                37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 
                                79, 83, 89, 97, 101, 103, 107, 109, 113, 
                                127, 131, 137, 139, 149, 151, 157, 163, 
                                167, 173, 179, 181, 191, 193, 197, 199, 
                                211, 223, 227, 229, 233, 239, 241, 251, 
                                257, 263, 269, 271, 277, 281, 283, 293, 
                                307, 311, 313, 317, 331, 337, 347, 349, 
                                353, 359, 367, 373, 379, 383, 389, 397, 
                                401, 409, 419, 421, 431, 433, 439, 443, 
                                449, 457, 461, 463, 467, 479, 487, 491, 
                                499, 503, 509, 521, 523, 541, 547, 557, 
                                563, 569, 571, 577, 587, 593, 599, 601, 
                                607, 613, 617, 619, 631, 641, 643, 647, 
                                653, 659, 661, 673, 677, 683, 691, 701, 
                                709, 719, 727, 733, 739, 743, 751, 757, 
                                761, 769, 773, 787, 797, 809, 811, 821, 
                                823, 827, 829, 839, 853, 857, 859, 863, 
                                877, 881, 883, 887, 907, 911, 919, 929, 
                                937, 941, 947, 953, 967, 971, 977, 983, 
                                991, 997};

  //before using a probablistic primality check, check directly using the small primes list
  for (int n=1;n<NsmallPrimes;n++) {
    if (N==smallPrimeList[n])   return 1; //true
    if (N%smallPrimeList[n]==0) return 0; //false
  }

  //if we're testing a large number switch to Miller-Rabin primality test
  /* Q2.1: Complete this part of the isProbablyPrime function using the Miller-Rabin pseudo-code */
  ulong r = 0;
  ulong d = N-1;
  while (d%2 == 0) {
    d /= 2;
    r += 1;
  }

  for (int n=0;n<NsmallPrimes;n++) {
    ulong k = smallPrimeList[n];
    ulong x = modExp(k,d,N);

    if ((x==1) || (x==N-1)) continue;

    for (int i=1;i<r-1;i++) {
      x = modprod(x,x,N);
      if (x == 1) return 0; //false
      if (x == N-1) break;
    }
    // see whether we left the loop becasue x==N-1
    if (x == N-1) continue; 

    return 0; //false
  }
  return 1; //true
}

//Finds a generator of Z_p using the assumption that p=2*q+1
ulong findGenerator(ulong p) {
  ulong g;
  ulong q = (p-1)/2;

  do {
    //make a random number 1<= g < p
    g = randXbitInt(32)%p; //could also have passed n to findGenerator
  } while (g==0 || (modExp(g,q,p)==1) || (modExp(g,2,p)==1));
  
  return g;
}

void setupElGamal(unsigned int n, ulong *p, ulong *g, 
                                  ulong *h, ulong *x) {

  /* Use isProbablyPrime and randomXbitInt to find a new random n-bit prime number 
     which satisfies p=2*q+1 where q is also prime */
  ulong q;
  do {
    *p = randXbitInt(n);
    q = (*p-1)/2;
  } while (!isProbablyPrime(*p) || !isProbablyPrime(q));


  /* Use the fact that p=2*q+1 to quickly find a generator */
  *g = findGenerator(*p);
  
  //pick a secret key,  x
  *x = randXbitInt(n)%(*p);

  //compute h
  *h = modExp(*g,*x,*p);
  
  printf("ElGamal Setup successful.\n");
  printf("p = %llu. \n", *p);  
  printf("g = %llu is a generator of Z_%llu \n", *g, *p);  
  printf("Secret key: x = %llu \n", *x);
  printf("h = g^x = %llu\n", *h);
  printf("\n");
}

void ElGamalEncrypt(ulong *m, ulong *a, unsigned int Nints, 
                    ulong p, ulong g, ulong h) {

  for (unsigned int i=0; i<Nints;i++) {
    //pick y in Z_p randomly
    ulong y;
    do {
      y = randXbitInt(32)%p;
    } while (y==0); //dont allow y=0

    //compute a = g^y
    a[i] = modExp(g,y,p);

    //compute s = h^y
    ulong s = modExp(h,y,p);

    //encrypt m by multiplying with s
    m[i] = modprod(m[i],s,p);  
  }
}

void ElGamalDecrypt(ulong *m, ulong *a, unsigned int Nints,
                    ulong p, ulong x) {

  for (unsigned int i=0; i<Nints;i++) {
    //compute s = a^x
    ulong s = modExp(a[i],x,p);

    //compute s^{-1} = s^{p-2}
    ulong invS = modExp(s,p-2,p);
    
    //decrypt message by multplying by invS
    m[i] = modprod(m[i],invS,p);
  }
}

//Pad the end of string so its length is divisible by Nchars
// Assume there is enough allocated storage for the padded string 
void padString(unsigned char* string, unsigned int charsPerInt) {

  int length    = mystrlen(string);
  int newlength = (length%charsPerInt==0) ? length : length + charsPerInt-length%charsPerInt;

  for (int i=length; i<newlength; i++) {
    string[i] = ' ';
  }
  string[newlength] = '\0';
}


void convertStringToZ(unsigned char *string, unsigned int Nchars,
                      ulong  *Z, unsigned int Nints) {

  int charsPerInt = Nchars/Nints;

  for (int i=0; i<Nints; i++) {
    Z[i] = 0;
    for (int n=0;n<charsPerInt;n++) {
      Z[i] *= 256; //shift left by 8 bits
      Z[i] += (ulong) string[i*charsPerInt + n]; //add the next character as a uint
    }
  }
}


void convertZToString(ulong  *Z,      unsigned int Nints,
                      unsigned char *string, unsigned Nchars) {

  int charsPerInt = Nchars/Nints;

  for (int i=0; i<Nints; i++) {
    ulong z = Z[i];
    for (int n=0;n<charsPerInt;n++) {
      string[i*charsPerInt + charsPerInt -1 - n] = z%256; // recover the character in the first 8 bits
      z /= 256; //shift right 8 bits
    }
  }
  string[Nints*charsPerInt] = '\0';
}

//cuda surprisingly doesn't have a strlen function for unsigned chars....
unsigned int mystrlen(unsigned char* string) {

  int i=0;
  while (string[i]!='\0') i++;
  return i;
}

int compareValue(const void *a, const void *b) {

  keyValuePair *ka = (keyValuePair *) a;
  keyValuePair *kb = (keyValuePair *) b;

  return (*ka).value > (*kb).value;
}

unsigned int binarySearch(keyValuePair* G, unsigned int M, ulong beta) {

  unsigned int low = 0;
  unsigned int high = M-1;

  while(1) {
    if (low>=high) return 0;

    unsigned int i = low + (high-low)/2;
    if (G[i].value == beta) return G[i].key;

    if (G[i].value < beta) 
      low = i+1;
    else
      high = i;
  }
}