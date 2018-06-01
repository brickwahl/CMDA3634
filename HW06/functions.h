#define ulong unsigned long long int

typedef struct 
{
	unsigned int key;
	ulong value;
} keyValuePair;

//compute a*b mod p safely
ulong modprod(ulong a, ulong b, ulong p);

//compute a^b mod p safely
ulong modExp(ulong a, ulong b, ulong p);

//returns either 0 or 1 randomly
ulong randomBit();

//returns a random integer which is between 2^{n-1} and 2^{n}
ulong randXbitInt(unsigned int n);

//tests for primality and return 1 if N is probably prime and 0 if N is composite
ulong isProbablyPrime(ulong N);

//Finds a generator of Z_p using the assumption that p=2*q+1
ulong findGenerator(ulong p);

//Sets up an ElGamal cryptographic system
void setupElGamal(unsigned int n, ulong *p, ulong *g, 
                                  ulong *h, ulong *x);

//encrypt a number *m using ElGamal and return the 
//  coefficient *a used in the encryption.
void ElGamalEncrypt(ulong *m, ulong *a, unsigned int Nints,
                    ulong p, ulong g, ulong h);

//decryp a number *m using ElGamal using the coefficent
//  *a and the secret key x.
void ElGamalDecrypt(ulong *m, ulong *a, unsigned int Nints, ulong p, ulong x);


//Pad the end of string so its length is divisible by Nchars
// Assume there is enough allocated storage for the padded string 
void padString(unsigned char* string, unsigned int charsPerInt);

void convertStringToZ(unsigned char *string, unsigned int Nchars,
                      ulong  *Z,      unsigned int Nints);

void convertZToString(ulong  *Z,      unsigned int Nints,
                      unsigned char *string, unsigned int Nchars);

//cuda surprisingly doesn't have a strlen function for unsigned chars....
unsigned int mystrlen(unsigned char* string);

int compareValue(const void *a, const void *b);

unsigned int binarySearch(keyValuePair* G, unsigned int M, ulong alpha);
