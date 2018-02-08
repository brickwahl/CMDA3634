#include<stdio.h>
#include<stdlib.h>
#include<math.h>

void main() {

  int x;
  int y;
  int gcd = 1;

  printf("Enter the first number: ");
  scanf("%d", &x);

  printf("Enter the second number: ");
  scanf("%d", &y);

  for (int i = 1; i <= x && i <= y; i++) { //If the inputs, x or y, become very large then th                                           //e program might hang due to computational load
    if ((x % i == 0 ) && (y % i == 0)) {
      gcd = i;
    }
  }

  printf("The greatest common divisor of %d and %d is %d.\n", x, y, gcd);

}
