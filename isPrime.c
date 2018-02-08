#include<stdio.h>
#include<stdlib.h>
#include<math.h>

void main() {

  int a, i, x = 0;
  printf("Enter a number: ");
  scanf("%d",&a);

  for( i = 2; i < a / 2; i++) {

    if (a % i == 0) {
      x = 1;
    }

  }

  if (x == 0) printf("%d is prime.\n",a);
  else printf("%d is not prime.\n",a);
}
