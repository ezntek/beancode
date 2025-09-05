#include <math.h>
#include <stdio.h>

#define NUM 750000

int is_prime(int n) {
    for (int i = 2; i <= (int)sqrt(n); i++) {
        if (n % i == 0)
            return 0;
    }

    return 1;
}

void prime_torture(int n) {
    for (int i = 1; i <= n; i++) {
        if (is_prime(i))
            printf("%d, ", i);
        fflush(stdout);
    }
}

int main(void) {
    prime_torture(NUM);
    puts("");
    return 0;
}
