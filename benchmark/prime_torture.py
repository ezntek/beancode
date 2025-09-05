import sys

NUM = 750000

def is_prime(n: int) -> bool:
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

def prime_torture(n: int):
    for i in range(1, n+1):
        if is_prime(i):
            sys.stdout.write(str(i) + ", ")
            sys.stdout.flush()

prime_torture(NUM)
print()
