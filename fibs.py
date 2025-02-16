x = 0
y = 1

for i in range(20000):
    z = x + y
    print(z)
    y = x
    x = z
