import sys

a = sys.argv[1]
b = sys.argv[2]

a = float(a)
b = float(b)

print(round(100. * (b-a)/a, 2))
