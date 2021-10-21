from os import read
import numpy as np

f = open('gt.txt', 'w')
t = [45, 54, 55]
for i in range(100):
    if i in t:
        f.write('1\n')
    else:
        f.write('0\n')
f.close()
