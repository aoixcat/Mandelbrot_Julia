# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

length = 1000
roop = 50
threshold = 2
C = -0.8 + 0.163j
image = [[i for i in range(length)] for j in range(length)]

range_x = [-2, 2]
range_y = [-2, 2]
dif_x = range_x[1] - range_x[0]
dif_y = range_y[1] - range_y[0]

def init():    
    for i in range(length):
        if (i % 10 == 0):
            print("i = " + str(i))
        for j in range(length):
           image[j][i] = julia(range_x[0] + (dif_x/length) * i, range_y[0] + (dif_y/length) * j)
            
            
def julia(zx, zy):
    z = zx + 1j * zy
    _z = 0 + 0j
    for i in range(roop):
        _z = z * z + C
        if (np.abs(_z) >= threshold):
            return i
            #return 0
        z = _z
    return np.abs(_z)
    
if __name__ == "__main__":
    init()
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="afmhot")
    plt.savefig("j3.png")
    
