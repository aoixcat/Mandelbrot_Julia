# -*- coding: utf-8 -*-

# -1.254 0.024

import numpy as np
import matplotlib.pyplot as plt

length = 1000
roop = 80
threshold = 2
image = [[i for i in range(length)] for j in range(length)]

"""
range_x = [-2, 1]
range_y = [-1.5, 1.5]
"""
range_x = np.array([0.1405, 0.1405])
range_y = np.array([0.648, 0.648])

dif_x = range_x[1] - range_x[0]
dif_y = range_y[1] - range_y[0]

def init():    
    for i in range(length):
        if (i % 100 == 0):
            print("i = " + str(i))
        for j in range(length):  
            image[j][i] = mandelbrot(range_x[0] + (dif_x/length) * i, range_y[0] + (dif_y/length) * j)
            
            
def mandelbrot(R, C):
    z = 0 + 0j
    _z = 0 + 0j
    C = C * 1j
    for i in range(roop):
        _z = z * z + R + C
        if (np.abs(_z) >= threshold):
            #return 0
            return i
        z = _z
    return np.abs(_z)
    
if __name__ == "__main__":
    init()
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.savefig("m2.png", transparent=True)
    
