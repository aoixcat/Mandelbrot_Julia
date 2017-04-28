import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
length * 2 = block + grid
"""
length = 1 << 10
block_x = 1 << 10
block_y = 1
grid_x = 1 << 10
grid_y = 1

roop = 80
threshold = 2.0
frames = 2000
scaling = 0.994

r_x = np.array([-2.0, 1.0])
r_y = np.array([-1.5, 1.5])

"""
1
range_x = np.array([-1.2538, -1.2538])
range_y = np.array([0.0245, 0.0245])
"""
"""
2
"""
range_x = np.array([0.14033, 0.14033])
range_y = np.array([0.6479, 0.6479])
"""
3
range_x = np.array([-1.7654494830546, -1.7654494830546])
range_y = np.array([0.0349703607266, 0.0349703607266])
"""
"""
4
range_x = np.array([1.3946584121866, 1.3946584121866])
range_y = np.array([0.0018083625612, 0.0018083625612])
"""

d_x = r_x - range_x
d_y = r_y - range_y

mod = SourceModule("""

__device__ void subst(float *a, float *b)
{
     a[0] = b[0];
     a[1] = b[1];
}

__device__ void cadd(float *a, float *b, float *c)
{
    c[0] = a[0] + b[0];
    c[1] = a[1] + b[1];
}

__device__ void cmul(float *a, float *b, float *c)
{
    c[0] = a[0] * b[0] - a[1] * b[1];
    c[1] = a[0] * b[1] + a[1] * b[0];
}

 __global__ void manderblot(float *x, float *y, int *image, int N, int roop, float threshold)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    float z[2] = {0, 0};
    float _z[2] = {0, 0};
    int i;
    float c[2] = {x[idx % N], y[idx / N]};
    for (i = 0; i < roop; ++i)
    {
        cmul(z,z,_z);
        cadd(_z, c, _z);
        if (_z[0] * _z[0] + _z[1] * _z[1] > threshold * threshold){
            image[idx] = i;
            return;
        }
        subst(z, _z);
    }
    image[idx] = 0;
    return;
}

""")


def init(i, frames, length, roop, threshold, block_x, block_y):
    if i != 0:
        plt.cla()
    
    if (i % 10 == 0):
        print("進捗：　" + str(i*100/frames) + "%")
    
    image_cpu = np.zeros(length**2).astype(np.int32)
    image_x_cpu = np.zeros(length).astype(np.float32)
    image_y_cpu = np.zeros(length).astype(np.float32)
    
    """
    image_gpu = cuda.mem_alloc(image_cpu.nbytes)
    image_x_gpu = cuda.mem_alloc(image_x_cpu.nbytes)
    image_y_gpu = cuda.mem_alloc(image_y_cpu.nbytes)
    """
    
    func = mod.get_function("manderblot")
    
    
    x = r_x - (1.0 - scaling**i) * d_x
    y = r_y - (1.0 - scaling**i) * d_y
    
    dif_x = x[1] - x[0]
    dif_y = y[1] - y[0]
    for i in range(length):
        image_x_cpu[i] = x[0] + dif_x * i / (length - 1)
    for j in range(length):
        image_y_cpu[j] = y[0] + dif_y * j / (length - 1)
    
    
    image_gpu = gpuarray.to_gpu(image_cpu)
    image_x_gpu = gpuarray.to_gpu(image_x_cpu)
    image_y_gpu = gpuarray.to_gpu(image_y_cpu)
    
    """
    cuda.memcpy_htod(image_gpu, image_cpu) 
    cuda.memcpy_htod(image_x_gpu, image_x_cpu) 
    cuda.memcpy_htod(image_y_gpu, image_y_cpu)
    """
    
    func(image_x_gpu, image_y_gpu, image_gpu, np.int32(length), np.int32(roop), np.float32(threshold), block=(block_x, block_y, 1), grid=(grid_x, grid_y))
    #cuda.memcpy_dtoh(image_cpu, image_gpu)
    
    image_cpu = image_gpu.get()
    plt.axis('off')
    
    #ims.append([plt.imshow(image_cpu.reshape((length, length)), cmap="afmhot")])
    #plt.imshow(image_cpu.reshape((length, length)), cmap="afmhot")
    plt.imshow(image_cpu.reshape((length, length)), cmap="viridis")

def main():    
    
    
    #ims = []
    fig = plt.figure(figsize=(10, 10))
    
    """
    for i in range(frames):
        init(i, frames, ims, length, roop, threshold, block_x, block_y)
    """
    
    
    #anim = animation.ArtistAnimation(fig, ims, interval=30)
    anim = animation.FuncAnimation(fig, init, fargs=(frames, length, roop, threshold, block_x, block_y), frames=frames, interval=10)
    print("Saving mp4...")
    anim.save('mandelbrot2.mp4')

if __name__ == "__main__":
    main()

