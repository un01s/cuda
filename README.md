# intro to CUDA

## add.cpp on CPU

```
$ g++ add.cpp -o add
$ time ./add 
Max error: 0
./add  0.02s user 0.00s system 9% cpu 0.298 total
```

### kernel

Turn the function ```add()``` into a function that the GPU can run, called a kernel in CUDA by adding a specifier ```__global__``` to the function. This specifier tells the CUDA C++ compiler that this is a function that runs on the GPU and can be called from CPU code.

```c++
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}
```

### device code vs. host code 

These ```__global__``` functions are known as kernels, and code that runs on the GPU is often called device code, while the code runs on the CPU is host code.

### unified memory in CUDA

Unified memory makes it easy by providing a single memory space accessible by all GPUs and CPUs in the system. Use ```cudaMallocManaged()``` to allocate memory in unified memory space. The returned pointer can be used by both CPU and GPU. To free the memory, just pass the pointer to ```cudaFree()```.

### launch the kernel, i.e. run the kernel function on the GPU

use the triple angle bracket syntax <<< >>> to launch the kernel.

```
add<<<1, 1>>>(N, x, y)
```

The bracket has the executio configuration. The first is the number of blocks. The second parameter in the bracket is the number of threads in one block.

use ```cudaDeviceSynchonize()``` for the CPU to wait for GPU to finish before accessing.

Now compile the code and run it.

```
$ nvcc add.cu -o add_cuba
$ ./add_cuda
$ nvprof ./add_cuda
```

## run the code on colab

Run the code from https://developer.nvidia.com/blog/even-easier-introduction-cuda/ in [Google colab](https://colab.research.google.com/).

* Step1: create a “New Notebook”

* Step2: choose the runtime. Python3, and T4 GPU

* Step3: in the first code cell, paste the following:

```
!python --version
!nvcc --version
!pip install nvcc4jupyter
%load_ext nvcc4jupyter
```

Then run the cell. Now the cuda environment is set up.

```
Python 3.10.12
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.2, V12.2.140
Build cuda_12.2.r12.2/compiler.33191640_0
Collecting nvcc4jupyter
  Downloading nvcc4jupyter-1.2.1-py3-none-any.whl.metadata (5.1 kB)
Downloading nvcc4jupyter-1.2.1-py3-none-any.whl (10 kB)
Installing collected packages: nvcc4jupyter
Successfully installed nvcc4jupyter-1.2.1
Detected platform "Colab". Running its setup...
Source files will be saved in "/tmp/tmpkajvtdtl".
```

* Step4: paste the following code into the second code cell

The colab environment does not like ```blockIdx.x```. The problem is not blockIdx.x. It is the double-quote is not correct. However, the print could not be seen on the colab.

```
%%cuda
#include <stdio.h>
__global__ void hello(){
 printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}
int main(){
 hello<<<2, 2>>>();
 cudaDeviceSynchronize();
}
```

Notes:

```
%%cuda -c "-l curand"
```

Use the random cuRand library. Check out more examples from https://github.com/gittimos/cuda-colab.

