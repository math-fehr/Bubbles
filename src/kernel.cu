#include "assert_cuda.hpp"
#include "kernel.cuh"

#define THREADS_PER_BLOCK 128

// The surface where CUDA will write
surface<void, cudaSurfaceType2D> surf;

// The value of a pixel
struct RGBA {
  unsigned r : 8;
  unsigned g : 8;
  unsigned b : 8;
  unsigned a : 8;
};

/**
 * Entry CUDA kernel. This is the code for one pixel
 */
__global__ void kernel(const int width, const int height, const int counter) {
  // pixel coordinates
  const int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  const int x = idx % width;
  const int y = idx / width;

  RGBA rgbx;
  rgbx.r = (char)((float)(y) / (float)(height)*255);
  rgbx.g = (char)((float)(x) / (float)(width)*255);
  rgbx.b = (char)((float)(x + y) / (float)(width + height) * 255);

  surf2Dwrite(rgbx, surf, x * sizeof(rgbx), y, cudaBoundaryModeZero);
}

void kernel_launcher(cudaArray_const_t array, const int width,
                     const int height) {
  // Count the number of frames displayed
  static unsigned counter = 0;
  counter += 1;

  cuda(BindSurfaceToArray(surf, array));

  const int blocks =
      (width * height + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  if (blocks > 0) {
    kernel<<<blocks, THREADS_PER_BLOCK>>>(width, height, counter);
  }
}
