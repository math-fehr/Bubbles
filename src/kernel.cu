#include "assert_cuda.hpp"
#include "kernel.cuh"

#include "geom.h"
#include "object.h"
#include "camera.h"

#define THREADS_PER_BLOCK 256

// The surface where CUDA will write
surface<void, cudaSurfaceType2D> surf;

// The value of a pixel
struct RGBA {
  unsigned r : 8;
  unsigned g : 8;
  unsigned b : 8;
  unsigned a : 8;
};

__device__ real intersect(Object object, Rayf ray) {
  if(object.type == ObjectType::sphere) {
    return object.sphere.inter(ray);
  } else if(object.type == ObjectType::plane) {
    return object.plane.inter(ray);
  } else {
    return -1.f;
  }
}


/**
 * Entry CUDA kernel. This is the code for one pixel
 */
__global__ void kernel(int counter, Object* objects, unsigned n_objects, Camera camera) {
  // pixel coordinates
  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  int x_pixel = idx % camera.screen_width;
  int y_pixel = idx / camera.screen_width;

  Rayf ray = camera.get_ray(x_pixel, y_pixel);

  RGBA rgbx;
  rgbx.r = 0, rgbx.g=0,rgbx.b=0;

  for(int i = 0; i < n_objects ; ++i) {
    if(intersect(objects[i], ray) >= 0.f) {
      rgbx.r=objects[i].color.r * 255;
      rgbx.g=objects[i].color.g * 255;
      rgbx.b=objects[i].color.b * 255;
    }
  }

  if(idx < camera.screen_height * camera.screen_width) {
    surf2Dwrite(rgbx, surf, x_pixel * sizeof(rgbx), y_pixel, cudaBoundaryModeZero);
  }
}

void kernel_launcher(cudaArray_const_t array, Object* objects, unsigned n_objects, Camera camera) {
  // Count the number of frames displayed
  static unsigned counter = 0;
  counter += 1;

  cuda(BindSurfaceToArray(surf, array));

  const int blocks =
      (camera.screen_width * camera.screen_height + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  if (blocks > 0) {
    kernel<<<blocks, THREADS_PER_BLOCK>>>(counter, objects, n_objects, camera);
  }
}
